import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.plot import spec_to_figure
from modules.decoder import ShallowDiffusionOutput
from modules.losses import RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from modules.vocoder import Vocoder
from .pl_module_base import BaseLightningModule


# ============================================================================
# AdaLossNorm — EMA-based per-loss magnitude normalization
# ============================================================================

class AdaLossNorm:
    """EMA magnitude tracker with online normalization.

    Each loss sub-component maintains an EMA of its recent mean.
    normalize() divides by (ema + 1e-6) so all losses contribute ~1.0
    before base_weights are applied.  This eliminates the 16x HNR vs 1x mel
    magnitude mismatch identified in the Q&A doc.
    """
    def __init__(self, ema_decay: float = 0.99):
        self._decay = ema_decay
        self._emas: dict[str, torch.Tensor] = {}

    def normalize(self, name: str, loss_val: torch.Tensor) -> torch.Tensor:
        detached = loss_val.detach()
        if name not in self._emas:
            self._emas[name] = detached.clone()
        else:
            self._emas[name] = (
                self._emas[name] * self._decay
                + detached * (1 - self._decay)
            )
        ema = self._emas[name]
        if ema > 0:
            return loss_val / (ema + 1e-6)
        return loss_val


class AcousticLightningModule(BaseLightningModule):
    def build_model(self) -> DiffSingerAcoustic:
        self.use_sweet_breathy_loss = False  # pure v3
        return DiffSingerAcoustic(self.model_config)

    def configure_optimizers(self):
        """Per-module LR groups based on DiffSinger SFCB knowledge base best practices.

        Multi-module from-scratch training causes gradient competition when all
        modules share a single LR.  The knowledge base (§20 Joint Training)
        recommends differentiated LRs:
          - ConvNeXt aux decoder: 0.5x (4 loss paths pulling simultaneously)
          - Glottal predictor:     2.0x (LF model is small, needs stronger signal)
          - DDSP source / CVAE:    1.0x (from scratch, standard rate)
          - FS2 encoder / LynxNet: 1.0x (base rate)
        """
        opt_cfg = self.training_config.optimizer
        base_lr = opt_cfg.kwargs.get('lr', 0.0002)

        def _make_group(pattern, lr_scale, params):
            filtered = {k: v for k, v in params if pattern in k}
            if not filtered:
                return None
            return {
                'params': list(filtered.values()),
                'lr': base_lr * lr_scale,
            }

        all_params = dict(self.named_parameters()).items()
        groups = []

        # Group 1: ConvNeXt aux decoder (0.5x — 4 loss paths)
        g = _make_group('aux_decoder', opt_cfg.aux_lr_scale, all_params)
        if g:
            groups.append(g)

        # Group 2: Glottal predictor + adapter + projection (2.0x — tiny LF model)
        g = _make_group('glottal_predictor', opt_cfg.glottal_lr_scale, all_params)
        if g:
            groups.append(g)
        g = _make_group('glottal_adapter', opt_cfg.glottal_lr_scale, all_params)
        if g:
            groups.append(g)
        g = _make_group('_glottal_proj', opt_cfg.glottal_lr_scale, all_params)
        if g:
            groups.append(g)

        # Group 3: DDSP source (1.0x — from scratch)
        g = _make_group('ddsp_source', opt_cfg.ddsp_lr_scale, all_params)
        if g:
            groups.append(g)
        g = _make_group('source_filter_fusion', opt_cfg.ddsp_lr_scale, all_params)
        if g:
            groups.append(g)

        # Group 4: Aperiodic CVAE (1.0x — from scratch)
        g = _make_group('aperiodic_cvae', opt_cfg.cvae_lr_scale, all_params)
        if g:
            groups.append(g)
        g = _make_group('periodic_aperiodic_merger', opt_cfg.cvae_lr_scale, all_params)
        if g:
            groups.append(g)

        # Group 5: Everything else (1.0x — FS2, LynxNet, embeddings, speaker, etc.)
        grouped_keys = set()
        for g in groups:
            grouped_keys.update(id(p) for p in g['params'])
        default_group = {
            'params': [v for k, v in all_params if id(v) not in grouped_keys],
            'lr': base_lr,
        }
        groups.append(default_group)

        optimizer = torch.optim.AdamW(
            groups,
            betas=opt_cfg.kwargs.get('betas', [0.9, 0.98]),
            weight_decay=opt_cfg.kwargs.get('weight_decay', 0),
        )
        # Adaptive step_size: total_steps / 8 (≈ 4 decays over full training)
        # This keeps LR above 1e-4 at 120k, preventing starvation
        total_steps = self.training_config.trainer.max_steps
        step_size = self.training_config.lr_scheduler.kwargs.get(
            'step_size',
            max(10000, total_steps // 8)
        )
        gamma = self.training_config.lr_scheduler.kwargs.get('gamma', 0.8)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.training_config.lr_scheduler.unit,
                "frequency": 1,
                "monitor": "total_loss",
                "strict": False,
            }
        }

    def _get_loss_warmup_coef(self, loss_name: str) -> float:
        """Per-loss linear warmup coefficient from config loss_warmup dict.

        Config format: loss_warmup: { name: [start_step, full_step] }
        Returns 0 if before start, 1 if after full, else linear ramp.
        Losses not listed get full weight immediately (coef=1.0).
        """
        cfg = getattr(self.training_config.loss, "sweet_breathy", None)
        if cfg is None:
            return 1.0  # pure v3
        wu = getattr(cfg, 'loss_warmup', {})
        pair = wu.get(loss_name)
        if pair is None:
            return 1.0
        start, full = pair[0], pair[1]
        if self.global_step < start:
            return 0.0
        if self.global_step >= full:
            return 1.0
        return (self.global_step - start) / (full - start)

    # noinspection PyAttributeOutsideInit
    def post_init(self) -> None:
        if self.training_config.validation.use_vocoder:
            self.vocoder = Vocoder(self.training_config.validation.vocoder)
        else:
            self.vocoder = None
        self.logged_gt_wav_indices = set()

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.vocoder is not None:
            self.vocoder.to(self.device)

    def register_losses_and_metrics(self) -> None:
        if self.model_config.spec_decoder.use_shallow_diffusion:
            aux_loss_type = self.training_config.loss.spec_decoder.aux_loss_type
            if aux_loss_type == "L1":
                aux_spec_loss = nn.L1Loss()
            elif aux_loss_type == "L2":
                aux_spec_loss = nn.MSELoss()
            else:
                raise ValueError(f"Invalid spec_decoder.aux_loss_type: {aux_loss_type}")
            self.register_loss("aux_spec_loss", aux_spec_loss)
        main_loss_type = self.training_config.loss.spec_decoder.main_loss_type
        if main_loss_type not in ["L1", "L2"]:
            raise ValueError(f"Invalid spec_decoder.main_loss_type: {main_loss_type}")
        diff_spec_loss = RectifiedFlowLoss(
            loss_type=main_loss_type,
            log_norm=self.training_config.loss.spec_decoder.main_loss_log_norm
        )
        self.register_loss("diff_spec_loss", diff_spec_loss)

        if self.use_sweet_breathy_loss:
            pass

        if self.use_sweet_breathy_loss:
            pass





    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        tokens = sample["tokens"]
        languages = sample["languages"]
        durations = sample["ph_dur"]
        f0 = sample["f0"]
        spk_id = sample["spk_id"]
        mel = sample["mel"]
        key_shift = sample["key_shift"].unsqueeze(1)
        speed = sample["speed"].unsqueeze(1)
        variances = {v_name: sample[v_name] for v_name in self.model_config.embeddings.embedded_variance_names}
        acoustic_output = self.model(
            tokens=tokens, durations=durations, languages=languages, spk_ids=spk_id,
            f0=f0, key_shift=key_shift, speed=speed, spec_gt=mel, infer=infer,
            aperiodic_mel=sample.get('aperiodic_mel', None),
            phase=4,  # all SFC modules active; per-loss warmup handles scheduling
            **variances
        )
        model_out = acoustic_output.decoder_out
        mask = acoustic_output.mask
        glottal_breathiness = acoustic_output.glottal_breathiness
        sfc = {'glottal_b': True, 'ddsp': True, 'cvae': True, 'perceptual': True}
        model_out: ShallowDiffusionOutput
        if infer:
            outputs = {
                "aux_spec": model_out.aux_out,
                "diff_spec": model_out.diff_out,
            }
            if acoustic_output.fused_mel is not None:
                outputs["fused_mel"] = acoustic_output.fused_mel
            if acoustic_output.merged_mel is not None:
                outputs["merged_mel"] = acoustic_output.merged_mel
            if acoustic_output.aperiodic_mel_pred is not None:
                outputs["aperiodic_mel_pred"] = acoustic_output.aperiodic_mel_pred
            return outputs
        else:
            losses = {}
            if model_out.aux_out is not None:
                aux_spec_loss = self.losses["aux_spec_loss"](model_out.aux_out, model_out.norm_gt)
                if torch.isnan(aux_spec_loss) or torch.isinf(aux_spec_loss):
                    aux_spec_loss = torch.nan_to_num(aux_spec_loss, nan=0.0, posinf=0.0, neginf=0.0)
                losses["aux_spec_loss"] = aux_spec_loss * self.training_config.loss.spec_decoder.aux_loss_lambda
            v_pred, v_gt, t = model_out.diff_out
            diff_spec_loss = self.losses["diff_spec_loss"](v_pred, v_gt, t=t, non_padding=mask.unsqueeze(-1).to(t))
            if torch.isnan(diff_spec_loss) or torch.isinf(diff_spec_loss):
                diff_spec_loss = torch.nan_to_num(diff_spec_loss, nan=0.0, posinf=0.0, neginf=0.0)
            losses["diff_spec_loss"] = diff_spec_loss

            if self.use_sweet_breathy_loss and hasattr(self, 'brathy_loss') and model_out.aux_out is not None:
                cfg = self.training_config.loss.sweet_breathy
                # Progressive warmup: linearly ramp brathy loss from 0 at step 0 to
                # full weight at warmup_end (default 32000). Before warmup, brathy loss
                # gradients are heavily attenuated to prevent overfitting-triggered spikes.
                warmup_end = getattr(
                    self.training_config.loss.sweet_breathy, 'warmup_full_step', 64000
                )
                warmup_coef = min(1.0, self.global_step / max(warmup_end, 1))
                breathy_losses = self.brathy_loss(
                    model_out.aux_out.transpose(1, 2),  # [B, T, M] -> [B, M, T]
                    model_out.norm_gt.transpose(1, 2),
                    breathiness_pred=variances.get('breathiness') if variances else None,
                    breathiness_gt=sample.get('breathiness'),
                    voicing_pred=variances.get('voicing') if variances else None,
                    voicing_gt=sample.get('voicing'),
                    mask=mask,
                )
                # BrathyAwareLoss.forward returns a 'total_loss' key (internal weighted sum)
                # plus 5 sub-component keys. Pop internal total to avoid double-counting.
                breathy_total_raw = breathy_losses.pop('total_loss')
                # NaN guard: if any brathy component is NaN, zero it to prevent
                # NaN gradients from destroying the entire model (diff + aux included).
                # brathy_total and sub-components are logged separately so a
                # partial NaN only kills the brathy path, not the whole training.
                if torch.isnan(breathy_total_raw) or torch.isinf(breathy_total_raw):
                    breathy_total_raw = torch.nan_to_num(breathy_total_raw, nan=0.0, posinf=0.0, neginf=0.0)
                losses['brathy_total'] = breathy_total_raw * warmup_coef
                # Log sub-components individually (prefixed with brathy/ for TensorBoard group)
                for k, v in breathy_losses.items():
                    v_safe = v
                    if torch.isnan(v) or torch.isinf(v):
                        v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                    losses[f'brathy/{k}'] = v_safe * warmup_coef

            # ---- Glottal Breathiness Direct Supervision ----
            # GlottalToDiffSingerAdapter maps Rd/OQ/AQ/noise → breathiness [B,T].
            # Compare against ground truth breathiness from audio to provide direct
            # supervision to GlottalParamPredictor (which otherwise only receives
            # weak indirect signal via diff_spec_loss on cond+glottal_cond).
            if glottal_breathiness is not None and sample.get('breathiness') is not None and sfc['glottal_b']:
                glottal_b_loss = F.l1_loss(
                    glottal_breathiness * mask,
                    sample['breathiness'] * mask,
                ) / (mask.sum() + 1e-8) * mask.shape[0]
                glottal_b_loss = torch.nan_to_num(glottal_b_loss, nan=0.0, posinf=0.0, neginf=0.0)
                wu_coef = self._get_loss_warmup_coef('glottal_breathiness')
                losses['glottal_breathiness_loss'] = glottal_b_loss * wu_coef * 0.5
                losses['glottal_breathiness_loss_raw'] = glottal_b_loss * 0.5

            # ---- DDSP Diffusion Hybrid Loss ----
            if (sfc['ddsp']
                and hasattr(self, 'ddsp_diffusion_loss')
                and acoustic_output.ddsp_excitation_mel is not None
                and acoustic_output.fused_mel is not None):
                target_mel = model_out.norm_gt.squeeze(0) if model_out.norm_gt.dim() == 4 else model_out.norm_gt
                ddsp_loss = self.ddsp_diffusion_loss(
                    ddsp_source=(acoustic_output.ddsp_excitation_mel, target_mel),
                    diff_residual=(acoustic_output.fused_mel, target_mel),
                    final_mel=(acoustic_output.fused_mel, target_mel),
                    lpc_coeffs=acoustic_output.ddsp_lpc_coeffs,
                    harmonic_amps=acoustic_output.ddsp_harmonic_amps,
                )
                if not (torch.isnan(ddsp_loss) or torch.isinf(ddsp_loss)):
                    wu = self._get_loss_warmup_coef('ddsp_source')
                    losses['ddsp_diffusion_loss'] = ddsp_loss * wu
                    if wu < 1.0:
                        losses['ddsp_diffusion_loss_raw'] = ddsp_loss

            # ---- Period Singer CVAE Loss ----
            if (sfc['cvae']
                and hasattr(self, 'period_singer_loss')
                and acoustic_output.aperiodic_mel_pred is not None
                and acoustic_output.cvae_kl_loss is not None):
                pred_dict = {
                    'mel_ap_pred': acoustic_output.aperiodic_mel_pred,
                    'merged_mel': acoustic_output.merged_mel,
                    'mu': acoustic_output.cvae_mu,
                    'logvar': acoustic_output.cvae_logvar,
                    'kl_loss': acoustic_output.cvae_kl_loss,
                }
                # Align target tensors to pred temporal dim
                T_pred = acoustic_output.aperiodic_mel_pred.shape[1]
                ap_gt = sample.get('aperiodic_mel', acoustic_output.aperiodic_mel_pred)
                if ap_gt.shape[1] > T_pred:
                    ap_gt = ap_gt[:, :T_pred, :]
                elif ap_gt.shape[1] < T_pred:
                    ap_gt = F.pad(ap_gt, (0, 0, 0, T_pred - ap_gt.shape[1]), mode='replicate')
                norm_gt = model_out.norm_gt
                if norm_gt.dim() == 4:
                    norm_gt = norm_gt.squeeze(0)
                if norm_gt.shape[1] > T_pred:
                    norm_gt = norm_gt[:, :T_pred, :]
                elif norm_gt.shape[1] < T_pred:
                    norm_gt = F.pad(norm_gt, (0, 0, 0, T_pred - norm_gt.shape[1]), mode='replicate')
                target_dict = {
                    'aperiodic_mel': ap_gt,
                    'mel': norm_gt,
                }
                cvae_breathy_loss = self.period_singer_loss(pred_dict, target_dict)
                if not (torch.isnan(cvae_breathy_loss) or torch.isinf(cvae_breathy_loss)):
                    cvae_scale = getattr(cfg, 'period_singer_loss_scale', 0.3)
                    wu = self._get_loss_warmup_coef('cvae_merged')
                    losses['period_singer_loss'] = cvae_breathy_loss * wu * cvae_scale
                    if wu < 1.0:
                        losses['period_singer_loss_raw'] = cvae_breathy_loss * cvae_scale

            # ---- SweetBreathyPerceptualLoss ----
            if sfc['perceptual'] and hasattr(self, 'perceptual_loss'):
                pred_for_perceptual = (
                    acoustic_output.merged_mel
                    if acoustic_output.merged_mel is not None
                    else acoustic_output.fused_mel
                    if acoustic_output.fused_mel is not None
                    else model_out.aux_out
                )
                if pred_for_perceptual is not None:
                    global_step_t = torch.tensor(self.global_step, device=pred_for_perceptual.device, dtype=torch.float32)
                    Rd = acoustic_output.lf_params['Rd'] if acoustic_output.lf_params else None
                    perceptual_losses = self.perceptual_loss(
                        pred_spec=pred_for_perceptual,
                        target_spec=model_out.norm_gt.squeeze(0) if model_out.norm_gt.dim() == 4 else model_out.norm_gt,
                        Rd=Rd,
                        global_step=global_step_t,
                    )
                    p_total = perceptual_losses.pop('total', pred_for_perceptual.sum() * 0.0)
                    if not (torch.isnan(p_total) or torch.isinf(p_total)):
                        perc_scale = getattr(cfg, 'perceptual_loss_scale', 0.3)
                        # Use per-loss warmup for perceptual activation
                        wu = self._get_loss_warmup_coef('perceptual_h1h2')
                        losses['perceptual_loss'] = p_total * wu * perc_scale

                    # Per-sub-component: AdaLossNorm + warmup for MONITORING ONLY.
                    # p_total (the sum of ALL sub-components) is already added as
                    # 'perceptual_loss' for backward pass.  Adding sub-components
                    # again double-counts gradients and causes explosion when
                    # multiple warmup gates fire simultaneously (e.g. at step 40k).
                    _warmup_map = {
                        'h1h2': 'perceptual_h1h2',
                        'hnr': 'perceptual_hnr',
                        'air_energy': 'perceptual_air',
                        'noise_mod': 'perceptual_noise',
                        'fant': 'perceptual_fant',
                    }
                    for k, v in perceptual_losses.items():
                        v_safe = v
                        if torch.isnan(v) or torch.isinf(v):
                            v_safe = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                        # AdaLossNorm: normalize to ~1.0 magnitude
                        if hasattr(self, 'adaloss_norm'):
                            v_safe = self.adaloss_norm.normalize(f'p_{k}', v_safe)
                        # Per-loss warmup
                        wu_name = _warmup_map.get(k)
                        if wu_name:
                            v_safe = v_safe * self._get_loss_warmup_coef(wu_name)
                        # DETACH: logged for monitoring only, backward pass uses
                        # 'perceptual_loss' (p_total * warmup * scale).
                        losses[f'perceptual/{k}'] = v_safe.detach()

            return losses

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            spec_len = self.valid_dataset.info["mel"][data_idx]
            f0_len = self.valid_dataset.info["f0"][data_idx]
            if data_idx >= self.training_config.validation.max_plots:
                continue
            gt_spec = sample["mel"][i, :spec_len]
            f0 = sample["f0"][i, :f0_len]
            if self.vocoder is not None and data_idx not in self.logged_gt_wav_indices:
                gt_wav = self.vocoder.run(gt_spec.unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                self.plot_wav(f"waveform/gt_wav_{data_idx}", gt_wav)
                self.logged_gt_wav_indices.add(data_idx)
            spk_name = self.valid_dataset.info["spk_names"][data_idx]
            item_name = self.valid_dataset.info["item_names"][data_idx]
            title = f"{spk_name} - {item_name}"
            if (aux_spec := outputs.get("aux_spec")) is not None:
                self.plot_spec(
                    tag=f"spectrogram/aux_spec_{data_idx}",
                    spec_gt=sample["mel"][i, :spec_len],
                    spec_pred=aux_spec[i, :spec_len],
                    title=title
                )
                if self.vocoder is not None:
                    aux_wav = self.vocoder.run(outputs["aux_spec"][i].unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                    self.plot_wav(f"waveform/aux_wav_{data_idx}", aux_wav)
            if (diff_spec := outputs.get("diff_spec")) is not None:
                self.plot_spec(
                    tag=f"spectrogram/diff_spec_{data_idx}",
                    spec_gt=sample["mel"][i, :spec_len],
                    spec_pred=diff_spec[i, :spec_len],
                    title=title
                )
                if self.vocoder is not None:
                    diff_wav = self.vocoder.run(outputs["diff_spec"][i].unsqueeze(0), f0=f0.unsqueeze(0)).squeeze(0)
                    self.plot_wav(f"waveform/diff_wav_{data_idx}", diff_wav)

    def plot_spec(self, tag: str, spec_gt: torch.Tensor, spec_pred: torch.Tensor, title=None):
        vmin = self.training_config.validation.spec_vmin
        vmax = self.training_config.validation.spec_vmax
        spec_compare = torch.cat([(spec_pred - spec_gt).abs() + vmin, spec_gt, spec_pred], -1)
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(tag, spec_to_figure(
            spec_compare, vmin=vmin, vmax=vmax, title=title
        ), global_step=self.global_step)

    def plot_wav(self, tag: str, wav: torch.Tensor):
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_audio(
            tag, wav.cpu().numpy(),
            sample_rate=self.vocoder.sample_rate, global_step=self.global_step
        )
