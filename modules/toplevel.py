import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from utils.hparams import hparams
from lib.config.schema import ModelConfig
from .commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from .commons.tts_modules import LocalUpsample
from .decoder import DiffusionDecoder, ShallowDiffusionOutput
from .embedding import ParameterEmbeddings
from .encoder import LinguisticEncoder, MelodyEncoder
from .normalizer import FeatureNormalizer

__all__ = [
    "DiffSingerAcoustic",
    "DiffSingerVariance",
]

@dataclass
class DiffSingerAcousticOutput:
    decoder_out: 'ShallowDiffusionOutput'
    mask: torch.Tensor
    glottal_breathiness: torch.Tensor | None = None
    ddsp_excitation_mel: torch.Tensor | None = None
    ddsp_harmonic_amps: torch.Tensor | None = None
    ddsp_lpc_coeffs: torch.Tensor | None = None
    ddsp_noise_amp: torch.Tensor | None = None
    fused_mel: torch.Tensor | None = None
    aperiodic_mel_pred: torch.Tensor | None = None
    merged_mel: torch.Tensor | None = None
    cvae_mu: torch.Tensor | None = None
    cvae_logvar: torch.Tensor | None = None
    cvae_kl_loss: torch.Tensor | None = None
    lf_params: dict | None = None

class DiffSingerAcoustic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # --- Legacy ONNX-compatible init path ---
        # The v2 ONNX deployment code (deployment/modules/toplevel.py) passes
        # (vocab_size, out_dims) as positional args.  In that case we bypass
        # all v3-specific initialisation and only set the minimal attributes
        # that the ONNX subclass expects (self.fs2, self.diffusion, etc.).
        if args and not isinstance(args[0], ModelConfig):
            self.fs2 = None
            self.diffusion = None
            self.diffusion_type = hparams.get('diffusion_type', 'reflow')
            self.backbone_type = hparams.get('backbone_type', 'lynxnet')
            self.backbone_args = hparams.get('backbone_args', {})
            self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', True)
            # SFC Breathness (read from hparams for ONNX export)
            sfc_cfg = hparams.get('sfc_breathy', {})
            self.sfc_enabled = sfc_cfg.get('enabled', True)
            self.use_lf_glottal = sfc_cfg.get('use_lf_glottal', True)
            # DDSP/CVAE disabled - glottal only
            self.use_ddsp_hybrid = False
            self.use_aperiodic_cvae = False
            if self.sfc_enabled and self.use_lf_glottal:
                from modules.sweet_breathy.lf_glottal_conditioner import (
                    GlottalParamPredictor, GlottalToDiffSingerAdapter,
                )
                self.glottal_predictor = GlottalParamPredictor(
                    d_cond=hparams.get('hidden_size', 256))
                self.glottal_adapter = GlottalToDiffSingerAdapter()
                self._glottal_proj = nn.Linear(1, hparams.get('hidden_size', 256))
                self._glottal_step = nn.Parameter(torch.zeros(1), requires_grad=False)
                self._glottal_warmup_steps = sfc_cfg.get('glottal_warmup_steps', 5000)
            else:
                self.glottal_predictor = None
                self.glottal_adapter = None
            self.sweet_breathy_enabled = self.sfc_enabled
            return
        config = args[0] if args else kwargs.get('config')
        if config is None:
            raise TypeError("DiffSingerAcoustic requires ModelConfig as first argument")
        # --- V3 config-based init ---
        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()  # tokens to frames
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.speaker_embedding = Embedding(config.num_spk, config.condition_dim)
        self.parameter_embeddings = ParameterEmbeddings(config=config.embeddings)
        # SFC Breathness: LF Glottal Conditioner (default enabled)
        self.condition_dim = config.condition_dim
        self.sweet_breathy_enabled = (
            config.sweet_breathy is not None
            and config.sweet_breathy.enabled
        )
        if self.sweet_breathy_enabled and config.sweet_breathy.use_lf_glottal:
            from modules.sweet_breathy.lf_glottal_conditioner import (
                GlottalParamPredictor, GlottalToDiffSingerAdapter,
            )
            self.glottal_predictor = GlottalParamPredictor(d_cond=config.condition_dim)
            self.glottal_adapter = GlottalToDiffSingerAdapter()
        else:
            self.glottal_predictor = None
            self.glottal_adapter = None
        # Projection for glottal condition (always created to match checkpoint state_dict)
        self._glottal_proj = nn.Linear(1, config.condition_dim)
        # Step counter for glottal_cond injection warmup — predictor starts from
        # random weights at phase 2 onset; linearly ramp cond injection over N steps.
        self.register_buffer('_glottal_step', torch.zeros(1, dtype=torch.long))
        self._glottal_warmup_steps = 100
        self.spec_decoder = DiffusionDecoder(
            sample_dim=config.sample_dim,
            condition_dim=config.condition_dim,
            normalizer=FeatureNormalizer(
                num_channels=config.sample_dim, num_features=1, num_repeats=None,
                squeeze_channel_dim=False, squeeze_feature_dim=True,
                norm_mins=[config.normalization.spec_min],
                norm_maxs=[config.normalization.spec_max],
            ),
            config=config.spec_decoder
        )

    def forward(
            self, tokens, durations, languages, f0, spk_ids=None,
            spk_embed=None, spec_gt=None, infer=True, phase=2, **kwargs
    ) -> DiffSingerAcousticOutput:
        encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self.local_upsample(encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.speaker_embedding(spk_ids)[:, None, :]
            cond = cond + spk_embed
        cond = self.parameter_embeddings(cond, f0=f0, **kwargs)
        glottal_breathiness = None  # for loss supervision
        lf_params = None
        if self.glottal_predictor is not None and phase >= 2:
            # Use frame-level condition (cond) for glottal prediction, not phoneme-level encoder_out
            lf_params = self.glottal_predictor(cond, f0)
            Rd, OQ = lf_params['Rd'], lf_params['OQ']
            adapter_out = self.glottal_adapter(Rd, OQ, lf_params['AQ'], lf_params['noise_amp'])
            # adapter_out = (breathiness, voicing, tension) each [B, T], [0,1]
            glottal_breathiness = adapter_out[0].clamp(0, 1)  # [B, T] -- clamp to valid range
            glottal_cond = torch.stack([glottal_breathiness], dim=-1)  # shape [B, T, 1]
            # Project to condition_dim as residual
            glottal_cond = self._glottal_proj.to(glottal_cond.device)(glottal_cond)
            # Apply glottal blend factor if provided
            glottal_blend = kwargs.pop('glottal_blend', None)
            if glottal_blend is not None:
                glottal_cond = glottal_cond * glottal_blend.unsqueeze(-1)
            # Warmup: glottal predictor starts from random weights at phase 2 onset.
            # Linearly ramp cond injection over _glottal_warmup_steps so the predictor
            # has time to converge before fully influencing the shared condition.
            if not infer:
                gw = min(1.0, self._glottal_step.item() / max(self._glottal_warmup_steps, 1))
                glottal_cond = glottal_cond * gw
                self._glottal_step += 1
            cond = cond + glottal_cond
        decoder_out = self.spec_decoder(condition=cond, sample_gt=spec_gt, infer=infer)

        return DiffSingerAcousticOutput(
            decoder_out=decoder_out, mask=mask,
            glottal_breathiness=glottal_breathiness,
            lf_params=lf_params,
        )

class DiffSingerVariance(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.predict_pitch = config.prediction.predict_pitch
        var_norm_mins = []
        var_norm_maxs = []
        var_clip_mins = []
        var_clip_maxs = []
        variance_list = config.prediction.predicted_variance_names
        if config.prediction.predict_energy:
            var_norm_mins.append(config.normalization.energy_db_min)
            var_norm_maxs.append(config.normalization.energy_db_max)
            var_clip_mins.append(config.normalization.energy_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_breathiness:
            var_norm_mins.append(config.normalization.breathiness_db_min)
            var_norm_maxs.append(config.normalization.breathiness_db_max)
            var_clip_mins.append(config.normalization.breathiness_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_voicing:
            var_norm_mins.append(config.normalization.voicing_db_min)
            var_norm_maxs.append(config.normalization.voicing_db_max)
            var_clip_mins.append(config.normalization.voicing_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_tension:
            var_norm_mins.append(config.normalization.tension_logit_min)
            var_norm_maxs.append(config.normalization.tension_logit_max)
            var_clip_mins.append(config.normalization.tension_logit_min)
            var_clip_maxs.append(config.normalization.tension_logit_max)
        self.predict_variances = len(variance_list) > 0
        self.variance_list = variance_list
        if not self.predict_pitch and not self.predict_variances:
            raise ValueError("Nothing to predict.")

        self.linguistic_encoder = LinguisticEncoder(config=config.linguistic_encoder)
        self.local_upsample = LocalUpsample()
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.spk_embed = Embedding(config.num_spk, config.condition_dim)
        if self.predict_pitch:
            self.melody_encoder = MelodyEncoder(config=config.melody_encoder)
            self.pitch_predictor = DiffusionDecoder(
                sample_dim=config.normalization.pitch_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=1, num_repeats=config.normalization.pitch_repeat_bins,
                    squeeze_channel_dim=True, squeeze_feature_dim=True,
                    norm_mins=[config.normalization.pitd_norm_min],
                    norm_maxs=[config.normalization.pitd_norm_max],
                    clip_mins=[config.normalization.pitd_clip_min],
                    clip_maxs=[config.normalization.pitd_clip_max],
                ),
                config=config.pitch_predictor
            )
        if self.predict_variances:
            total_repeat_bins = config.normalization.variance_total_repeat_bins
            if total_repeat_bins % len(self.variance_list) != 0:
                raise ValueError(
                    f"variance_total_repeat_bins must be divisible by "
                    f"number of variances ({len(self.variance_list)})."
                )
            self.pitch_embedding = Linear(1, config.condition_dim)
            self.variance_predictor = DiffusionDecoder(
                sample_dim=total_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=len(self.variance_list),
                    num_repeats=total_repeat_bins // len(self.variance_list),
                    squeeze_channel_dim=True, squeeze_feature_dim=False,
                    norm_mins=var_norm_mins, norm_maxs=var_norm_maxs,
                    clip_mins=var_clip_mins, clip_maxs=var_clip_maxs,
                ),
                config=config.variance_predictor
            )

    def forward(
            self, tokens, durations, languages, spk_ids=None, spk_embed=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None,
            base_pitch=None, pitch=None,
            infer=True, **kwargs
    ):
        linguistic_encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond, mask = self.local_upsample(linguistic_encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.spk_embed(spk_ids)[:, None, :]
            cond = cond + spk_embed
        if self.predict_pitch:
            melody_encoder_out = self.melody_encoder(
                note_midi=note_midi, note_rest=note_rest, note_dur=note_dur, glide=note_glide
            )
            # TODO: add pitch retaking and expressiveness
            pitch_cond = cond + self.local_upsample(melody_encoder_out, ups=note_dur)[0]
            pitch_predictor_out = self.pitch_predictor(condition=pitch_cond, sample_gt=pitch - base_pitch, infer=infer)
            pitch_predictor_out = pitch_predictor_out.diff_out  # no shallow diffusion yet
            if infer:
                pitch_predictor_out = pitch_predictor_out + base_pitch
        else:
            pitch_predictor_out = None
        if self.predict_variances:
            variance_cond = cond + self.pitch_embedding(pitch[:, :, None])
            variance_predictor_out = self.variance_predictor(
                condition=variance_cond,
                sample_gt=[kwargs.get(v_name) for v_name in self.variance_list],
                infer=infer
            )
            variance_predictor_out = variance_predictor_out.diff_out  # no shallow diffusion yet
        else:
            variance_predictor_out = None
        return pitch_predictor_out, variance_predictor_out, mask
