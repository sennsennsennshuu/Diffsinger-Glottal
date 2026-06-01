"""
SFC Breathness -- BrathyAwareLoss

Custom loss module that combines mel-spectrogram L1 loss with
high-frequency emphasis, spectral-tilt awareness, and optional
breathiness/voicing supervision.

Reference: spectral tilt measured via cosine similarity on per-frame
frequency slope vectors rather than MSE, making it scale-invariant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrathyAwareLoss(nn.Module):
    """
    Multi-component loss for breathy-aware mel-spectrogram training.

    Components:
      - mel_l1_loss:          L1 on mel with high-frequency emphasis mask
      - high_freq_loss:       extra weighting on mel bins > 40
      - spectral_tilt_loss:  cosine similarity of frame-wise freq slopes
      - breathiness_loss:    MSE on breathiness prediction
      - voicing_loss:         MSE on voicing prediction

    Forward signature:
        forward(mel_pred, mel_gt, breathiness_pred, breathiness_gt,
                voicing_pred, voicing_gt, mask) -> dict of scalar losses
    """

    def __init__(self,
                 mel_l1_weight: float = 1.0,
                 high_freq_weight: float = 0.5,
                 spectral_tilt_weight: float = 0.3,
                 breathiness_weight: float = 0.3,
                 voicing_weight: float = 0.3):
        super().__init__()

        self.mel_l1_weight = mel_l1_weight
        self.high_freq_weight = high_freq_weight
        self.spectral_tilt_weight = spectral_tilt_weight
        self.breathiness_weight = breathiness_weight
        self.voicing_weight = voicing_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_high_freq_mask(self, n_mels: int, device: torch.device) -> torch.Tensor:
        """
        Linear ramp: weights = 1.0 for bins <= 40,
        ramp from 1.0 up to 1.5 for bins > 40.

        Returns [n_mels] tensor.
        """
        weights = torch.ones(n_mels, device=device)
        ramp_start = 40
        if n_mels > ramp_start:
            ramp_len = n_mels - ramp_start
            ramp = torch.linspace(1.0, 1.5, ramp_len, device=device)
            weights[ramp_start:] = ramp
        return weights

    def _compute_spectral_tilt(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Compute per-frame spectral tilt as the log-magnitude slope
        across mel bins (treated as a proxy for frequency).

        Args:
            mel: [B, n_mels, T]

        Returns:
            slope: [B, T]  -- slope of log-mag vs bin index
        """
        B, n_mels, T = mel.shape
        # Bin indices as the independent variable: [0, 1, 2, ..., n_mels-1]
        bins = torch.arange(n_mels, device=mel.device, dtype=mel.dtype)  # [n_mels]

        # Center bins for numerical stability
        bins_centered = bins - bins.mean()

        # log-magnitude (with small epsilon to avoid log(0))
        log_mel = torch.log(mel.clamp(min=1e-6))

        # For each frame, linear regression slope:
        #   slope = sum((x_i - x_bar) * (y_i - y_bar)) / sum((x_i - x_bar)^2)
        # Operate over the n_mels dimension (dim=1)

        log_mel_centered = log_mel - log_mel.mean(dim=1, keepdim=True)  # [B, n_mels, T]

        numerator = (bins_centered.view(1, n_mels, 1) * log_mel_centered).sum(dim=1)  # [B, T]
        denominator = (bins_centered ** 2).sum()  # scalar

        slope = numerator / denominator.clamp(min=1e-10)  # [B, T]
        return slope

    def _apply_mask(self, loss_per_frame: torch.Tensor, mask: torch.Tensor):
        """
        Apply time-domain mask and return mean loss over valid positions.

        Args:
            loss_per_frame: [B, T] or [B, T, D]  per-frame(/bin) loss values
            mask:           [B, T]  binary (or float) mask

        Returns:
            scalar mean loss over masked positions.
        """
        if mask.dim() < loss_per_frame.dim():
            mask = mask.unsqueeze(-1)
        masked = loss_per_frame * mask
        total = mask.sum()
        if total <= 0:
            return torch.tensor(0.0, device=loss_per_frame.device,
                                dtype=loss_per_frame.dtype)
        return masked.sum() / total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                mel_pred: torch.Tensor,
                mel_gt: torch.Tensor,
                breathiness_pred: torch.Tensor | None,
                breathiness_gt: torch.Tensor | None,
                voicing_pred: torch.Tensor | None,
                voicing_gt: torch.Tensor | None,
                mask: torch.Tensor) -> dict:
        """
        Args:
            mel_pred:          [B, n_mels, T]  predicted mel spectrogram
            mel_gt:            [B, n_mels, T]  ground-truth mel
            breathiness_pred:  [B, T] | None   predicted breathiness
            breathiness_gt:    [B, T] | None   ground-truth breathiness
            voicing_pred:      [B, T] | None   predicted voicing
            voicing_gt:        [B, T] | None   ground-truth voicing
            mask:              [B, T]           frame-level mask

        Returns:
            dict with per-component scalar losses AND 'total_loss'.
        """
        B, n_mels, T = mel_pred.shape
        device = mel_pred.device
        dtype = mel_pred.dtype

        losses = {}

        # ---- 1. Mel L1 Loss (with high-frequency emphasis) ----

        hf_mask = self._build_high_freq_mask(n_mels, device)  # [n_mels]
        hf_mask = hf_mask.view(1, n_mels, 1)  # [1, n_mels, 1]

        # Per time-frequency bin L1
        l1_per_bin = (mel_pred - mel_gt).abs()  # [B, n_mels, T]

        # Apply high-freq weighting
        weighted_l1 = l1_per_bin * hf_mask  # [B, n_mels, T]

        # Mean over mel bins -> per-frame L1
        l1_per_frame = weighted_l1.mean(dim=1)  # [B, T]

        mel_l1_loss = self._apply_mask(l1_per_frame, mask)
        losses['mel_l1_loss'] = mel_l1_loss

        # ---- 2. High-freq loss (extra penalty on bins > 40) ----

        # Additional weighting: only non-zero for bins > 40
        hf_extra = torch.zeros(n_mels, device=device, dtype=dtype)
        if n_mels > 40:
            ramp_len = n_mels - 40
            hf_extra[40:] = torch.linspace(0.0, 0.5, ramp_len, device=device)
        hf_extra = hf_extra.view(1, n_mels, 1)

        hf_per_bin = l1_per_bin * hf_extra  # [B, n_mels, T]
        hf_per_frame = hf_per_bin.mean(dim=1)  # [B, T]

        high_freq_loss = self._apply_mask(hf_per_frame, mask)
        losses['high_freq_loss'] = high_freq_loss

        # ---- 3. Spectral Tilt Loss (cosine similarity) ----

        tilt_pred = self._compute_spectral_tilt(mel_pred)  # [B, T]
        tilt_gt = self._compute_spectral_tilt(mel_gt)      # [B, T]

        # Cosine similarity per batch element:
        #   cos_sim = sum(tilt_pred * tilt_gt) / (||tilt_pred|| * ||tilt_gt||)
        # We compute it per batch over the time dimension, masked.
        tilt_pred_masked = tilt_pred * mask
        tilt_gt_masked = tilt_gt * mask

        dot = (tilt_pred_masked * tilt_gt_masked).sum(dim=1)  # [B]
        norm_pred = (tilt_pred_masked ** 2).sum(dim=1).sqrt()  # [B]
        norm_gt = (tilt_gt_masked ** 2).sum(dim=1).sqrt()      # [B]

        cos_sim = dot / (norm_pred * norm_gt).clamp(min=1e-8)  # [B]
        # Loss: 1 - cosine similarity, averaged over batch
        spectral_tilt_loss = (1.0 - cos_sim).mean()
        spectral_tilt_loss = spectral_tilt_loss.clamp(min=0.0)
        losses['spectral_tilt_loss'] = spectral_tilt_loss

        # ---- 4. Breathiness loss (optional) ----

        if breathiness_pred is not None and breathiness_gt is not None:
            breath_l1 = (breathiness_pred - breathiness_gt).abs()  # [B, T]
            breathiness_loss = self._apply_mask(breath_l1, mask)
        else:
            breathiness_loss = torch.tensor(0.0, device=device, dtype=dtype)
        losses['breathiness_loss'] = breathiness_loss

        # ---- 5. Voicing loss (optional) ----

        if voicing_pred is not None and voicing_gt is not None:
            voicing_l1 = (voicing_pred - voicing_gt).abs()  # [B, T]
            voicing_loss = self._apply_mask(voicing_l1, mask)
        else:
            voicing_loss = torch.tensor(0.0, device=device, dtype=dtype)
        losses['voicing_loss'] = voicing_loss

        # ---- Total loss ----

        total = (
            self.mel_l1_weight * mel_l1_loss +
            self.high_freq_weight * high_freq_loss +
            self.spectral_tilt_weight * spectral_tilt_loss +
            self.breathiness_weight * breathiness_loss +
            self.voicing_weight * voicing_loss
        )
        losses['total_loss'] = total

        return losses
