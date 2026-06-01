"""
SFC Breathness — Module 1: LF Glottal Parameterization

Differentiable LF (Liljencrants-Fant) glottal pulse model, glottal parameter
predictor from phoneme features, and adapter to DiffSinger variance space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# LF Glottal Model
# =============================================================================

class LFGlottalModel(nn.Module):
    """
    Differentiable LF glottal pulse model.

    Input: LF parameters (Ee, Tp, Te, Ta, Oq) per frame
    Output: glottal pulse spectrogram [B, T, n_mels]

    Rd = (1 - Oq) / Oq controls pulse shape:
      - Rd ~ 0.5: modal (short, sharp pulse)
      - Rd ~ 2.0: breathy (long, soft pulse)
    """

    def __init__(self, sample_rate: int = 44100, n_mels: int = 128, fft_size: int = 1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        self.pulse_length = fft_size

        # Precompute STFT window
        self.register_buffer("window", torch.hann_window(fft_size))

        # Mel filterbank approximation: linear ramp
        mel_weights = torch.linspace(1.0, 1024.0, n_mels).pow(0.5) / 32.0
        self.register_buffer("mel_weights", mel_weights.view(1, 1, n_mels))

    def _generate_pulse(self, Ee, Tp, Te, Ta, Oq):
        """
        Generate a single LF glottal pulse of length pulse_length.

        LF model:
          Open phase (t < Te):  Ee * exp(alpha*t) * sin(wg*t)
          Return phase (Te <= t < Tc): correction to zero

        Simplified version: Oq controls pulse width, Ee controls amplitude.
        """
        device = Ee.device
        B, T = Ee.shape
        L = self.pulse_length
        # Sample positions normalized to [0, 1]
        t = torch.linspace(0, 1.0, L, device=device).view(1, 1, L)

        # Open quotient determines pulse width fraction
        Oq = Oq.clamp(0.1, 0.9)
        open_phase = t < Oq.unsqueeze(-1)

        # Within open phase: damped sinusoid
        # Frequency of glottal pulse ~ 1/(2*Tp) normalized
        wg = math.pi / (Tp.unsqueeze(-1).clamp(0.05, 0.5) + 1e-5)
        alpha = -3.0 / (Oq.unsqueeze(-1) + 1e-5)

        pulse = torch.zeros(B, T, L, device=device)
        open_region = open_phase.float()

        # Generate damped sine within open phase
        envelope = torch.exp(alpha * t) * open_region
        sine = torch.sin(wg * t * math.pi) * open_region
        pulse = Ee.unsqueeze(-1) * envelope * sine

        # Normalize
        peak = pulse.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        pulse = pulse / peak * 0.5

        return pulse

    def forward(self, lf_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lf_params: [B, T, 5] tensor with (Ee, Tp, Te, Ta, Oq) per frame

        Returns:
            glottal_spec: [B, T, n_mels] mel spectrogram
        """
        B, T, _ = lf_params.shape
        device = lf_params.device

        # Extract parameters
        Ee = lf_params[..., 0].clamp(0.0, 2.0)       # excitation strength
        Tp = lf_params[..., 1].clamp(0.05, 0.45)     # peak time (normalized)
        Te = lf_params[..., 2].clamp(0.01, 0.2)       # excitation end
        Ta = lf_params[..., 3].clamp(0.001, 0.1)      # return time constant
        Oq = lf_params[..., 4].clamp(0.15, 0.85)      # open quotient

        # Generate pulse for each frame
        pulses = self._generate_pulse(Ee, Tp, Te, Ta, Oq)  # [B, T, L]

        # STFT
        pulses_flat = pulses.reshape(B * T, self.pulse_length)
        pulses_flat = pulses_flat * self.window.unsqueeze(0)

        spec = torch.stft(
            pulses_flat,
            n_fft=self.fft_size,
            hop_length=self.fft_size,  # single frame per pulse
            win_length=self.fft_size,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()

        # Take magnitude at positive frequencies
        spec = spec[:, :self.fft_size // 2 + 1]  # [B*T, freq_bins]
        spec = spec.reshape(B, T, -1)

        # Convert to mel-like representation (simplified)
        # Take first n_mels bins weighted
        spec_out = spec[..., :self.n_mels] * self.mel_weights
        # Log scale
        spec_out = torch.log1p(spec_out * 10.0)

        return spec_out


# =============================================================================
# Glottal Param Predictor
# =============================================================================

class GlottalParamPredictor(nn.Module):
    """
    Predicts LF glottal parameters (Rd, OQ, AQ, noise_amp) from
    condition embedding and pitch.

    Uses a lightweight 2-layer transformer encoder + 4 independent prediction heads.

    Physical constraint: OQ = min(OQ_pred, 0.11*Rd + 0.55)  (Fant 1997)
    """

    def __init__(self, d_cond: int = 256, d_hidden: int = 128, nhead: int = 4):
        super().__init__()
        self.d_cond = d_cond
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(d_cond + 1, d_hidden)  # +1 for pitch

        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=nhead, dim_feedforward=d_hidden * 2,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Prediction heads
        self.rd_head = nn.Linear(d_hidden, 1)
        self.oq_head = nn.Linear(d_hidden, 1)
        self.aq_head = nn.Linear(d_hidden, 1)
        self.noise_amp_head = nn.Linear(d_hidden, 1)

    def forward(self, cond: torch.Tensor, pitch: torch.Tensor) -> dict:
        """
        Args:
            cond: [B, T, d_cond] condition embedding from linguistic encoder
            pitch: [B, T] normalized pitch (0-1 range)

        Returns:
            dict with Rd, OQ, AQ, noise_amp: each [B, T]
        """
        B, T = cond.shape[:2]
        # Ensure pitch is [B, T, 1]
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if pitch.shape[-1] != 1:
            pitch = pitch.unsqueeze(-1)

        # Concatenate
        x = torch.cat([cond, pitch], dim=-1)
        x = self.input_proj(x)

        # Transformer
        x = self.transformer(x)  # [B, T, d_hidden]

        # Predict
        Rd_pred = self.rd_head(x).squeeze(-1)  # [B, T]
        OQ_pred = self.oq_head(x).squeeze(-1)
        AQ_pred = self.aq_head(x).squeeze(-1)
        noise_amp_pred = self.noise_amp_head(x).squeeze(-1)

        # Apply activation → physical range
        Rd = torch.sigmoid(Rd_pred) * 2.4 + 0.3     # Rd ∈ [0.3, 2.7]
        AQ = torch.sigmoid(AQ_pred) * 0.5            # AQ ∈ [0, 0.5]
        noise_amp = torch.sigmoid(noise_amp_pred)     # noise_amp ∈ [0, 1]

        # OQ with physical constraint: OQ ≤ 0.11*Rd + 0.55
        OQ_raw = torch.sigmoid(OQ_pred) * 0.5 + 0.3  # ∈ [0.3, 0.8]
        max_allowed_OQ = 0.11 * Rd + 0.55
        OQ = torch.min(OQ_raw, max_allowed_OQ)

        return {'Rd': Rd, 'OQ': OQ, 'AQ': AQ, 'noise_amp': noise_amp}


# =============================================================================
# Glottal to DiffSinger Adapter
# =============================================================================

class GlottalToDiffSingerAdapter(nn.Module):
    """
    Maps physical LF glottal parameters to DiffSinger variance space
    (breathiness, voicing, tension) via learnable sigmoid mappings.

    breathiness = σ(w_b0 * Rd/2.7 + w_b1 * noise_amp + w_b2)
    voicing     = σ(w_v0 * OQ   + w_v1 * (1 - Rd/2.7) + w_v2)
    tension     = σ(w_t0 * (2.7-Rd)/2.4 + w_t1 * (0.5-AQ) + w_t2)
    """

    def __init__(self):
        super().__init__()
        # Breathiness weights (3 params)
        self.w_b = nn.Parameter(torch.tensor([0.5, 0.3, 0.0]))
        # Voicing weights (3 params)
        self.w_v = nn.Parameter(torch.tensor([0.3, 0.4, 0.0]))
        # Tension weights (3 params)
        self.w_t = nn.Parameter(torch.tensor([0.4, 0.2, 0.0]))

    def forward(self, Rd: torch.Tensor, OQ: torch.Tensor,
                AQ: torch.Tensor, noise_amp: torch.Tensor):
        """
        Args:
            Rd: [B, T] glottal resistance parameter
            OQ: [B, T] open quotient
            AQ: [B, T] amplitude quotient
            noise_amp: [B, T] noise amplitude

        Returns:
            breathiness, voicing, tension: each [B, T] in [0, 1]
        """
        # Softmax to ensure positive contribution weights
        w_b = F.softplus(self.w_b)
        w_v = F.softplus(self.w_v)
        w_t = F.softplus(self.w_t)

        Rd_norm = Rd / 2.7

        # Breathiness: higher Rd and noise → more breathy
        breathiness = torch.sigmoid(
            w_b[0] * Rd_norm + w_b[1] * noise_amp + w_b[2]
        )

        # Voicing: higher OQ and lower Rd → more modal (voiced)
        voicing = torch.sigmoid(
            w_v[0] * OQ + w_v[1] * (1.0 - Rd_norm) + w_v[2]
        )

        # Tension: lower Rd and lower AQ → more tense/pressed
        tension = torch.sigmoid(
            w_t[0] * (1.0 - Rd_norm) + w_t[1] * (0.5 - AQ) + w_t[2]
        )

        return breathiness, voicing, tension
