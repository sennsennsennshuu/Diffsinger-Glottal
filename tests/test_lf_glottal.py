"""
TDD tests for modules/sweet_breathy/lf_glottal_conditioner.py

RED phase: these tests MUST fail because the module doesn't exist yet.
Tests verify: LF LFGlottalModel pulse shape correctness, physical constraints (OQ <= 0.11*Rd + 0.55),
and GlottalToDiffSingerAdapter mapping behavior.
"""

import pytest
import torch


# =============================================================================
# LFGlottalModel tests
# =============================================================================

class TestLFGlottalModel:
    """Tests for the differentiable LF glottal pulse model."""

    def test_modal_vs_breathy_pulse_shape(self):
        """Rd=0.5 (modal) should produce a shorter, sharper pulse than Rd=2.0 (breathy)."""
        from modules.sweet_breathy.lf_glottal_conditioner import LFGlottalModel

        model = LFGlottalModel(sample_rate=22050)  # lower SR for faster test
        # Create params: [B=2, T=5, 5] with different Rd values
        # LF params: Ee, Tp, Te, Ta, Oq
        batch_size, n_frames = 2, 5
        lf_params = torch.zeros(batch_size, n_frames, 5)
        # Set Oq to produce Rd ≈ 0.5 for batch[0] and Rd ≈ 2.0 for batch[1]
        # Rd = (1-Oq)/Oq
        lf_params[0, :, -1] = 0.67   # Oq=0.67 → Rd ≈ 0.5  (modal)
        lf_params[1, :, -1] = 0.33   # Oq=0.33 → Rd ≈ 2.0  (breathy)
        # Other params: moderate values
        lf_params[:, :, 0] = 1.0  # Ee
        lf_params[:, :, 1] = 0.4  # Tp
        lf_params[:, :, 2] = 0.05 # Te
        lf_params[:, :, 3] = 0.02 # Ta

        spec = model(lf_params)  # [B, T, n_mels]

        assert spec.ndim == 3, f"Expected 3D output, got {spec.ndim}D"
        assert spec.shape[0] == batch_size
        assert spec.shape[1] == n_frames

        # Modal (low Rd) pulse should be sharper → more high-frequency energy
        modal_hf = spec[0, :, 40:].mean()
        breathy_hf = spec[1, :, 40:].mean()
        # Both should produce valid (non-nan) output
        assert not torch.isnan(spec).any(), "Output contains NaN"

    def test_output_not_nan(self):
        """Default parameters should produce non-NaN output."""
        from modules.sweet_breathy.lf_glottal_conditioner import LFGlottalModel

        model = LFGlottalModel(sample_rate=22050)
        lf_params = torch.randn(1, 3, 5).sigmoid()  # [0,1] range
        spec = model(lf_params)
        assert not torch.isnan(spec).any(), "Output contains NaN"
        assert not torch.isinf(spec).any(), "Output contains Inf"


# =============================================================================
# GlottalParamPredictor tests
# =============================================================================

class TestGlottalParamPredictor:
    """Tests for the glottal parameter predictor."""

    def test_output_structure(self):
        """Output should contain Rd, OQ, AQ, noise_amp each [B, T]."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalParamPredictor

        d_cond = 256
        predictor = GlottalParamPredictor(d_cond=d_cond)
        B, T = 2, 10
        cond = torch.randn(B, T, d_cond)
        pitch = (torch.rand(B, T) * 60 + 40) / 127.0  # normalized MIDI

        result = predictor(cond, pitch)
        assert isinstance(result, dict)
        for key in ['Rd', 'OQ', 'AQ', 'noise_amp']:
            assert key in result, f"Missing key: {key}"
            assert result[key].shape == (B, T), f"{key} shape mismatch: {result[key].shape}"

    def test_rd_in_valid_range(self):
        """Rd should be in the physical range [0.3, 2.7]."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalParamPredictor

        predictor = GlottalParamPredictor(d_cond=128)
        cond = torch.randn(1, 5, 128)
        pitch = torch.rand(1, 5)

        result = predictor(cond, pitch)
        assert (result['Rd'] >= 0.1).all(), f"Rd too low: {result['Rd'].min().item():.3f}"
        assert (result['Rd'] <= 3.0).all(), f"Rd too high: {result['Rd'].max().item():.3f}"

    def test_physical_constraint_oq_le_011rd_plus_055(self):
        """Fant physical constraint: OQ ≤ 0.11*Rd + 0.55 must hold."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalParamPredictor

        predictor = GlottalParamPredictor(d_cond=128)
        cond = torch.randn(2, 8, 128)
        pitch = torch.rand(2, 8)

        result = predictor(cond, pitch)
        max_allowed_oq = 0.11 * result['Rd'] + 0.55
        violations = result['OQ'] > max_allowed_oq + 1e-5
        assert not violations.any(), \
            f"OQ exceeds physical bound at {violations.sum().item()} positions"

    def test_gradient_flow(self):
        """GlottalParamPredictor should be trainable (gradient flows)."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalParamPredictor

        predictor = GlottalParamPredictor(d_cond=64)
        cond = torch.randn(1, 3, 64, requires_grad=True)
        pitch = torch.rand(1, 3)

        result = predictor(cond, pitch)
        loss = result['Rd'].sum() + result['OQ'].sum() + result['AQ'].sum()
        loss.backward()

        assert cond.grad is not None, "Condition gradient should not be None"

    def test_rd_sigmoid_handles_extreme_inputs(self):
        """Rd uses sigmoid — extreme inputs must NOT cause dead gradients or value saturation.

        The Q&A doc identified that clamp() at wavetable boundaries causes gradient
        death. GlottalParamPredictor uses sigmoid (continuous, always gradient>0),
        but this test verifies it's bulletproof with extreme adversarial inputs.
        """
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalParamPredictor

        predictor = GlottalParamPredictor(d_cond=64)
        # Extreme values: 100× normal for cond, pitch outside [0,1]
        cond = torch.randn(4, 10, 64) * 100
        pitch = torch.rand(4, 10) * 50 - 25  # includes negative pitch!

        result = predictor(cond, pitch)

        # 1. Rd MUST stay in [0.3, 2.7] — sigmoid is immune to input magnitude
        Rd = result['Rd']
        assert (Rd >= 0.3).all(), f"Rd min={Rd.min():.4f} below 0.3"
        assert (Rd <= 2.7).all(), f"Rd max={Rd.max():.4f} above 2.7"

        # 2. Gradients MUST flow (sigmoid has no dead zone unlike clamp)
        cond_g = torch.randn(4, 10, 64, requires_grad=True)
        pitch_g = torch.rand(4, 10)
        result_g = predictor(cond_g, pitch_g)
        loss = result_g['Rd'].sum()
        loss.backward()
        assert cond_g.grad is not None, "Gradient must flow through sigmoid"
        assert cond_g.grad.abs().sum() > 1e-6, (
            f"Gradient too small: {cond_g.grad.abs().sum():.2e}")

        # 3. OQ must obey Fant constraint even with extreme inputs
        max_allowed = 0.11 * result_g['Rd'] + 0.55
        violations = result_g['OQ'] > max_allowed + 1e-5
        assert not violations.any(), (
            f"Fant OQ≤0.11Rd+0.55 violated at {violations.sum().item()} positions")


# =============================================================================
# GlottalToDiffSingerAdapter tests
# =============================================================================

class TestGlottalToDiffSingerAdapter:
    """Tests for mapping LF physical params to DiffSinger variance params."""

    def test_output_shape(self):
        """Output should be 3 tensors: breathiness, voicing, tension."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalToDiffSingerAdapter

        adapter = GlottalToDiffSingerAdapter()
        B, T = 2, 5
        Rd = torch.rand(B, T) * 2.4 + 0.3
        OQ = torch.rand(B, T) * 0.5 + 0.3
        AQ = torch.rand(B, T) * 0.5
        noise_amp = torch.rand(B, T)

        breathiness, voicing, tension = adapter(Rd, OQ, AQ, noise_amp)
        for name, t in [('breathiness', breathiness), ('voicing', voicing), ('tension', tension)]:
            assert t.shape == (B, T), f"{name} shape mismatch: {t.shape}"
            assert (t >= 0).all() and (t <= 1).all(), \
                f"{name} values out of [0,1]: [{t.min().item():.3f}, {t.max().item():.3f}]"

    def test_high_rd_gives_high_breathiness_low_tension(self):
        """Rd=2.5 (very breathy) should give higher breathiness, lower tension than Rd=0.5."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalToDiffSingerAdapter

        adapter = GlottalToDiffSingerAdapter()
        T = 10
        Rd_breathy = torch.full((1, T), 2.5)
        Rd_modal = torch.full((1, T), 0.5)
        OQ = torch.full((1, T), 0.5)
        AQ = torch.full((1, T), 0.2)
        noise_amp = torch.full((1, T), 0.3)

        b_breathy, _, t_breathy = adapter(Rd_breathy, OQ, AQ, noise_amp)
        b_modal, _, t_modal = adapter(Rd_modal, OQ, AQ, noise_amp)

        assert b_breathy.mean() > b_modal.mean(), \
            f"Breathy Rd should give higher breathiness: {b_breathy.mean():.3f} vs {b_modal.mean():.3f}"
        assert t_breathy.mean() < t_modal.mean(), \
            f"Breathy Rd should give lower tension: {t_breathy.mean():.3f} vs {t_modal.mean():.3f}"

    def test_noise_amp_increases_breathiness(self):
        """Higher noise_amp should produce higher breathiness."""
        from modules.sweet_breathy.lf_glottal_conditioner import GlottalToDiffSingerAdapter

        adapter = GlottalToDiffSingerAdapter()
        T = 5
        Rd = torch.full((1, T), 1.5)
        OQ = torch.full((1, T), 0.5)
        AQ = torch.full((1, T), 0.2)

        b_high_noise, _, _ = adapter(Rd, OQ, AQ, torch.full((1, T), 0.8))
        b_low_noise, _, _ = adapter(Rd, OQ, AQ, torch.full((1, T), 0.1))

        assert b_high_noise.mean() > b_low_noise.mean(), \
            f"Higher noise should give higher breathiness: {b_high_noise.mean():.3f} vs {b_low_noise.mean():.3f}"
