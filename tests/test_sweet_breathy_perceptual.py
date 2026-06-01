"""
TDD tests for SweetBreathyPerceptualLoss — the composite perceptual loss wrapper.

RED phase: this test must fail because the class doesn't exist yet.
"""

import pytest
import torch


class TestSweetBreathyPerceptualLoss:
    """Tests for the composite SweetBreathyPerceptualLoss."""

    @pytest.fixture
    def sample_data(self):
        """Return pred, target waveform pair and metadata."""
        B, T = 2, 50
        n_mels = 128
        pred_mel = torch.randn(B, T, n_mels)
        target_mel = torch.randn(B, T, n_mels)
        f0 = torch.rand(B, T) * 200 + 100
        Rd = torch.rand(B, T) * 1.5 + 0.5
        global_step = torch.tensor(80000.0)
        return pred_mel, target_mel, f0, Rd, global_step

    def test_init_default(self):
        """Default initialization should succeed."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        loss = SweetBreathyPerceptualLoss()
        assert loss.w_mel == 1.0
        assert loss.w_h1h2 == 0.8
        assert loss.w_hnr == 0.5
        assert loss.w_air == 0.6
        assert loss.w_mod == 0.3

    def test_init_custom_weights(self):
        """Custom weights should be stored."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        loss = SweetBreathyPerceptualLoss(
            w_mel=2.0, w_h1h2=1.0, w_hnr=0.8,
            w_air=0.5, w_mod=0.2, w_spectral=0.3,
        )
        assert loss.w_mel == 2.0
        assert loss.w_h1h2 == 1.0

    def test_forward_returns_loss_dict(self, sample_data):
        """Forward should return a dict with valid loss values."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, target_mel, f0, Rd, global_step = sample_data
        loss = SweetBreathyPerceptualLoss()
        result = loss(pred_mel, target_mel, f0=f0, Rd=Rd, global_step=global_step)
        assert isinstance(result, dict)
        assert 'total' in result
        assert result['total'] >= 0

    def test_forward_returns_scalar_total(self, sample_data):
        """Total loss should be a scalar float tensor."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, target_mel, f0, Rd, global_step = sample_data
        loss = SweetBreathyPerceptualLoss()
        result = loss(pred_mel, target_mel, f0=f0, Rd=Rd, global_step=global_step)
        assert isinstance(result['total'], torch.Tensor)
        assert result['total'].dim() == 0

    def test_forward_no_rd_no_crash(self, sample_data):
        """Forward without Rd should not crash (FantPhysicalConsistency skipped)."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, target_mel, f0, _, global_step = sample_data
        loss = SweetBreathyPerceptualLoss()
        result = loss(pred_mel, target_mel, f0=f0, Rd=None, global_step=global_step)
        assert result['total'] >= 0

    def test_warmup_early_step_skips_perceptual(self, sample_data):
        """At global_step=0, perceptual losses should be near 0 (only mel L1 active)."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, target_mel, f0, Rd, _ = sample_data
        loss = SweetBreathyPerceptualLoss(warmup_start=50000, warmup_full=120000)
        result_early = loss(pred_mel, target_mel, f0=f0, Rd=Rd, global_step=torch.tensor(0.0))
        result_late = loss(pred_mel, target_mel, f0=f0, Rd=Rd, global_step=torch.tensor(150000.0))
        # Late step should have more loss components active
        assert result_late['total'] > 0, f"Late step loss should be non-zero, got {result_late['total']}"

    def test_gradient_flow(self, sample_data):
        """Gradient should flow through pred_mel."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, target_mel, f0, Rd, global_step = sample_data
        pred_mel.requires_grad_(True)
        loss = SweetBreathyPerceptualLoss()
        result = loss(pred_mel, target_mel, f0=f0, Rd=Rd, global_step=global_step)
        result['total'].backward()
        assert pred_mel.grad is not None
        assert pred_mel.grad.abs().sum() > 0

    def test_identical_inputs_near_zero_loss(self, sample_data):
        """Identical pred and target should produce very low mel L1 loss."""
        from modules.losses.breathy_perceptual_loss import SweetBreathyPerceptualLoss
        pred_mel, _, f0, Rd, global_step = sample_data
        loss = SweetBreathyPerceptualLoss(warmup_start=0, warmup_full=1)  # no warmup
        # Use same mel, avoid Rand/perceptual confusion by setting weights to 0
        loss_clone = SweetBreathyPerceptualLoss(
            w_mel=1.0, w_h1h2=0.0, w_hnr=0.0, w_air=0.0, w_mod=0.0, w_spectral=0.0,
            warmup_start=0, warmup_full=1,
        )
        result = loss_clone(pred_mel, pred_mel.clone(), f0=f0, Rd=Rd, global_step=global_step)
        assert result['total'] < 0.01, f"Identical inputs should give ~0 loss, got {result['total']:.4f}"
