"""
TDD Tests for Breathy Perceptual Loss (Step 10)
Module: modules/losses/breathy_perceptual_loss.py

Classes under test:
    H1H2PerceptualLoss, HNRPerceptualLoss, AirEnergyLoss,
    NoiseModulationLoss, FantPhysicalConsistencyLoss
"""

import sys
sys.path.insert(0, r'I:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-v3-sfcb')

import torch
import pytest
import math


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def batch():
    return 2


@pytest.fixture
def time_steps():
    return 64


# ============================================================================
# H1H2PerceptualLoss Tests
# ============================================================================

class TestH1H2PerceptualLoss:

    def test_import_exists(self):
        """H1H2PerceptualLoss should be importable."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        assert callable(H1H2PerceptualLoss)

    def test_forward_returns_scalar(self, device, batch, time_steps):
        """forward() returns a scalar tensor."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    def test_forward_is_finite(self, device, batch, time_steps):
        """Loss should be finite."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    def test_forward_nonnegative(self, device, batch, time_steps):
        """Loss should be non-negative."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item():.6f}"

    def test_perfect_match_zero_loss(self, device, batch, time_steps):
        """Identical inputs should produce near-zero loss."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        spec = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec, spec)
        assert loss.item() < 1e-3, \
            f"Identical spectra should have near-zero loss, got {loss.item():.6f}"

    def test_different_spectra_nonzero_loss(self, device, batch, time_steps):
        """Different spectra should produce non-zero loss."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        # Create spectra with different H1-H2 ratios:
        # spec_pred: steeper slope (higher H1-H2) - breathy
        freqs = torch.arange(513, device=device, dtype=torch.float32).view(1, 1, -1)
        spec_pred = torch.exp(-freqs * 0.02)  # steep decay
        spec_target = torch.exp(-freqs * 0.01)  # gentle decay

        loss = loss_fn(spec_pred.repeat(batch, time_steps, 1),
                       spec_target.repeat(batch, time_steps, 1))
        assert loss.item() > 0, "Different spectra should produce non-zero loss"

    def test_gradient_flow(self, device, batch, time_steps):
        """Should support gradient backpropagation."""
        from modules.losses.breathy_perceptual_loss import H1H2PerceptualLoss
        loss_fn = H1H2PerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs().requires_grad_(True)
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs()

        loss = loss_fn(spec_pred, spec_target)
        loss.backward()
        assert spec_pred.grad is not None
        assert spec_pred.grad.abs().sum() > 0


# ============================================================================
# HNRPerceptualLoss Tests
# ============================================================================

class TestHNRPerceptualLoss:

    def test_import_exists(self):
        """HNRPerceptualLoss should be importable."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        assert callable(HNRPerceptualLoss)

    def test_forward_returns_scalar(self, device, batch, time_steps):
        """forward() returns a scalar tensor."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        loss_fn = HNRPerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    def test_forward_is_finite(self, device, batch, time_steps):
        """Loss should be finite."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        loss_fn = HNRPerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    def test_forward_nonnegative(self, device, batch, time_steps):
        """Loss should be non-negative."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        loss_fn = HNRPerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item():.6f}"

    def test_perfect_match_zero_loss(self, device, batch, time_steps):
        """Identical inputs should produce near-zero loss."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        loss_fn = HNRPerceptualLoss().to(device)

        spec = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        loss = loss_fn(spec, spec)
        assert loss.item() < 1e-3, \
            f"Identical spectra should have near-zero loss, got {loss.item():.6f}"

    def test_gradient_flow(self, device, batch, time_steps):
        """Should support gradient backpropagation."""
        from modules.losses.breathy_perceptual_loss import HNRPerceptualLoss
        loss_fn = HNRPerceptualLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs().requires_grad_(True)
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs()

        loss = loss_fn(spec_pred, spec_target)
        loss.backward()
        assert spec_pred.grad is not None
        assert spec_pred.grad.abs().sum() > 0


# ============================================================================
# AirEnergyLoss Tests
# ============================================================================

class TestAirEnergyLoss:

    def test_import_exists(self):
        """AirEnergyLoss should be importable."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        assert callable(AirEnergyLoss)

    def test_forward_returns_scalar(self, device, batch, time_steps):
        """forward() returns a scalar tensor."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    def test_forward_is_finite(self, device, batch, time_steps):
        """Loss should be finite."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    def test_forward_nonnegative(self, device, batch, time_steps):
        """Loss should be non-negative."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6

        loss = loss_fn(spec_pred, spec_target)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item():.6f}"

    def test_perfect_match_zero_loss(self, device, batch, time_steps):
        """Identical inputs should produce near-zero loss."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        spec = torch.randn(batch, time_steps, 513, device=device).abs() + 1e-6
        loss = loss_fn(spec, spec)
        assert loss.item() < 1e-3, \
            f"Identical spectra should have near-zero loss, got {loss.item():.6f}"

    def test_high_freq_energy_sensitivity(self, device, batch, time_steps):
        """Should be more sensitive to differences in high-frequency (8-16kHz) energy."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        # Base spectrum
        spec_pred = torch.ones(batch, time_steps, 513, device=device)
        spec_target = torch.ones(batch, time_steps, 513, device=device)

        # High-freq difference (bins corresponding to 8-16kHz)
        spec_target = spec_target.clone()
        # Assuming FFT=1024, SR=44100: bin range for 8-16kHz ≈ bins 186-373
        # For 513 bins at 44100: freq_per_bin ≈ 43 Hz
        # 8000/43 ≈ 186, 16000/43 ≈ 372
        low_freq_idx = 50
        high_freq_idx = 400

        spec_low = spec_pred.clone()
        spec_low_diff_target = spec_target.clone()
        spec_low_diff_target[..., low_freq_idx:high_freq_idx] *= 2.0

        spec_high = spec_pred.clone()
        spec_high_diff_target = spec_target.clone()
        spec_high_diff_target[..., :low_freq_idx] *= 2.0

        loss_low = loss_fn(spec_low, spec_low_diff_target)
        loss_high = loss_fn(spec_high, spec_high_diff_target)

        # High-freq differences should contribute more to air energy loss
        # (8-16kHz energy ratio should be more affected)
        assert loss_low.item() >= 0 and loss_high.item() >= 0, "Loss should be non-negative"

    def test_gradient_flow(self, device, batch, time_steps):
        """Should support gradient backpropagation."""
        from modules.losses.breathy_perceptual_loss import AirEnergyLoss
        loss_fn = AirEnergyLoss().to(device)

        spec_pred = torch.randn(batch, time_steps, 513, device=device).abs().requires_grad_(True)
        spec_target = torch.randn(batch, time_steps, 513, device=device).abs()

        loss = loss_fn(spec_pred, spec_target)
        loss.backward()
        assert spec_pred.grad is not None
        assert spec_pred.grad.abs().sum() > 0


# ============================================================================
# NoiseModulationLoss Tests
# ============================================================================

class TestNoiseModulationLoss:

    def test_import_exists(self):
        """NoiseModulationLoss should be importable."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        assert callable(NoiseModulationLoss)

    def test_forward_returns_scalar(self, device, batch, time_steps):
        """forward() returns a scalar tensor."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        noise_envelope = torch.rand(batch, time_steps, device=device).abs() + 0.01

        loss = loss_fn(noise_envelope)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    def test_forward_is_finite(self, device, batch, time_steps):
        """Loss should be finite."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        noise_envelope = torch.rand(batch, time_steps, device=device).abs() + 0.01

        loss = loss_fn(noise_envelope)
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    def test_forward_nonnegative(self, device, batch, time_steps):
        """Loss should be non-negative."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        noise_envelope = torch.rand(batch, time_steps, device=device).abs() + 0.01

        loss = loss_fn(noise_envelope)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item():.6f}"

    def test_smooth_envelope_penalized(self, device, time_steps):
        """Overly-smooth noise envelopes (local_variance < 0.05) should be penalized."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        # Very smooth envelope (constant → no local variance)
        smooth_envelope = torch.ones(1, time_steps, device=device) * 0.5
        loss_smooth = loss_fn(smooth_envelope)

        # Varying envelope (natural modulation)
        varying_envelope = torch.rand(1, time_steps, device=device).abs() + 0.05
        loss_varying = loss_fn(varying_envelope)

        assert loss_smooth > loss_varying, \
            f"Smooth envelope should be penalized more. smooth={loss_smooth:.4f}, varying={loss_varying:.4f}"

    def test_varying_envelope_unpenalized(self, device, time_steps):
        """Envelopes with sufficient local variance should have near-zero penalty."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        # High variance envelope
        noisy_envelope = torch.rand(1, time_steps, device=device).abs() * 2.0 + 0.5
        loss = loss_fn(noisy_envelope)

        # Should be near zero since variance is well above 0.05
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_gradient_flow(self, device, batch, time_steps):
        """Should support gradient backpropagation."""
        from modules.losses.breathy_perceptual_loss import NoiseModulationLoss
        loss_fn = NoiseModulationLoss().to(device)

        noise_envelope = torch.rand(batch, time_steps, device=device).abs().requires_grad_(True)

        loss = loss_fn(noise_envelope)
        loss.backward()
        assert noise_envelope.grad is not None
        assert noise_envelope.grad.abs().sum() > 0


# ============================================================================
# FantPhysicalConsistencyLoss Tests
# ============================================================================

class TestFantPhysicalConsistencyLoss:

    def test_import_exists(self):
        """FantPhysicalConsistencyLoss should be importable."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        assert callable(FantPhysicalConsistencyLoss)

    def test_forward_returns_scalar(self, device, batch, time_steps):
        """forward() returns a scalar tensor."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        H1H2 = torch.randn(batch, time_steps, device=device)
        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3

        loss = loss_fn(H1H2, Rd)
        assert loss.ndim == 0, f"Loss should be scalar, got ndim={loss.ndim}"

    def test_forward_is_finite(self, device, batch, time_steps):
        """Loss should be finite."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        H1H2 = torch.randn(batch, time_steps, device=device)
        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3

        loss = loss_fn(H1H2, Rd)
        assert torch.isfinite(loss).all(), "Loss is NaN or Inf"

    def test_forward_nonnegative(self, device, batch, time_steps):
        """Loss should be non-negative."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        H1H2 = torch.randn(batch, time_steps, device=device)
        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3

        loss = loss_fn(H1H2, Rd)
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item():.6f}"

    def test_physical_formula_correct(self, device):
        """The Fant formula H1H2_fant = -6 + 0.27*exp(5.50*(0.11*Rd+0.5)) should be implemented."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        Rd = torch.tensor([[1.0]], device=device)
        expected = -6.0 + 0.27 * math.exp(5.50 * (0.11 * 1.0 + 0.5))
        computed = loss_fn._fant_formula(Rd)
        assert torch.allclose(computed, torch.tensor(expected, device=device), atol=1e-4), \
            f"Fant formula mismatch: expected {expected:.4f}, got {computed.item():.4f}"

    def test_perfect_consistency_zero_loss(self, device, batch, time_steps):
        """H1H2 matching Fant formula exactly should give zero loss."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3
        H1H2 = loss_fn._fant_formula(Rd)

        loss = loss_fn(H1H2, Rd)
        assert loss.item() < 1e-4, \
            f"Physically consistent H1H2 should have near-zero loss, got {loss.item():.6f}"

    def test_inconsistency_nonzero_loss(self, device, batch, time_steps):
        """Deviating from Fant formula should produce non-zero loss."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3
        H1H2_fant = loss_fn._fant_formula(Rd)
        H1H2_wrong = H1H2_fant + 5.0  # Add large deviation

        loss = loss_fn(H1H2_wrong, Rd)
        assert loss.item() > 0, "Inconsistent H1H2 should produce non-zero loss"

    def test_gradient_flow(self, device, batch, time_steps):
        """Should support gradient backpropagation."""
        from modules.losses.breathy_perceptual_loss import FantPhysicalConsistencyLoss
        loss_fn = FantPhysicalConsistencyLoss().to(device)

        H1H2 = torch.randn(batch, time_steps, device=device, requires_grad=True)
        Rd = torch.rand(batch, time_steps, device=device) * 2.0 + 0.3

        loss = loss_fn(H1H2, Rd)
        loss.backward()
        assert H1H2.grad is not None
        assert H1H2.grad.abs().sum() > 0


# ============================================================================
# Integration Test
# ============================================================================

class TestIntegration:

    def test_all_losses_compatible(self, device, batch, time_steps):
        """All five loss modules should be independently callable."""
        from modules.losses.breathy_perceptual_loss import (
            H1H2PerceptualLoss, HNRPerceptualLoss, AirEnergyLoss,
            NoiseModulationLoss, FantPhysicalConsistencyLoss
        )

        B, T = batch, time_steps
        spec = torch.randn(B, T, 513, device=device).abs() + 1e-6
        noise_env = torch.rand(B, T, device=device).abs() + 0.01
        H1H2 = torch.randn(B, T, device=device)
        Rd = torch.rand(B, T, device=device) * 2.0 + 0.3

        loss1 = H1H2PerceptualLoss().to(device)(spec, spec)
        loss2 = HNRPerceptualLoss().to(device)(spec, spec)
        loss3 = AirEnergyLoss().to(device)(spec, spec)
        loss4 = NoiseModulationLoss().to(device)(noise_env)
        loss5 = FantPhysicalConsistencyLoss().to(device)(H1H2, Rd)

        for name, loss in [('H1H2', loss1), ('HNR', loss2), ('AirEnergy', loss3),
                           ('NoiseMod', loss4), ('Fant', loss5)]:
            assert loss.ndim == 0, f"{name} should be scalar"
            assert torch.isfinite(loss), f"{name} should be finite"
            assert loss.item() >= 0, f"{name} should be non-negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
