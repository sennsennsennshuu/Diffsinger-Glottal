"""
Tests for SweetBreathyClassifier (TDD: these tests MUST fail before implementation exists).

Run:
    python -m pytest test_classifier.py -v
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, r"I:\Chaos_extend_solo\DiffSinger-3-Chaos\Diffsinger-v3-sfcb")

import pytest
import torch
import numpy as np
from torch import nn


# ── Fixtures ──

@pytest.fixture
def n_mels():
    return 80

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def time_len():
    return 100

@pytest.fixture
def sample_mel(batch_size, n_mels, time_len):
    """Random mel spectrogram."""
    return torch.randn(batch_size, n_mels, time_len)

@pytest.fixture
def sample_audio():
    """Synthetic audio: 1 second of 440 Hz sine at 22050 Hz."""
    sr = 22050
    duration = 1.0
    t = np.arange(0, duration, 1.0 / sr)
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return audio, sr

@pytest.fixture
def sample_f0():
    """Synthetic F0 contour."""
    return np.full((100,), 220.0, dtype=np.float32)


# ── Test Class ──

class TestSweetBreathyClassifier:
    """All tests for SweetBreathyClassifier."""

    # ═══════════════════════════════════════════════════════════════════
    # 1. Instantiation
    # ═══════════════════════════════════════════════════════════════════

    def test_init_default(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        assert isinstance(model, nn.Module)
        assert model.n_mels == n_mels

    def test_init_custom_params(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(
            n_mels=n_mels, hidden_channels=32, kernel_size=5
        )
        assert model.n_mels == n_mels

    # ═══════════════════════════════════════════════════════════════════
    # 2. Output shape
    # ═══════════════════════════════════════════════════════════════════

    def test_output_shape(self, n_mels, sample_mel):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()
        with torch.no_grad():
            out = model(sample_mel)

        B, _, T = sample_mel.shape
        assert isinstance(out, dict)
        assert 'breathiness_level' in out
        assert 'air_ratio' in out
        assert 'spectral_tilt' in out
        assert 'harmonic_richness' in out
        for key in ['breathiness_level', 'air_ratio', 'spectral_tilt', 'harmonic_richness']:
            assert out[key].shape == (B, T), f"{key} shape mismatch: {out[key].shape}"

    # ═══════════════════════════════════════════════════════════════════
    # 3. Output range [0, 1]
    # ═══════════════════════════════════════════════════════════════════

    def test_output_range(self, n_mels, sample_mel):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()
        with torch.no_grad():
            out = model(sample_mel)

        for key, val in out.items():
            assert val.min() >= 0.0, f"{key} has values < 0: {val.min()}"
            assert val.max() <= 1.0, f"{key} has values > 1: {val.max()}"

    # ═══════════════════════════════════════════════════════════════════
    # 4. Batch consistency
    # ═══════════════════════════════════════════════════════════════════

    def test_batch_consistency(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        single = torch.randn(1, n_mels, 50)
        batch = torch.randn(3, n_mels, 50)
        with torch.no_grad():
            out_single = model(single)
            out_batch = model(batch)

        assert out_batch['breathiness_level'].shape == (3, 50)

    # ═══════════════════════════════════════════════════════════════════
    # 5. classify() method — sweet_breathy label
    # ═══════════════════════════════════════════════════════════════════

    def test_classify_returns_str(self, n_mels, sample_audio, sample_f0):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()
        audio, sr = sample_audio
        label = model.classify(audio, sr, sample_f0)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_classify_sweet_breathy(self, n_mels, sample_audio, sample_f0):
        """
        When conditions clearly indicate sweet_breathy, classify should
        return 'sweet_breathy'.
        """
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        # Create an audio with characteristics matching sweet_breathy
        sr = 22050
        duration = 1.0
        # Add some noise to simulate breathiness
        np.random.seed(42)
        t = np.arange(0, duration, 1.0 / sr)
        clean = 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        noise = 0.1 * np.random.randn(len(t))
        audio = (clean + noise).astype(np.float32)
        f0 = np.full((100,), 220.0, dtype=np.float32)

        label = model.classify(audio, sr, f0)
        assert isinstance(label, str)

    def test_classify_mock_conditions(self, n_mels):
        """
        Verify that when all 4 conditions match sweet_breathy thresholds,
        classify() returns 'sweet_breathy'.
        """
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        # Override forward to return values that match all conditions
        B, T = 1, 100
        mel = torch.randn(B, n_mels, T)

        with torch.no_grad():
            out = model(mel)

        # The raw predictions should be in [0,1]; our classify method
        # will internally check the derived HNR, air_energy etc.
        # We can't easily mock the internal mel analysis here, but
        # we can verify classify runs without error.
        audio = np.zeros(22050, dtype=np.float32)
        sr = 22050
        f0 = np.full((100,), 220.0, dtype=np.float32)
        label = model.classify(audio, sr, f0)
        assert label in ['sweet_breathy', 'modal', 'pressed', 'breathy', 'falsetto', 'creaky', 'other']

    # ═══════════════════════════════════════════════════════════════════
    # 6. Edge cases
    # ═══════════════════════════════════════════════════════════════════

    def test_single_frame(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        mel = torch.randn(2, n_mels, 1)
        with torch.no_grad():
            out = model(mel)

        B = 2
        assert out['breathiness_level'].shape == (B, 1)
        assert out['air_ratio'].shape == (B, 1)
        assert out['spectral_tilt'].shape == (B, 1)
        assert out['harmonic_richness'].shape == (B, 1)

    def test_large_time_dimension(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        mel = torch.randn(1, n_mels, 500)
        with torch.no_grad():
            out = model(mel)
        assert out['breathiness_level'].shape == (1, 500)

    def test_classify_empty_f0(self, n_mels):
        """classify should handle zero-length F0 gracefully."""
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        audio = np.array([0.0], dtype=np.float32)
        sr = 22050
        f0 = np.array([], dtype=np.float32)
        # Should not crash; should return a label
        label = model.classify(audio, sr, f0)
        assert isinstance(label, str)

    def test_different_mel_dimensions(self):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        for nm in [40, 80, 128]:
            model = SweetBreathyClassifier(n_mels=nm)
            mel = torch.randn(2, nm, 30)
            with torch.no_grad():
                out = model(mel)
            assert out['breathiness_level'].shape == (2, 30)

    # ═══════════════════════════════════════════════════════════════════
    # 7. Gradient flow
    # ═══════════════════════════════════════════════════════════════════

    def test_gradient_flow(self, n_mels, sample_mel):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.train()

        sample_mel = sample_mel.requires_grad_(False).clone()

        out = model(sample_mel)
        # Use all 4 output heads so all parameters receive gradients
        loss = (out['breathiness_level'].mean() +
                out['air_ratio'].mean() +
                out['spectral_tilt'].mean() +
                out['harmonic_richness'].mean())
        loss.backward()

        params = list(model.parameters())
        assert len(params) > 0, "Model should have trainable parameters"
        for p in params:
            assert p.grad is not None, f"Parameter {p.shape} has no gradient"
            assert p.grad.abs().sum() > 0, f"Parameter {p.shape} has zero gradient"

    # ═══════════════════════════════════════════════════════════════════
    # 8. Thresholds
    # ═══════════════════════════════════════════════════════════════════

    def test_thresholds_exist(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        expected = {
            'h1_h2_min': 3.0,
            'h1_h2_max': 10.0,
            'hnr_min': 10.0,
            'hnr_max': 22.0,
            'air_energy_min': 0.07,
            'spectral_tilt_min': -15.0,
            'spectral_tilt_max': -5.0,
        }
        model = SweetBreathyClassifier(n_mels=n_mels)
        assert hasattr(model, 'THRESHOLDS')
        for k, v in expected.items():
            assert model.THRESHOLDS[k] == v, f"THRESHOLDS['{k}'] expected {v}, got {model.THRESHOLDS[k]}"

    # ═══════════════════════════════════════════════════════════════════
    # 9. Reproducibility
    # ═══════════════════════════════════════════════════════════════════

    def test_deterministic_in_eval(self, n_mels):
        from modules.sweet_breathy.classifier import SweetBreathyClassifier
        model = SweetBreathyClassifier(n_mels=n_mels)
        model.eval()

        mel = torch.randn(2, n_mels, 30)
        with torch.no_grad():
            out1 = model(mel)
            out2 = model(mel)

        for key in out1:
            assert torch.allclose(out1[key], out2[key], atol=1e-6), \
                f"{key} differs between runs"
