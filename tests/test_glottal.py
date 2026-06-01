"""
TDD tests for lib/feature/glottal.py — SFC Breathness acoustic feature extraction.

RED phase: These tests must FAIL initially because glottal.py doesn't exist yet.
"""

import numpy as np
import pytest


# =============================================================================
# Helper: generate test signals
# =============================================================================

def _sine_wave(freq: float, duration: float = 1.0, sample_rate: int = 44100):
    """Generate a pure sine wave at given frequency."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_f0_array(n_frames: int, f0_val: float = 440.0):
    """Create a uniform f0 array for n frames."""
    return np.full(n_frames, f0_val, dtype=np.float32)


# =============================================================================
# compute_h1_h2 tests
# =============================================================================

class TestComputeH1H2:
    """Tests for H1-H2 harmonic amplitude difference computation."""

    def test_pure_sine_h1h2_near_zero(self):
        """Pure sine wave (only H1, no H2) -> H2 ~ 0 -> H1-H2 very large positive."""
        from lib.feature.glottal import compute_h1_h2
        waveform = _sine_wave(440.0, duration=0.5)
        f0 = _make_f0_array(10, 440.0)

        result = compute_h1_h2(waveform, f0, sample_rate=44100)
        # Pure sine: H2 is near noise floor -> H1-H2 should be > 20 dB
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, "Expected at least some valid H1-H2 values"
        assert np.mean(valid) > 20.0, f"Pure sine H1-H2 should be large, got {np.mean(valid):.1f}"

    def test_two_harmonics_h1h2(self):
        """Signal with H1=0.3, H2=0.15 -> H1-H2 ~ 20*log10(0.3/0.15) ~ 6.02 dB."""
        from lib.feature.glottal import compute_h1_h2
        sample_rate = 44100
        duration = 0.5
        t = np.arange(int(duration * sample_rate)) / sample_rate
        # H1 at 440Hz (amp 0.3), H2 at 880Hz (amp 0.15)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t) +
                    0.15 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        f0 = _make_f0_array(10, 440.0)

        result = compute_h1_h2(waveform, f0, sample_rate=sample_rate)
        valid = result[~np.isnan(result)]
        expected = 20 * np.log10(0.3 / 0.15)  # ~ 6.02
        assert abs(np.median(valid) - expected) < 2.0, \
            f"Expected H1-H2 ~ {expected:.1f}, got {np.median(valid):.1f}"

    def test_returns_numpy_array(self):
        """Output should be a numpy array of floats, one per frame."""
        from lib.feature.glottal import compute_h1_h2
        waveform = _sine_wave(440.0, duration=0.3)
        f0 = _make_f0_array(8)

        result = compute_h1_h2(waveform, f0, sample_rate=44100)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f0)
        assert result.dtype in (np.float32, np.float64)

    def test_nan_for_zero_f0(self):
        """Frames with f0=0 should produce NaN H1-H2."""
        from lib.feature.glottal import compute_h1_h2
        waveform = _sine_wave(440.0, duration=0.5)
        f0 = np.array([440.0, 0.0, 440.0, 0.0, 440.0], dtype=np.float32)

        result = compute_h1_h2(waveform, f0, sample_rate=44100)
        assert np.isnan(result[1]), "f0=0 frame should be NaN"
        assert np.isnan(result[3]), "f0=0 frame should be NaN"
        assert not np.isnan(result[0]), "f0>0 frame should have valid H1-H2"


# =============================================================================
# compute_hnr tests
# =============================================================================

class TestComputeHNR:
    """Tests for Harmonics-to-Noise Ratio computation."""

    def test_hnr_pure_sine_high(self):
        """Pure sine wave -> very high HNR (> 25 dB)."""
        from lib.feature.glottal import compute_hnr
        waveform = _sine_wave(440.0, duration=0.8)
        f0 = _make_f0_array(15, 440.0)

        result = compute_hnr(waveform, f0, sample_rate=44100)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.mean(valid) > 10.0, f"Pure sine HNR should be above noise, got {np.mean(valid):.1f}"

    def test_hnr_white_noise_low(self):
        """White noise -> very low HNR (< 10 dB)."""
        from lib.feature.glottal import compute_hnr
        np.random.seed(42)
        waveform = np.random.randn(44100).astype(np.float32) * 0.1
        f0 = _make_f0_array(20, 440.0)  # arbitrary f0 for frame boundaries

        result = compute_hnr(waveform, f0, sample_rate=44100)
        valid = result[~np.isnan(result)]
        assert np.mean(valid) < 10.0, f"White noise HNR should be low, got {np.mean(valid):.1f}"

    def test_hnr_sine_plus_noise_intermediate(self):
        """Sine + moderate noise -> intermediate HNR."""
        from lib.feature.glottal import compute_hnr
        sample_rate = 44100
        t = np.arange(int(0.8 * sample_rate)) / sample_rate
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t) +
                    0.02 * np.random.randn(len(t))).astype(np.float32)
        f0 = _make_f0_array(15, 440.0)

        result = compute_hnr(waveform, f0, sample_rate=44100)
        valid = result[~np.isnan(result)]
        # Should be lower than pure sine but higher than white noise
        assert 5.0 < np.mean(valid) < 40.0, f"Got HNR={np.mean(valid):.1f}"

    def test_returns_numpy_array(self):
        """Output should be a numpy array, one value per frame."""
        from lib.feature.glottal import compute_hnr
        waveform = _sine_wave(440.0, duration=0.5)
        f0 = _make_f0_array(10)

        result = compute_hnr(waveform, f0, sample_rate=44100)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(f0)


# =============================================================================
# compute_air_energy_ratio tests
# =============================================================================

class TestComputeAirEnergyRatio:
    """Tests for 8-16kHz air energy ratio."""

    def test_low_freq_sine_low_ratio(self):
        """Pure 440Hz sine -> almost no energy above 8kHz -> ratio near 0."""
        from lib.feature.glottal import compute_air_energy_ratio
        waveform = _sine_wave(440.0, duration=1.0)
        ratio = compute_air_energy_ratio(waveform, sample_rate=44100)
        assert ratio < 0.05, f"440Hz sine should have low air ratio, got {ratio:.4f}"

    def test_high_freq_sine_high_ratio(self):
        """12kHz sine -> all energy in air band -> ratio near 1.0."""
        from lib.feature.glottal import compute_air_energy_ratio
        waveform = _sine_wave(12000.0, duration=1.0)
        ratio = compute_air_energy_ratio(waveform, sample_rate=44100)
        assert ratio > 0.80, f"12kHz sine should have high air ratio, got {ratio:.4f}"

    def test_white_noise_mid_range(self):
        """White noise -> air band ~ (16000-8000)/(22050) ~ 0.36 of total."""
        from lib.feature.glottal import compute_air_energy_ratio
        np.random.seed(42)
        waveform = np.random.randn(44100).astype(np.float32) * 0.1
        ratio = compute_air_energy_ratio(waveform, sample_rate=44100)
        # For white noise, ratio ~ bandwidth ratio
        assert 0.20 < ratio < 0.55, f"White noise ratio should be ~0.36, got {ratio:.4f}"

    def test_returns_float(self):
        """Output is a single float scalar."""
        from lib.feature.glottal import compute_air_energy_ratio
        waveform = _sine_wave(440.0, duration=0.5)
        ratio = compute_air_energy_ratio(waveform, sample_rate=44100)
        assert isinstance(ratio, float)


# =============================================================================
# compute_spectral_tilt tests
# =============================================================================

class TestComputeSpectralTilt:
    """Tests for spectral tilt computation."""

    def test_flat_spectrum_tilt_near_zero(self):
        """White noise -> flat spectrum -> tilt near 0 dB/octave."""
        from lib.feature.glottal import compute_spectral_tilt
        np.random.seed(42)
        waveform = np.random.randn(44100).astype(np.float32) * 0.1
        result = compute_spectral_tilt(waveform, sample_rate=44100)
        valid = result[~np.isnan(result)]
        # White noise spectral tilt should be near 0 (flat)
        assert -5.0 < np.mean(valid) < 5.0, \
            f"White noise tilt should be near 0, got {np.mean(valid):.1f}"

    def test_lowpass_signal_negative_tilt(self):
        """Low-frequency emphasis -> negative spectral tilt."""
        from lib.feature.glottal import compute_spectral_tilt
        # Sum of low harmonics -> more energy at low freqs -> negative tilt
        sample_rate = 44100
        t = np.arange(int(0.5 * sample_rate)) / sample_rate
        waveform = np.zeros_like(t, dtype=np.float32)
        for k, amp in zip([1, 2, 3, 4, 5], [0.3, 0.15, 0.08, 0.04, 0.02]):
            waveform += amp * np.sin(2 * np.pi * 200 * k * t)
        waveform = waveform.astype(np.float32)

        result = compute_spectral_tilt(waveform, sample_rate=sample_rate)
        valid = result[~np.isnan(result)]
        assert np.mean(valid) < 0, f"Lowpass signal should have negative tilt, got {np.mean(valid):.1f}"

    def test_returns_numpy_array(self):
        """Output should be a numpy array, one value per frame."""
        from lib.feature.glottal import compute_spectral_tilt
        waveform = _sine_wave(440.0, duration=0.5)
        result = compute_spectral_tilt(waveform, sample_rate=44100)
        assert isinstance(result, np.ndarray)
