"""
SFC Breathness acoustic feature extraction utilities.

Computes H1-H2 harmonic difference, Harmonics-to-Noise Ratio (HNR),
high-frequency air energy ratio, and spectral tilt from raw waveform.
"""

import numpy as np
from scipy.signal import correlate


def compute_h1_h2(
    waveform: np.ndarray,
    f0: np.ndarray,
    sample_rate: int = 44100,
    frame_length: float = 0.025,
    hop_length: float = 0.010,
) -> np.ndarray:
    """
    Compute per-frame H1-H2 amplitude difference in dB.

    For each frame, extracts the amplitude at f0 (H1) and 2*f0 (H2)
    via FFT peak picking. Returns NaN for frames where f0 == 0 or
    harmonics cannot be reliably detected.

    Sweet Breathy target: H1-H2 ∈ [3, 10] dB.

    :param waveform: Raw 1D audio signal, shape [T].
    :param f0: Per-frame fundamental frequency in Hz, shape [N].
    :param sample_rate: Audio sample rate.
    :param frame_length: Analysis window length in seconds.
    :param hop_length: Hop between frames in seconds.
    :return: H1-H2 values in dB, shape [N].
    """
    n_frames = len(f0)
    result = np.full(n_frames, np.nan, dtype=np.float64)

    win_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    fft_size = 1
    while fft_size < win_samples:
        fft_size *= 2

    noise_floor = 1e-8  # Amplitude floor for reliable peak detection

    for i in range(n_frames):
        f_val = f0[i]
        if f_val <= 0:
            continue

        start = i * hop_samples
        end = min(start + win_samples, len(waveform))
        if end - start < win_samples // 2:
            continue

        frame = waveform[start:end].astype(np.float64)
        if len(frame) < fft_size:
            frame = np.pad(frame, (0, fft_size - len(frame)))

        # Apply Hann window
        window = np.hanning(len(frame))
        frame = frame * window

        spec = np.fft.rfft(frame, n=fft_size)
        mag = np.abs(spec)
        freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        def _peak_amp(target_freq: float) -> float:
            bin_idx = int(target_freq * fft_size / sample_rate)
            if 1 <= bin_idx < len(mag) - 1:
                return float(mag[bin_idx])
            # Search within ±5% range
            lo = max(1, int(target_freq * 0.95 * fft_size / sample_rate))
            hi = min(len(mag) - 1, int(target_freq * 1.05 * fft_size / sample_rate))
            if lo >= hi:
                return 0.0
            local = mag[lo:hi + 1]
            return float(local.max()) if len(local) > 0 else 0.0

        h1_amp = _peak_amp(f_val)
        h2_amp = _peak_amp(2.0 * f_val)

        if h1_amp > noise_floor and h2_amp > noise_floor:
            result[i] = 20.0 * np.log10(h1_amp / h2_amp)
        elif h1_amp > noise_floor:
            result[i] = np.inf  # H1 present, H2 absent → very breathy

    return result


def compute_hnr(
    waveform: np.ndarray,
    f0: np.ndarray,
    sample_rate: int = 44100,
    frame_length: float = 0.025,
    hop_length: float = 0.010,
) -> np.ndarray:
    """
    Compute per-frame Harmonics-to-Noise Ratio in dB using autocorrelation.

    Sweet Breathy target: HNR ∈ [12, 20] dB.

    :param waveform: Raw 1D audio signal, shape [T].
    :param f0: Per-frame fundamental frequency in Hz, shape [N].
    :param sample_rate: Audio sample rate.
    :param frame_length: Analysis window length in seconds.
    :param hop_length: Hop between frames in seconds.
    :return: HNR values in dB, shape [N].
    """
    n_frames = len(f0)
    result = np.full(n_frames, np.nan, dtype=np.float64)

    win_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)

    for i in range(n_frames):
        f_val = f0[i]
        if f_val <= 0:
            continue

        start = i * hop_samples
        end = min(start + win_samples, len(waveform))
        if end - start < win_samples // 2:
            continue

        frame = waveform[start:end].astype(np.float64)
        window = np.hanning(len(frame))
        frame = frame * window

        # Autocorrelation
        acf = correlate(frame, frame, mode='full')
        acf = acf[len(acf) // 2:]  # Keep positive lags
        acf = acf / (acf[0] + 1e-10)  # Normalize

        # Find peak near pitch period
        pitch_period = int(sample_rate / f_val)
        if pitch_period <= 0 or pitch_period >= len(acf):
            continue

        # Harmonic energy: autocorrelation peak at pitch period
        search_lo = max(1, pitch_period // 2)
        search_hi = min(len(acf) - 1, pitch_period * 2)
        if search_hi <= search_lo:
            continue
        peak_idx = search_lo + np.argmax(acf[search_lo:search_hi + 1])
        harmonic_energy = acf[peak_idx]

        # Noise energy: 1 - harmonic correlation
        noise_energy = max(1e-10, 1.0 - harmonic_energy)

        hnr_linear = harmonic_energy / noise_energy
        result[i] = 10.0 * np.log10(hnr_linear + 1e-10)

    return result


def compute_air_energy_ratio(
    waveform: np.ndarray,
    sample_rate: int = 44100,
    low_freq: float = 8000.0,
    high_freq: float = 16000.0,
) -> float:
    """
    Compute the ratio of energy in the [low_freq, high_freq] band
    (the "air" or "breathiness" band) to total signal energy.

    Sweet Breathy target: ratio ∈ [0.08, 0.20].

    :param waveform: Raw 1D audio signal, shape [T].
    :param sample_rate: Audio sample rate.
    :param low_freq: Lower bound of air band in Hz.
    :param high_freq: Upper bound of air band in Hz.
    :return: Energy ratio as a float.
    """
    if len(waveform) == 0:
        return 0.0

    fft_size = len(waveform)
    spec = np.abs(np.fft.rfft(waveform, n=fft_size))
    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    # Air band mask
    air_mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)

    air_energy = np.sum(spec[air_mask] ** 2)
    total_energy = np.sum(spec ** 2)

    if total_energy < 1e-12:
        return 0.0

    return float(air_energy / total_energy)


def compute_spectral_tilt(
    waveform: np.ndarray,
    sample_rate: int = 44100,
    frame_length: float = 0.025,
    hop_length: float = 0.010,
) -> np.ndarray:
    """
    Compute per-frame spectral tilt in dB/octave via linear regression.

    Positive tilt = high frequencies emphasized.
    Negative tilt = low frequencies emphasized (typical for voice).
    Sweet Breathy target: tilt ∈ [-15, -5] dB/octave.

    :param waveform: Raw 1D audio signal, shape [T].
    :param sample_rate: Audio sample rate.
    :param frame_length: Analysis window length in seconds.
    :param hop_length: Hop between frames in seconds.
    :return: Spectral tilt values in dB/octave, shape [N].
    """
    win_samples = int(frame_length * sample_rate)
    hop_samples = int(hop_length * sample_rate)
    fft_size = 1
    while fft_size < win_samples:
        fft_size *= 2

    n_frames = 1 + max(0, (len(waveform) - win_samples)) // hop_samples
    result = np.full(n_frames, np.nan, dtype=np.float64)

    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    # Only use bins with positive frequency for regression
    valid_mask = freq_bins > 0
    log_freq = np.log2(freq_bins[valid_mask])

    if len(log_freq) < 2:
        return result

    for i in range(n_frames):
        start = i * hop_samples
        end = min(start + win_samples, len(waveform))
        frame = waveform[start:end].astype(np.float64)
        if len(frame) < win_samples // 4:
            continue

        frame = np.pad(frame, (0, fft_size - len(frame))) if len(frame) < fft_size else frame
        window = np.hanning(len(frame))
        frame = frame * window

        spec = np.abs(np.fft.rfft(frame, n=fft_size))
        log_mag = 20.0 * np.log10(spec[valid_mask] + 1e-10)

        # Linear regression: log_mag = slope * log_freq + intercept
        A = np.vstack([log_freq, np.ones_like(log_freq)]).T
        try:
            slope, _ = np.linalg.lstsq(A, log_mag, rcond=None)[0]
            result[i] = float(slope * np.log2(2.0))  # Convert slope to dB/octave
        except np.linalg.LinAlgError:
            continue

    return result
