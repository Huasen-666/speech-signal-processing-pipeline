from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


DSPIC_SAMPLE_RATE = 8000
DSPIC_FRAME = 128
DSPIC_ORDER = 16
PRE_EMPHASIS_ALPHA = 0.95

PARCOR_FEATURE_NAMES = [f"parcor_{idx + 1:02d}" for idx in range(DSPIC_ORDER)] + [
    "rms",
    "zcr",
]

BP_ONSET_EXTRA_FEATURE_NAMES = [
    "onset_rms_0_30ms",
    "onset_rms_30_80ms",
    "onset_rms_80_160ms",
    "rms_ratio_0_80_to_80_160ms",
    "zcr_0_80ms",
    "zcr_80_160ms",
    "high_band_ratio_0_80ms",
    "high_band_ratio_80_160ms",
    "low_band_ratio_0_80ms",
    "spectral_centroid_0_80ms",
    "spectral_centroid_80_160ms",
    "peak_time_0_160ms",
    "min_10ms_rms_20_100ms",
]

BP_ONSET_FEATURE_NAMES = PARCOR_FEATURE_NAMES + BP_ONSET_EXTRA_FEATURE_NAMES

N_MELS = 20
N_MFCC = 10
LOGMEL_STAT_FEATURE_NAMES = (
    [f"logmel_mean_{idx + 1:02d}" for idx in range(N_MELS)]
    + [f"logmel_std_{idx + 1:02d}" for idx in range(N_MELS)]
    + [f"mfcc_mean_{idx + 1:02d}" for idx in range(N_MFCC)]
    + [f"mfcc_std_{idx + 1:02d}" for idx in range(N_MFCC)]
)

MFCC_LOGMEL_ONSET_FEATURE_NAMES = BP_ONSET_FEATURE_NAMES + LOGMEL_STAT_FEATURE_NAMES

AVAILABLE_FEATURE_SETS = [
    "parcor",
    "bp_onset",
    "mfcc_logmel_onset",
    "onset_only",
    "logmel_mfcc",
    "parcor_logmel_mfcc",
    "onset_logmel_mfcc",
]


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int = DSPIC_SAMPLE_RATE) -> np.ndarray:
    """Resample using linear interpolation to mirror a lightweight embedded path."""
    mono = np.asarray(audio, dtype=np.float64)
    if source_rate == target_rate:
        return mono
    if len(mono) == 0:
        return mono

    output_len = max(1, round(len(mono) * target_rate / source_rate))
    source_x = np.arange(len(mono), dtype=np.float64)
    target_x = np.linspace(0, len(mono) - 1, output_len)
    return np.interp(target_x, source_x, mono)


def preemphasis(signal: np.ndarray, alpha: float = PRE_EMPHASIS_ALPHA) -> np.ndarray:
    """Apply y[n] = x[n] - alpha * x[n - 1]."""
    data = np.asarray(signal, dtype=np.float64)
    if len(data) == 0:
        return data
    out = np.empty_like(data)
    out[0] = data[0]
    out[1:] = data[1:] - alpha * data[:-1]
    return out


def autocorrelation(frame: np.ndarray, order: int = DSPIC_ORDER) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float64)
    values = np.zeros(order + 1, dtype=np.float64)
    for lag in range(order + 1):
        values[lag] = np.dot(data[: len(data) - lag], data[lag:])
    return values


def levinson_durbin(ref: np.ndarray, order: int = DSPIC_ORDER) -> tuple[np.ndarray, float]:
    """Return PARCOR reflection coefficients from autocorrelation values."""
    coeffs = np.zeros(order, dtype=np.float64)
    lpc = np.zeros(order, dtype=np.float64)
    error = float(ref[0])

    if error < 1e-10:
        return coeffs, error

    for step in range(order):
        reflection = -float(ref[step + 1])
        for prev in range(step):
            reflection -= lpc[prev] * ref[step - prev]
        if abs(error) < 1e-10:
            break

        reflection /= error
        reflection = float(np.clip(reflection, -1.0 + 1e-9, 1.0 - 1e-9))
        coeffs[step] = reflection

        updated = np.zeros(order, dtype=np.float64)
        updated[step] = reflection
        for prev in range(step):
            updated[prev] = lpc[prev] + reflection * lpc[step - 1 - prev]
        lpc = updated
        error *= 1.0 - reflection * reflection

    return coeffs, error


def rms_energy(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(frame, dtype=np.float64) ** 2)) / 32768.0)


def zero_crossing_rate(frame: np.ndarray) -> float:
    data = np.asarray(frame, dtype=np.float64)
    if len(data) <= 1:
        return 0.0
    zcr = np.sum(np.diff(np.sign(data)) != 0) / len(data)
    return float(min(zcr * 2.0, 1.0))


def window_ms(signal: np.ndarray, sample_rate: int, start_ms: float, end_ms: float) -> np.ndarray:
    start = max(0, round(sample_rate * start_ms / 1000.0))
    end = max(start + 1, round(sample_rate * end_ms / 1000.0))
    if len(signal) < end:
        signal = np.pad(signal, (0, end - len(signal)))
    return signal[start:end]


def band_energy_ratio(frame: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    data = np.asarray(frame, dtype=np.float64)
    if len(data) == 0:
        return 0.0
    windowed = data * np.hamming(len(data))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    total = float(np.sum(spectrum)) + 1e-12
    band = float(np.sum(spectrum[(freqs >= low_hz) & (freqs <= high_hz)]))
    return band / total


def spectral_centroid(frame: np.ndarray, sample_rate: int) -> float:
    data = np.asarray(frame, dtype=np.float64)
    if len(data) == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(data * np.hamming(len(data))))
    freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
    total = float(np.sum(spectrum)) + 1e-12
    return float(np.sum(freqs * spectrum) / total / (sample_rate / 2.0))


def normalized_rms(frame: np.ndarray) -> float:
    data = np.asarray(frame, dtype=np.float64)
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data**2)) / 32768.0)


def min_frame_rms(signal: np.ndarray, sample_rate: int, start_ms: float, end_ms: float, frame_ms: float = 10.0) -> float:
    region = window_ms(signal, sample_rate, start_ms, end_ms)
    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    if len(region) < frame_len:
        region = np.pad(region, (0, frame_len - len(region)))
    values = [
        normalized_rms(region[start : start + frame_len])
        for start in range(0, len(region) - frame_len + 1, frame_len)
    ]
    return float(min(values)) if values else 0.0


def trim_by_energy(audio: np.ndarray, frame_size: int = DSPIC_FRAME, hop_size: int = DSPIC_FRAME // 2) -> np.ndarray:
    """Trim leading/trailing silence using Dave-compatible relative energy thresholding."""
    data = np.asarray(audio, dtype=np.float64)
    if len(data) < frame_size:
        data = np.pad(data, (0, frame_size - len(data)))

    energies = [
        float(np.mean(data[start : start + frame_size] ** 2))
        for start in range(0, len(data) - frame_size, hop_size)
    ]
    if not energies:
        return data

    threshold = max(energies) * 0.04
    active = [idx for idx, value in enumerate(energies) if value > threshold]
    if len(active) < 2:
        return data

    start = active[0] * hop_size
    end = active[-1] * hop_size + frame_size
    return data[start:end]


def extract_parcor_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract Dave-style 18D features: 16 PARCOR + RMS + ZCR."""
    mono = np.asarray(audio, dtype=np.float64)
    if mono.ndim != 1:
        raise ValueError("extract_parcor_features expects a mono waveform")
    if len(mono) == 0:
        return np.zeros(len(PARCOR_FEATURE_NAMES), dtype=np.float64)

    peak = float(np.max(np.abs(mono)))
    if peak < 1e-12:
        return np.zeros(len(PARCOR_FEATURE_NAMES), dtype=np.float64)

    scaled = mono / peak * 32767.0
    scaled = resample_linear(scaled, sample_rate, DSPIC_SAMPLE_RATE)
    scaled = preemphasis(scaled)
    scaled = trim_by_energy(scaled)

    if len(scaled) < DSPIC_FRAME:
        scaled = np.pad(scaled, (0, DSPIC_FRAME - len(scaled)))

    window = np.hamming(DSPIC_FRAME)
    parcor_sum = np.zeros(DSPIC_ORDER, dtype=np.float64)
    rms_sum = 0.0
    zcr_sum = 0.0
    frame_count = 0

    for start in range(0, len(scaled) - DSPIC_FRAME + 1, DSPIC_FRAME):
        frame = scaled[start : start + DSPIC_FRAME] * window
        ref = autocorrelation(frame, DSPIC_ORDER)
        if ref[0] < 1e-10:
            continue
        parcor, _ = levinson_durbin(ref, DSPIC_ORDER)
        parcor_sum += parcor
        rms_sum += rms_energy(frame)
        zcr_sum += zero_crossing_rate(frame)
        frame_count += 1

    if frame_count == 0:
        return np.zeros(len(PARCOR_FEATURE_NAMES), dtype=np.float64)

    return np.concatenate(
        [
            parcor_sum / frame_count,
            np.array([rms_sum / frame_count, zcr_sum / frame_count], dtype=np.float64),
        ]
    )


def prepare_active_dsp_signal(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float64)
    if len(mono) == 0:
        return mono
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-12:
        return np.zeros(1, dtype=np.float64)
    scaled = mono / peak * 32767.0
    scaled = resample_linear(scaled, sample_rate, DSPIC_SAMPLE_RATE)
    scaled = preemphasis(scaled)
    return trim_by_energy(scaled)


def extract_bp_onset_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Add onset cues for B/P: burst energy, aspiration region, ZCR, and band ratios."""
    parcor = extract_parcor_features(audio, sample_rate)
    active = prepare_active_dsp_signal(audio, sample_rate)
    if len(active) == 0:
        return np.concatenate([parcor, np.zeros(len(BP_ONSET_EXTRA_FEATURE_NAMES), dtype=np.float64)])

    w_0_30 = window_ms(active, DSPIC_SAMPLE_RATE, 0.0, 30.0)
    w_30_80 = window_ms(active, DSPIC_SAMPLE_RATE, 30.0, 80.0)
    w_0_80 = window_ms(active, DSPIC_SAMPLE_RATE, 0.0, 80.0)
    w_80_160 = window_ms(active, DSPIC_SAMPLE_RATE, 80.0, 160.0)
    w_0_160 = window_ms(active, DSPIC_SAMPLE_RATE, 0.0, 160.0)

    rms_0_30 = normalized_rms(w_0_30)
    rms_30_80 = normalized_rms(w_30_80)
    rms_80_160 = normalized_rms(w_80_160)
    rms_0_80 = normalized_rms(w_0_80)
    ratio_0_80_to_80_160 = rms_0_80 / (rms_80_160 + 1e-8)
    peak_time = float(np.argmax(np.abs(w_0_160)) / max(1, len(w_0_160) - 1))

    extras = np.array(
        [
            rms_0_30,
            rms_30_80,
            rms_80_160,
            ratio_0_80_to_80_160,
            zero_crossing_rate(w_0_80),
            zero_crossing_rate(w_80_160),
            band_energy_ratio(w_0_80, DSPIC_SAMPLE_RATE, 1500.0, 3900.0),
            band_energy_ratio(w_80_160, DSPIC_SAMPLE_RATE, 1500.0, 3900.0),
            band_energy_ratio(w_0_80, DSPIC_SAMPLE_RATE, 80.0, 500.0),
            spectral_centroid(w_0_80, DSPIC_SAMPLE_RATE),
            spectral_centroid(w_80_160, DSPIC_SAMPLE_RATE),
            peak_time,
            min_frame_rms(active, DSPIC_SAMPLE_RATE, 20.0, 100.0),
        ],
        dtype=np.float64,
    )
    return np.concatenate([parcor, extras])


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, fft_size: int, n_mels: int = N_MELS) -> np.ndarray:
    min_mel = float(hz_to_mel(80.0))
    max_mel = float(hz_to_mel(sample_rate / 2.0))
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)

    filters = np.zeros((n_mels, fft_size // 2 + 1), dtype=np.float64)
    for mel_idx in range(1, n_mels + 1):
        left = bins[mel_idx - 1]
        center = bins[mel_idx]
        right = bins[mel_idx + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for bin_idx in range(left, min(center, filters.shape[1])):
            filters[mel_idx - 1, bin_idx] = (bin_idx - left) / max(center - left, 1)
        for bin_idx in range(center, min(right, filters.shape[1])):
            filters[mel_idx - 1, bin_idx] = (right - bin_idx) / max(right - center, 1)
    return filters


def frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    data = np.asarray(signal, dtype=np.float64)
    if len(data) < frame_len:
        data = np.pad(data, (0, frame_len - len(data)))
    starts = np.arange(0, len(data) - frame_len + 1, hop_len)
    return np.stack([data[start : start + frame_len] for start in starts])


def logmel_mfcc_stats(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = round(sample_rate * 0.025)
    hop_len = round(sample_rate * 0.010)
    fft_size = 256
    frames = frame_signal(signal, frame_len, hop_len)
    windowed = frames * np.hamming(frame_len)[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_size, axis=1)) ** 2
    filters = mel_filterbank(sample_rate, fft_size, N_MELS)
    mel_energy = spectrum @ filters.T
    logmel = np.log(np.maximum(mel_energy, 1e-10))
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

    return np.concatenate(
        [
            logmel.mean(axis=0),
            logmel.std(axis=0),
            mfcc.mean(axis=0),
            mfcc.std(axis=0),
        ]
    ).astype(np.float64)


def extract_mfcc_logmel_onset_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Codex v2 features: Dave-compatible PARCOR plus B/P onset and log-mel/MFCC statistics."""
    onset = extract_bp_onset_features(audio, sample_rate)
    active = prepare_active_dsp_signal(audio, sample_rate)
    if len(active) == 0:
        stats = np.zeros(len(LOGMEL_STAT_FEATURE_NAMES), dtype=np.float64)
    else:
        stats = logmel_mfcc_stats(active, DSPIC_SAMPLE_RATE)
    return np.concatenate([onset, stats])


def feature_names_for_set(feature_set: str) -> list[str]:
    if feature_set == "parcor":
        return PARCOR_FEATURE_NAMES
    if feature_set == "bp_onset":
        return BP_ONSET_FEATURE_NAMES
    if feature_set == "mfcc_logmel_onset":
        return MFCC_LOGMEL_ONSET_FEATURE_NAMES
    if feature_set == "onset_only":
        return BP_ONSET_EXTRA_FEATURE_NAMES
    if feature_set == "logmel_mfcc":
        return LOGMEL_STAT_FEATURE_NAMES
    if feature_set == "parcor_logmel_mfcc":
        return PARCOR_FEATURE_NAMES + LOGMEL_STAT_FEATURE_NAMES
    if feature_set == "onset_logmel_mfcc":
        return BP_ONSET_EXTRA_FEATURE_NAMES + LOGMEL_STAT_FEATURE_NAMES
    raise ValueError(f"Unknown feature set: {feature_set}")


def extract_feature_set(audio: np.ndarray, sample_rate: int, feature_set: str) -> np.ndarray:
    if feature_set == "parcor":
        return extract_parcor_features(audio, sample_rate)
    if feature_set == "bp_onset":
        return extract_bp_onset_features(audio, sample_rate)

    full = extract_mfcc_logmel_onset_features(audio, sample_rate)
    parcor_end = len(PARCOR_FEATURE_NAMES)
    onset_end = parcor_end + len(BP_ONSET_EXTRA_FEATURE_NAMES)

    if feature_set == "mfcc_logmel_onset":
        return full
    if feature_set == "onset_only":
        return full[parcor_end:onset_end]
    if feature_set == "logmel_mfcc":
        return full[onset_end:]
    if feature_set == "parcor_logmel_mfcc":
        return np.concatenate([full[:parcor_end], full[onset_end:]])
    if feature_set == "onset_logmel_mfcc":
        return full[parcor_end:]
    raise ValueError(f"Unknown feature set: {feature_set}")
