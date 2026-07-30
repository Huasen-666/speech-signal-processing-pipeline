import math

import numpy as np


def amplitude_to_dbfs(value: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(float(value), eps))


def dbfs_to_amplitude(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def rms(audio: np.ndarray) -> float:
    mono = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(mono**2)))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio)))


def rms_normalize(audio: np.ndarray, target_dbfs: float = -24.0) -> tuple[np.ndarray, dict[str, float]]:
    """Scale waveform so its RMS approaches target_dbfs."""
    mono = np.asarray(audio, dtype=np.float32)
    current_rms = rms(mono)
    target_rms = dbfs_to_amplitude(target_dbfs)
    gain = target_rms / max(current_rms, 1e-12)
    normalized = mono * gain

    return normalized.astype(np.float32), {
        "target_rms_dbfs": float(target_dbfs),
        "input_rms_dbfs": round(amplitude_to_dbfs(current_rms), 3),
        "gain": round(float(gain), 6),
        "gain_db": round(amplitude_to_dbfs(gain), 3),
    }


def rms_normalize_with_mask(
    audio: np.ndarray,
    mask: np.ndarray,
    target_dbfs: float = -24.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale waveform using RMS measured only on selected samples."""
    mono = np.asarray(audio, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != mono.shape:
        raise ValueError("mask must have the same shape as audio")

    if np.count_nonzero(mask) == 0:
        normalized, info = rms_normalize(mono, target_dbfs=target_dbfs)
        info["mode"] = "full_audio_fallback"
        info["selected_sample_count"] = 0
        return normalized, info

    selected_rms = rms(mono[mask])
    target_rms = dbfs_to_amplitude(target_dbfs)
    gain = target_rms / max(selected_rms, 1e-12)
    normalized = mono * gain

    return normalized.astype(np.float32), {
        "mode": "speech_only_mask",
        "target_rms_dbfs": float(target_dbfs),
        "selected_rms_dbfs": round(amplitude_to_dbfs(selected_rms), 3),
        "full_input_rms_dbfs": round(amplitude_to_dbfs(rms(mono)), 3),
        "selected_sample_count": int(np.count_nonzero(mask)),
        "selected_sample_ratio": round(float(np.count_nonzero(mask) / max(1, len(mask))), 6),
        "gain": round(float(gain), 6),
        "gain_db": round(amplitude_to_dbfs(gain), 3),
    }

'''
因为 RMS normalization 可能会放大音频。放大后如果某些瞬间超过数字音频最大值，就会 clipping
'''
def peak_protect(audio: np.ndarray, max_peak_dbfs: float = -1.0) -> tuple[np.ndarray, dict[str, float]]:
    """Scale down only when peak exceeds the requested headroom."""
    mono = np.asarray(audio, dtype=np.float32)
    max_peak = dbfs_to_amplitude(max_peak_dbfs)
    current_peak = peak(mono)

    if current_peak <= max_peak:
        scale = 1.0
    else:
        scale = max_peak / max(current_peak, 1e-12)

    protected = mono * scale
    return protected.astype(np.float32), {
        "max_peak_dbfs": float(max_peak_dbfs),
        "input_peak_dbfs": round(amplitude_to_dbfs(current_peak), 3),
        "scale": round(float(scale), 6),
        "scale_db": round(amplitude_to_dbfs(scale), 3),
    }
