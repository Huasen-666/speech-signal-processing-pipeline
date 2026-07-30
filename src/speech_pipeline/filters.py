'''
use to put DSP filter function responding the part of “the conponent to change signal”
such as low frequency shaking, remove high frequence noise and fix frequency
'''
import numpy as np
from scipy import signal
'''
A normal audio waveform should oscillate above and below 0.

If the entire waveform is shifted upward or downward, it is called a DC offset.
'''
def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Remove constant waveform bias."""
    mono = np.asarray(audio, dtype=np.float32)
    return (mono - float(np.mean(mono))).astype(np.float32)


def _safe_sosfiltfilt(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if len(audio) < 128:
        return signal.sosfilt(sos, audio).astype(np.float32)
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter."""
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return _safe_sosfiltfilt(sos, np.asarray(audio, dtype=np.float32))


def lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 7600.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter."""
    nyquist = sample_rate / 2.0
    cutoff_hz = min(cutoff_hz, nyquist * 0.95)
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return _safe_sosfiltfilt(sos, np.asarray(audio, dtype=np.float32))

'''
conbine high pass and low pass
'''
def bandpass(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 80.0,
    high_hz: float = 7600.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a conservative speech band-pass filter."""
    filtered = highpass(audio, sample_rate, cutoff_hz=low_hz, order=order)
    return lowpass(filtered, sample_rate, cutoff_hz=high_hz, order=order)


def notch(audio: np.ndarray, sample_rate: int, frequency_hz: float, q: float = 30.0) -> np.ndarray:
    """Apply a narrow notch filter for stable tonal interference."""
    if frequency_hz <= 0:
        return np.asarray(audio, dtype=np.float32)
    if frequency_hz >= sample_rate / 2.0:
        raise ValueError("notch frequency must be below Nyquist")

    b, a = signal.iirnotch(w0=frequency_hz, Q=q, fs=sample_rate)
    if len(audio) < 128:
        return signal.lfilter(b, a, audio).astype(np.float32)
    return signal.filtfilt(b, a, audio).astype(np.float32)
