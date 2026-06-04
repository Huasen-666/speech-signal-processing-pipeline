import numpy as np


def frame_signal(signal: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a waveform into overlapping frames."""
    frame_len = round(sample_rate * frame_ms / 1000.0)
    hop_len = round(sample_rate * hop_ms / 1000.0)

    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    starts = np.arange(0, len(signal) - frame_len + 1, hop_len)
    frames = np.stack([signal[start : start + frame_len] for start in starts])
    frame_times = starts / sample_rate
    return frame_times, frames


def stft_magnitude(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    fft_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a simple Hann-windowed STFT magnitude spectrogram."""
    frame_times, frames = frame_signal(signal, sample_rate, frame_ms, hop_ms)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=fft_size, axis=1)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return frame_times, frequencies, np.abs(spectrum)


def magnitude_to_db(magnitude: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert magnitude values to relative dB with the max value at 0 dB."""
    db = 20.0 * np.log10(np.maximum(magnitude, eps))
    return db - np.max(db)

