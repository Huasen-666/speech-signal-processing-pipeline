from pathlib import Path
import wave

import numpy as np


def read_wav_float(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file and return sample rate plus mono float waveform."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit samples")

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    audio = audio.reshape(-1, channels)
    mono = audio[:, 0] if channels == 1 else audio.mean(axis=1)
    return sample_rate, mono


def write_wav_float(path: str | Path, sample_rate: int, audio: np.ndarray) -> None:
    """Write a mono float waveform as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim != 1:
        raise ValueError("write_wav_float expects a mono 1-D waveform")

    clipped = np.clip(mono, -1.0, 1.0)
    integer = (clipped * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(integer.tobytes())
