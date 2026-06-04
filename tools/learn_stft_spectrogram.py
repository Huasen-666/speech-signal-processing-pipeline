import argparse
import json
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("data/raw/example.wav")


def read_wav_float(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width != 2:
        raise ValueError("This learning script expects 16-bit PCM WAV.")

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    audio = audio.reshape(-1, channels)
    mono = audio[:, 0] if channels == 1 else audio.mean(axis=1)
    return sample_rate, mono


def frame_signal(signal: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame_len = round(sample_rate * frame_ms / 1000.0)
    hop_len = round(sample_rate * hop_ms / 1000.0)

    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    starts = np.arange(0, len(signal) - frame_len + 1, hop_len)
    frames = np.stack([signal[start : start + frame_len] for start in starts])
    frame_times = starts / sample_rate
    return frame_times, frames


def stft(
    signal: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    fft_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_times, frames = frame_signal(signal, sample_rate, frame_ms, hop_ms)
    frame_len = frames.shape[1]
    window = np.hanning(frame_len).astype(np.float32)

    windowed_frames = frames * window[None, :]
    spectrum = np.fft.rfft(windowed_frames, n=fft_size, axis=1)
    magnitude = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    return frame_times, frequencies, magnitude


def magnitude_to_db(magnitude: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(magnitude, eps))
    return db - np.max(db)


def plot_waveform(path: Path, segment: np.ndarray, sample_rate: int, start_sec: float) -> None:
    time = start_sec + np.arange(len(segment)) / sample_rate
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(time, segment, linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title("Waveform segment used for STFT")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_window_example(path: Path, sample_rate: int, frame_ms: float) -> None:
    frame_len = round(sample_rate * frame_ms / 1000.0)
    time_ms = np.arange(frame_len) / sample_rate * 1000.0
    rectangular = np.ones(frame_len)
    hann = np.hanning(frame_len)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_ms, rectangular, label="Rectangular window", linewidth=1.2)
    ax.plot(time_ms, hann, label="Hann window", linewidth=1.5)
    ax.set_title(f"Window shape for one {frame_ms:g} ms frame")
    ax.set_xlabel("Time inside frame (ms)")
    ax.set_ylabel("Gain")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_single_frame_spectrum(
    path: Path,
    segment: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    fft_size: int,
    frame_index: int,
) -> dict[str, float]:
    _, frames = frame_signal(segment, sample_rate, frame_ms, frame_ms)
    frame_index = min(max(frame_index, 0), len(frames) - 1)
    frame = frames[frame_index]
    window = np.hanning(len(frame)).astype(np.float32)
    windowed_frame = frame * window

    raw_spectrum = np.abs(np.fft.rfft(frame, n=fft_size))
    windowed_spectrum = np.abs(np.fft.rfft(windowed_frame, n=fft_size))
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    raw_db = magnitude_to_db(raw_spectrum)
    windowed_db = magnitude_to_db(windowed_spectrum)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    time_ms = np.arange(len(frame)) / sample_rate * 1000.0
    axes[0].plot(time_ms, frame, label="Raw frame", linewidth=0.8)
    axes[0].plot(time_ms, windowed_frame, label="After Hann window", linewidth=0.9)
    axes[0].set_title("One frame before and after windowing")
    axes[0].set_xlabel("Time inside frame (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    axes[1].plot(freqs, raw_db, label="Raw frame FFT", linewidth=0.8)
    axes[1].plot(freqs, windowed_db, label="Windowed frame FFT", linewidth=0.9)
    axes[1].set_title("Frequency spectrum of one frame")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB, relative)")
    axes[1].set_ylim(-100, 5)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {
        "frame_index": frame_index,
        "frame_peak_amplitude": round(float(np.max(np.abs(frame))), 6),
        "windowed_frame_peak_amplitude": round(float(np.max(np.abs(windowed_frame))), 6),
    }


def plot_stft_spectrogram(
    path: Path,
    segment: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    fft_size: int,
) -> dict[str, float]:
    times, freqs, magnitude = stft(segment, sample_rate, frame_ms, hop_ms, fft_size)
    mag_db = magnitude_to_db(magnitude)

    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(
        mag_db.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1] + frame_ms / 1000.0, freqs[0], freqs[-1]],
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_title("Manual STFT spectrogram")
    ax.set_xlabel("Time inside segment (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Magnitude (dB, relative)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {
        "num_frames": int(magnitude.shape[0]),
        "num_frequency_bins": int(magnitude.shape[1]),
        "frequency_resolution_hz": round(sample_rate / fft_size, 3),
        "time_step_sec": round(hop_ms / 1000.0, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn STFT by building a spectrogram manually.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--start-sec", type=float, default=16.5)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--frame-index", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input
    if not source.exists():
        raise FileNotFoundError(source)

    outdir = args.outdir or source.with_name(f"{source.stem}_stft_learning")
    outdir.mkdir(parents=True, exist_ok=True)

    sample_rate, mono = read_wav_float(source)
    start = round(args.start_sec * sample_rate)
    end = min(len(mono), round((args.start_sec + args.duration_sec) * sample_rate))
    segment = mono[start:end]

    plot_waveform(outdir / "01_waveform_segment.png", segment, sample_rate, args.start_sec)
    plot_window_example(outdir / "02_window_shape.png", sample_rate, args.frame_ms)
    frame_info = plot_single_frame_spectrum(
        outdir / "03_single_frame_spectrum.png",
        segment,
        sample_rate,
        args.frame_ms,
        args.fft_size,
        args.frame_index,
    )
    stft_info = plot_stft_spectrogram(
        outdir / "04_manual_stft_spectrogram.png",
        segment,
        sample_rate,
        args.frame_ms,
        args.hop_ms,
        args.fft_size,
    )

    frame_len = round(sample_rate * args.frame_ms / 1000.0)
    hop_len = round(sample_rate * args.hop_ms / 1000.0)
    manifest = {
        "input": str(source),
        "sample_rate_hz": sample_rate,
        "segment": {
            "start_sec": args.start_sec,
            "duration_sec": round(len(segment) / sample_rate, 3),
        },
        "stft_parameters": {
            "frame_ms": args.frame_ms,
            "frame_len_samples": frame_len,
            "hop_ms": args.hop_ms,
            "hop_len_samples": hop_len,
            "fft_size": args.fft_size,
        },
        "single_frame": frame_info,
        "spectrogram": stft_info,
    }

    (outdir / "stft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved STFT learning plots in: {outdir}")


if __name__ == "__main__":
    main()
