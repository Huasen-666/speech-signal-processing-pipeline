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
        raise ValueError("This learning script currently expects 16-bit PCM WAV.")

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    audio = audio.reshape(-1, channels)
    mono = audio[:, 0] if channels == 1 else audio.mean(axis=1)
    return sample_rate, mono


def dbfs(value: float, eps: float = 1e-12) -> float:
    return 20.0 * np.log10(max(float(value), eps))


def short_time_rms(mono: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float):
    frame_len = round(sample_rate * frame_ms / 1000.0)
    hop_len = round(sample_rate * hop_ms / 1000.0)
    starts = np.arange(0, len(mono) - frame_len + 1, hop_len)
    power = mono.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    rms = np.sqrt((cumulative[starts + frame_len] - cumulative[starts]) / frame_len)
    return starts / sample_rate, np.array([dbfs(v) for v in rms])


def moving_rms_db(mono: np.ndarray, sample_rate: int, window_sec: float) -> tuple[np.ndarray, np.ndarray]:
    window = max(1, round(sample_rate * window_sec))
    hop = max(1, round(sample_rate * 0.25))
    starts = np.arange(0, len(mono) - window + 1, hop)
    power = mono.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    rms = np.sqrt((cumulative[starts + window] - cumulative[starts]) / window)
    return starts / sample_rate, np.array([dbfs(v) for v in rms])


def pick_windows(mono: np.ndarray, sample_rate: int, window_sec: float, speech_threshold_dbfs: float):
    duration = len(mono) / sample_rate
    long_times, long_db = moving_rms_db(mono, sample_rate, window_sec)
    frame_times, frame_db = short_time_rms(mono, sample_rate, 25.0, 10.0)

    loud_idx = int(np.argmax(long_db))
    quiet_idx = int(np.argmin(long_db))

    speech_candidates = np.flatnonzero(frame_db >= speech_threshold_dbfs)
    if len(speech_candidates):
        first_speech_start = max(0.0, float(frame_times[speech_candidates[0]]) - 0.5)
    else:
        first_speech_start = 0.0

    # A mid-energy speech window is useful because the loudest window can be a burst or artifact.
    speech_db = frame_db[speech_candidates] if len(speech_candidates) else frame_db
    target_db = float(np.percentile(speech_db, 70))
    target_idx = int(np.argmin(np.abs(frame_db - target_db)))
    typical_start = max(0.0, min(duration - window_sec, float(frame_times[target_idx]) - 1.0))

    windows = [
        {"name": "first_speech", "start_sec": first_speech_start, "duration_sec": window_sec},
        {"name": "typical_speech", "start_sec": typical_start, "duration_sec": window_sec},
        {"name": "loudest_5s", "start_sec": float(long_times[loud_idx]), "duration_sec": window_sec},
        {"name": "quietest_5s", "start_sec": float(long_times[quiet_idx]), "duration_sec": window_sec},
    ]

    for window in windows:
        window["start_sec"] = round(max(0.0, min(duration - window_sec, window["start_sec"])), 3)
        start = round(window["start_sec"] * sample_rate)
        end = round((window["start_sec"] + window_sec) * sample_rate)
        chunk = mono[start:end]
        window["peak_dbfs"] = round(dbfs(np.max(np.abs(chunk))), 2)
        window["rms_dbfs"] = round(dbfs(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))), 2)

    return windows


def plot_window(path: Path, mono: np.ndarray, sample_rate: int, start_sec: float, duration_sec: float, title: str) -> None:
    start = round(start_sec * sample_rate)
    end = min(len(mono), round((start_sec + duration_sec) * sample_rate))
    chunk = mono[start:end]
    time_axis = start_sec + np.arange(len(chunk)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    axes[0].plot(time_axis, chunk, linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_title(f"{title}: waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-1.0, 1.0)

    axes[1].specgram(chunk, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    axes[1].set_title(f"{title}: spectrogram")
    axes[1].set_xlabel("Time inside window (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create zoomed waveform and spectrogram windows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--speech-threshold-dbfs", type=float, default=-45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input
    if not source.exists():
        raise FileNotFoundError(source)

    outdir = args.outdir or source.with_name(f"{source.stem}_windows")
    outdir.mkdir(parents=True, exist_ok=True)

    sample_rate, mono = read_wav_float(source)
    windows = pick_windows(mono, sample_rate, args.window_sec, args.speech_threshold_dbfs)

    for window in windows:
        plot_window(
            outdir / f"{window['name']}.png",
            mono,
            sample_rate,
            window["start_sec"],
            window["duration_sec"],
            window["name"],
        )

    manifest = {
        "input": str(source),
        "sample_rate_hz": sample_rate,
        "window_sec": args.window_sec,
        "windows": windows,
    }
    (outdir / "windows_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved zoom plots in: {outdir}")


if __name__ == "__main__":
    main()
