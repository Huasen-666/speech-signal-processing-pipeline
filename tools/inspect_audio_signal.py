import argparse
import csv
import json
import math
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("data/raw/example.wav")
'''
把 WAV 里的原始 PCM 二进制采样数据
转换成 Python/Numpy 里好处理的浮点 waveform
bytes / integer samples--------------float samples in roughly [-1.0, 1.0]
'''

def pcm_to_float(raw: bytes, sample_width: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_width == 1:
        integer = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        scale = 128.0
    elif sample_width == 2:
        integer = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        scale = 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((bytes_.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = bytes_
        sign = bytes_[:, 2] >= 128
        padded[sign, 3] = 255
        integer = padded.view("<i4").reshape(-1)
        scale = 8388608.0
    elif sample_width == 4:
        integer = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    integer = integer.reshape(-1, channels)
    audio = integer.astype(np.float32) / scale#除掉scale让他们的值单位化成1
    return integer, audio


def dbfs(value: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(float(value), eps))

'''
平均有效能量
'''
def frame_rms_db(mono: np.ndarray, sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(mono) < frame_len:
        mono = np.pad(mono, (0, frame_len - len(mono)))

    starts = np.arange(0, len(mono) - frame_len + 1, hop_len)
    power = mono.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    frame_power = (cumulative[starts + frame_len] - cumulative[starts]) / frame_len
    frame_rms = np.sqrt(np.maximum(frame_power, 0.0))
    frame_db = np.array([dbfs(v) for v in frame_rms], dtype=np.float32)
    frame_time = starts / sample_rate
    return frame_time, frame_db

'''
如果某一帧 RMS 能量高于阈值，比如 -45 dBFS 就认为它可能是语音
阈值越低，检测到的“语音”越多；阈值越高，检测更严格，但可能漏掉轻声词。这个就是 VAD 的 precision/recall tradeoff
'''
def energy_segments(
    frame_time: np.ndarray,
    frame_db: np.ndarray,
    frame_ms: float,
    threshold_dbfs: float,
    merge_gap_sec: float,
    min_duration_sec: float,
) -> list[dict[str, float]]:
    voiced = frame_db >= threshold_dbfs
    segments = []
    start = None
    frame_dur = frame_ms / 1000.0

    for idx, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = idx
        elif not is_voiced and start is not None:
            segments.append([frame_time[start], frame_time[idx - 1] + frame_dur])
            start = None
    if start is not None:
        segments.append([frame_time[start], frame_time[-1] + frame_dur])

    merged = []
    for seg_start, seg_end in segments:
        if not merged or seg_start - merged[-1][1] > merge_gap_sec:
            merged.append([float(seg_start), float(seg_end)])
        else:
            merged[-1][1] = float(seg_end)

    return [
        {
            "start_sec": round(seg_start, 3),
            "end_sec": round(seg_end, 3),
            "duration_sec": round(seg_end - seg_start, 3),
        }
        for seg_start, seg_end in merged
        if seg_end - seg_start >= min_duration_sec
    ]


def envelope_for_plot(mono: np.ndarray, sample_rate: int, bucket_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    bucket = max(1, round(sample_rate * bucket_ms / 1000.0))
    usable = (len(mono) // bucket) * bucket
    if usable == 0:
        return np.array([0.0]), np.array([float(np.max(np.abs(mono)))])
    reshaped = np.abs(mono[:usable]).reshape(-1, bucket)
    envelope = reshaped.max(axis=1)
    times = np.arange(len(envelope)) * bucket / sample_rate
    return times, envelope


def write_segments_csv(path: Path, segments: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_sec", "end_sec", "duration_sec"])
        writer.writeheader()
        writer.writerows(segments)


def plot_overview(
    out_path: Path,
    mono: np.ndarray,
    sample_rate: int,
    frame_time: np.ndarray,
    frame_db: np.ndarray,
    threshold_dbfs: float,
) -> None:
    env_time, envelope = envelope_for_plot(mono, sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    axes[0].plot(env_time, envelope, linewidth=0.7)
    axes[0].set_title("Peak amplitude envelope")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(0, min(1.0, max(0.05, float(envelope.max()) * 1.1)))

    axes[1].plot(frame_time, frame_db, linewidth=0.7)
    axes[1].axhline(threshold_dbfs, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Short-time RMS energy")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("dBFS")
    axes[1].set_ylim(-90, 0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_spectrogram(out_path: Path, mono: np.ndarray, sample_rate: int, preview_sec: float) -> None:
    preview = mono[: int(sample_rate * preview_sec)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.specgram(preview, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    ax.set_title(f"Spectrogram preview, first {preview_sec:g} seconds")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a WAV file before speech cleaning.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input WAV file.")
    parser.add_argument("--outdir", type=Path, help="Directory for JSON, CSV, and PNG outputs.")
    parser.add_argument("--frame-ms", type=float, default=25.0, help="RMS analysis frame length.")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="RMS analysis hop length.")
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.20)
    parser.add_argument("--min-segment-sec", type=float, default=0.15)
    parser.add_argument("--spectrogram-preview-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input
    if not source.exists():
        raise FileNotFoundError(source)

    outdir = args.outdir or source.with_name(f"{source.stem}_inspection")
    outdir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(source), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        compression = wav.getcomptype()
        raw = wav.readframes(frames)

    integer, audio = pcm_to_float(raw, sample_width, channels)
    mono = audio[:, 0] if channels == 1 else audio.mean(axis=1)

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    dc_offset = float(np.mean(mono))
    duration = frames / sample_rate
    max_integer = (2 ** (8 * sample_width - 1)) - 1
    clipping_samples = int(np.count_nonzero(np.abs(integer.astype(np.int64)) >= max_integer))

    frame_time, frame_db = frame_rms_db(mono, sample_rate, args.frame_ms, args.hop_ms)
    segments = energy_segments(
        frame_time,
        frame_db,
        args.frame_ms,
        args.silence_threshold_dbfs,
        args.merge_gap_sec,
        args.min_segment_sec,
    )

    voiced_duration = sum(segment["duration_sec"] for segment in segments)
    report = {
        "input": str(source),
        "format": {
            "channels": channels,
            "sample_width_bytes": sample_width,
            "sample_rate_hz": sample_rate,
            "num_frames": frames,
            "duration_sec": round(duration, 3),
            "compression": compression,
        },
        "level": {
            "peak_dbfs": round(dbfs(peak), 2),
            "rms_dbfs": round(dbfs(rms), 2),
            "dc_offset": round(dc_offset, 8),
            "clipping_samples": clipping_samples,
            "clipping_ratio": round(clipping_samples / max(1, integer.size), 8),
        },
        "short_time_energy": {
            "frame_ms": args.frame_ms,
            "hop_ms": args.hop_ms,
            "silence_threshold_dbfs": args.silence_threshold_dbfs,
            "p05_dbfs": round(float(np.percentile(frame_db, 5)), 2),
            "p10_dbfs": round(float(np.percentile(frame_db, 10)), 2),
            "p50_dbfs": round(float(np.percentile(frame_db, 50)), 2),
            "p90_dbfs": round(float(np.percentile(frame_db, 90)), 2),
            "p95_dbfs": round(float(np.percentile(frame_db, 95)), 2),
        },
        "rough_energy_vad": {
            "segment_count": len(segments),
            "voiced_duration_sec": round(voiced_duration, 3),
            "voiced_ratio": round(voiced_duration / duration, 4),
            "merge_gap_sec": args.merge_gap_sec,
            "min_segment_sec": args.min_segment_sec,
        },
    }

    report_path = outdir / "inspection_report.json"
    segments_path = outdir / "energy_vad_segments.csv"
    overview_path = outdir / "overview.png"
    spectrogram_path = outdir / "spectrogram_preview.png"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_segments_csv(segments_path, segments)
    plot_overview(overview_path, mono, sample_rate, frame_time, frame_db, args.silence_threshold_dbfs)
    plot_spectrogram(spectrogram_path, mono, sample_rate, args.spectrogram_preview_sec)

    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")
    print(f"Saved rough VAD CSV: {segments_path}")
    print(f"Saved overview plot: {overview_path}")
    print(f"Saved spectrogram preview: {spectrogram_path}")


if __name__ == "__main__":
    main()
