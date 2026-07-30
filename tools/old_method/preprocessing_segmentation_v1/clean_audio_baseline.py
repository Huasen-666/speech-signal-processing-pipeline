import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.filters import bandpass, notch, remove_dc_offset
from speech_pipeline.normalization import peak_protect, rms_normalize, rms_normalize_with_mask
from speech_pipeline.quality import (
    estimate_speech_threshold_dbfs,
    frame_rms_db,
    quality_warnings,
    summarize_audio,
)


def envelope_for_plot(audio: np.ndarray, sample_rate: int, bucket_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    bucket = max(1, round(sample_rate * bucket_ms / 1000.0))
    usable = (len(audio) // bucket) * bucket
    if usable == 0:
        return np.array([0.0]), np.array([0.0])
    envelope = np.abs(audio[:usable]).reshape(-1, bucket).max(axis=1)
    times = np.arange(len(envelope)) * bucket / sample_rate
    return times, envelope


def plot_before_after(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
    speech_threshold_dbfs: float,
) -> None:
    before_time, before_env = envelope_for_plot(before, sample_rate)
    after_time, after_env = envelope_for_plot(after, sample_rate)
    before_frame_time, before_frame_db = frame_rms_db(before, sample_rate)
    after_frame_time, after_frame_db = frame_rms_db(after, sample_rate)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex="col")
    axes[0, 0].plot(before_time, before_env, linewidth=0.7)
    axes[0, 0].set_title("Before: peak amplitude envelope")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_ylim(0, 1.0)

    axes[0, 1].plot(after_time, after_env, linewidth=0.7)
    axes[0, 1].set_title("After: peak amplitude envelope")
    axes[0, 1].set_ylim(0, 1.0)

    axes[1, 0].plot(before_frame_time, before_frame_db, linewidth=0.7)
    axes[1, 0].axhline(speech_threshold_dbfs, color="red", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Before: short-time RMS energy")
    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 0].set_ylabel("dBFS")
    axes[1, 0].set_ylim(-90, 0)

    axes[1, 1].plot(after_frame_time, after_frame_db, linewidth=0.7)
    axes[1, 1].axhline(speech_threshold_dbfs, color="red", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("After: short-time RMS energy")
    axes[1, 1].set_xlabel("Time (seconds)")
    axes[1, 1].set_ylim(-90, 0)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def clean_audio(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    notch_hz: float,
    notch_q: float,
    target_rms_dbfs: float,
    max_peak_dbfs: float,
    speech_threshold_dbfs: float,
    threshold_info: dict,
    normalization_mode: str,
) -> tuple[np.ndarray, dict]:
    stages = {}

    cleaned = remove_dc_offset(audio)
    stages["dc_removed_mean"] = round(float(np.mean(cleaned)), 8)

    cleaned = bandpass(cleaned, sample_rate, low_hz=low_hz, high_hz=high_hz)
    stages["bandpass"] = {
        "low_hz": float(low_hz),
        "high_hz": float(min(high_hz, sample_rate / 2.0 * 0.95)),
    }

    if notch_hz > 0:
        cleaned = notch(cleaned, sample_rate, frequency_hz=notch_hz, q=notch_q)
        stages["notch"] = {"frequency_hz": float(notch_hz), "q": float(notch_q)}
    else:
        stages["notch"] = None

    if normalization_mode == "speech_only":
        frame_times, frame_db = frame_rms_db(cleaned, sample_rate)
        hop_len = round(sample_rate * 10.0 / 1000.0)
        frame_len = round(sample_rate * 25.0 / 1000.0)
        mask = np.zeros(len(cleaned), dtype=bool)
        voiced_frame_indexes = np.flatnonzero(frame_db >= speech_threshold_dbfs)
        starts = np.round(frame_times[voiced_frame_indexes] * sample_rate).astype(int)
        for start in starts:
            mask[start : min(len(mask), start + frame_len)] = True
        cleaned, rms_info = rms_normalize_with_mask(cleaned, mask, target_dbfs=target_rms_dbfs)
        rms_info["speech_threshold_dbfs"] = float(speech_threshold_dbfs)
        rms_info["speech_threshold_info"] = threshold_info
        rms_info["frame_ms"] = 25.0
        rms_info["hop_ms"] = round(hop_len / sample_rate * 1000.0, 3)
    else:
        cleaned, rms_info = rms_normalize(cleaned, target_dbfs=target_rms_dbfs)
        rms_info["mode"] = "full_audio"

    cleaned, peak_info = peak_protect(cleaned, max_peak_dbfs=max_peak_dbfs)
    stages["rms_normalization"] = rms_info
    stages["peak_protection"] = peak_info

    return cleaned.astype(np.float32), stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative DSP speech cleaning baseline with QA report.")
    parser.add_argument("--input", type=Path, required=True, help="Input 16-bit PCM WAV.")
    parser.add_argument("--output", type=Path, help="Output cleaned WAV.")
    parser.add_argument("--outdir", type=Path, help="Output directory for QA report and plots.")
    parser.add_argument("--low-hz", type=float, default=80.0)
    parser.add_argument("--high-hz", type=float, default=7600.0)
    parser.add_argument("--notch-hz", type=float, default=0.0, help="Set to 60 or 120 for tonal hum removal.")
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument("--target-rms-dbfs", type=float, default=-24.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--speech-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--threshold-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--adaptive-noise-margin-db", type=float, default=15.0)
    parser.add_argument("--adaptive-speech-margin-db", type=float, default=25.0)
    parser.add_argument("--normalization-mode", choices=["speech_only", "full_audio"], default="speech_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    output = args.output or Path("data/processed") / f"{args.input.stem}_clean.wav"
    outdir = args.outdir or Path("experiments") / f"{args.input.stem}_cleaning"
    outdir.mkdir(parents=True, exist_ok=True)

    sample_rate, audio = read_wav_float(args.input)
    _, raw_frame_db = frame_rms_db(audio, sample_rate)
    if args.threshold_mode == "adaptive":
        speech_threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(
            raw_frame_db,
            noise_margin_db=args.adaptive_noise_margin_db,
            speech_margin_db=args.adaptive_speech_margin_db,
        )
    else:
        speech_threshold_dbfs = args.speech_threshold_dbfs
        threshold_info = {"method": "fixed", "threshold_dbfs": float(speech_threshold_dbfs)}

    before = summarize_audio(
        audio,
        sample_rate,
        speech_threshold_dbfs=speech_threshold_dbfs,
        threshold_info=threshold_info,
    )
    cleaned, stages = clean_audio(
        audio=audio,
        sample_rate=sample_rate,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        notch_hz=args.notch_hz,
        notch_q=args.notch_q,
        target_rms_dbfs=args.target_rms_dbfs,
        max_peak_dbfs=args.max_peak_dbfs,
        speech_threshold_dbfs=speech_threshold_dbfs,
        threshold_info=threshold_info,
        normalization_mode=args.normalization_mode,
    )
    after = summarize_audio(
        cleaned,
        sample_rate,
        speech_threshold_dbfs=speech_threshold_dbfs,
        threshold_info=threshold_info,
    )

    write_wav_float(output, sample_rate, cleaned)
    plot_before_after(outdir / "before_after_overview.png", audio, cleaned, sample_rate, speech_threshold_dbfs)

    report = {
        "input": str(args.input),
        "output": str(output),
        "sample_rate_hz": sample_rate,
        "pipeline": "conservative_dsp_baseline",
        "threshold_mode": args.threshold_mode,
        "speech_threshold_dbfs": speech_threshold_dbfs,
        "speech_threshold_info": threshold_info,
        "stages": stages,
        "before": before,
        "after": after,
        "warnings": {
            "before": quality_warnings(before),
            "after": quality_warnings(after),
        },
    }
    report_path = outdir / "before_after_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved cleaned WAV: {output}")
    print(f"Saved QA report: {report_path}")
    print(f"Saved overview plot: {outdir / 'before_after_overview.png'}")


if __name__ == "__main__":
    main()
