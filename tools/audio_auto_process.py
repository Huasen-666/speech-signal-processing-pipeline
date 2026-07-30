from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.filters import highpass, remove_dc_offset
from speech_pipeline.normalization import amplitude_to_dbfs, dbfs_to_amplitude, peak_protect, rms_normalize_with_mask
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments

from build_dataset_manifest import B_WORDS, P_WORDS

DEFAULT_SOURCE_AUDIO = PROJECT_ROOT / "data" / "raw" / "input.wav"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "audio_auto_process_v2" / "speaker"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "audio_auto_process_v2_manifest.csv"

MANIFEST_FIELDS = [
    "speaker",
    "label",
    "word_index",
    "global_index",
    "word",
    "audio_path",
    "source_audio",
    "converted_wav",
    "preprocessed_wav",
    "source_segment_index",
    "start_sec",
    "end_sec",
    "duration_sec",
    "segment_peak_dbfs_before_clip_norm",
    "segment_rms_dbfs_before_clip_norm",
    "clip_active_rms_dbfs_before",
    "clip_gain_db",
    "clip_active_rms_dbfs_after",
    "clip_peak_dbfs_after",
    "burst_offset_sec",
    "onset_aligned",
    "warnings",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audio auto-processing for B/P word-list recordings: convert to 16 kHz mono, remove DC, apply a light high-pass, "
            "normalize session loudness before segmentation, then normalize each exported word clip into a "
            "consistent speech-RMS dBFS range."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE_AUDIO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/audio_auto_process_v2"))
    parser.add_argument("--speaker", default="speaker")
    parser.add_argument("--threshold-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--speech-threshold-dbfs", type=float, default=-40.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.08)
    parser.add_argument("--padding-sec", type=float, default=0.03)
    parser.add_argument("--preview-gap-sec", type=float, default=0.15)
    parser.add_argument("--highpass-hz", type=float, default=60.0)
    parser.add_argument("--session-target-rms-dbfs", type=float, default=-20.0)
    parser.add_argument("--clip-target-rms-dbfs", type=float, default=-20.0)
    parser.add_argument("--max-clip-gain-db", type=float, default=12.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--clip-rms-tolerance-db", type=float, default=3.0)
    parser.add_argument("--quiet-warning-dbfs", type=float, default=-30.0)
    parser.add_argument(
        "--align-onset",
        dest="align_onset",
        action="store_true",
        default=True,
        help="Re-align each exported clip to the consonant burst using a loudness-invariant detector, "
        "so the onset sits at a consistent position across sessions. On by default.",
    )
    parser.add_argument("--no-align-onset", dest="align_onset", action="store_false")
    parser.add_argument(
        "--onset-preroll-ms",
        type=float,
        default=30.0,
        help="Context kept before the detected burst when aligning a clip.",
    )
    parser.add_argument(
        "--burst-highpass-hz",
        type=float,
        default=1500.0,
        help="High-pass cutoff (Hz) for the burst-detection energy envelope.",
    )
    parser.add_argument(
        "--burst-rise-db",
        type=float,
        default=12.0,
        help="High-band energy rise (dB) above the clip baseline that marks the burst.",
    )
    parser.add_argument(
        "--burst-search-ms",
        type=float,
        default=180.0,
        help="Only look for the initial burst within this many ms from the clip start "
        "(prevents latching onto the vowel/coda; weak voiced bursts otherwise get skipped).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wav_is_16k_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getframerate() == 16000
                and handle.getsampwidth() == 2
            )
    except (wave.Error, EOFError, OSError):
        return False


def convert_to_16k_mono(source: Path, experiment_dir: Path, overwrite: bool) -> Path:
    if wav_is_16k_mono(source):
        return source

    experiment_dir.mkdir(parents=True, exist_ok=True)
    converted = experiment_dir / f"{source.stem}_16k_mono.wav"
    if converted.exists() and not overwrite:
        return converted

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(converted),
    ]
    subprocess.run(command, check=True)
    return converted


def waveform_stats(audio: np.ndarray) -> dict[str, float]:
    mono = np.asarray(audio, dtype=np.float32)
    if len(mono) == 0:
        return {"rms_dbfs": -240.0, "peak_dbfs": -240.0, "dc_offset": 0.0}
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    peak = float(np.max(np.abs(mono)))
    return {
        "rms_dbfs": round(amplitude_to_dbfs(rms), 3),
        "peak_dbfs": round(amplitude_to_dbfs(peak), 3),
        "dc_offset": round(float(np.mean(mono)), 8),
    }


def choose_threshold(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    _, frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    if args.threshold_mode == "fixed":
        return float(args.speech_threshold_dbfs), {"method": "fixed", "threshold_dbfs": float(args.speech_threshold_dbfs)}
    return estimate_speech_threshold_dbfs(frame_db)


def frame_mask_from_threshold(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if len(mono) == 0:
        return np.zeros(0, dtype=bool)

    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(mono) < frame_len:
        mono = np.pad(mono, (0, frame_len - len(mono)))

    _, frame_db = frame_rms_db(mono, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    mask = np.zeros(len(mono), dtype=bool)
    for frame_index, db_value in enumerate(frame_db):
        if db_value >= threshold_dbfs:
            start = frame_index * hop_len
            end = min(len(mask), start + frame_len)
            mask[start:end] = True
    return mask[: len(audio)]


def preprocess_session(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    before = waveform_stats(audio)
    processed = remove_dc_offset(audio)
    if args.highpass_hz > 0:
        processed = highpass(processed, sample_rate, cutoff_hz=args.highpass_hz, order=2)

    rough_threshold, rough_threshold_info = choose_threshold(processed, sample_rate, args)
    rough_mask = frame_mask_from_threshold(
        processed,
        sample_rate,
        rough_threshold,
        args.frame_ms,
        args.hop_ms,
    )
    processed, session_norm = rms_normalize_with_mask(
        processed,
        rough_mask,
        target_dbfs=args.session_target_rms_dbfs,
    )
    if len(processed):
        processed, peak_info = peak_protect(processed, max_peak_dbfs=args.max_peak_dbfs)
    else:
        peak_info = {"scale": 1.0, "scale_db": 0.0}

    after = waveform_stats(processed)
    return processed.astype(np.float32), {
        "before": before,
        "after": after,
        "dc_removed": True,
        "highpass_hz": float(args.highpass_hz),
        "rough_threshold": rough_threshold_info,
        "session_normalization": session_norm,
        "peak_protection": peak_info,
    }


def detect_word_segments(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[list[dict], dict]:
    threshold_dbfs, threshold_info = choose_threshold(audio, sample_rate, args)
    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        padding_sec=args.padding_sec,
    )
    segments, summary = detect_energy_segments(audio, sample_rate, threshold_dbfs, config=config)
    return segments, {"threshold_info": threshold_info, "summary": summary}


def active_rms_dbfs(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[float, np.ndarray, dict[str, float]]:
    if len(audio) == 0:
        return -240.0, np.zeros(0, dtype=bool), {"method": "empty"}
    threshold, threshold_info = choose_threshold(audio, sample_rate, args)
    mask = frame_mask_from_threshold(audio, sample_rate, threshold, args.frame_ms, args.hop_ms)
    if np.count_nonzero(mask) == 0:
        mask = np.ones(len(audio), dtype=bool)
        threshold_info = {**threshold_info, "mask_fallback": 1.0}
    selected = audio[mask]
    value = float(np.sqrt(np.mean(selected.astype(np.float64) ** 2) + 1e-12))
    return round(amplitude_to_dbfs(value), 3), mask, threshold_info


def detect_burst_onset(
    clip: np.ndarray,
    sample_rate: int,
    highpass_hz: float,
    rise_db: float,
    search_ms: float = 180.0,
    frame_ms: float = 4.0,
    hop_ms: float = 2.0,
) -> int | None:
    """Loudness-invariant consonant-burst detector.

    Normalizes the clip to remove level, isolates high-band energy (where the burst /
    aspiration lives), and returns the sample index of the first sharp high-band rise.
    Returns None if no clear burst is found, so the caller can fall back to the
    energy-segment start. This makes onset position consistent across sessions even
    when one recording is quieter (a quieter session otherwise crosses an absolute
    threshold later, shifting the onset).
    """
    mono = np.asarray(clip, dtype=np.float64)
    min_len = int(sample_rate * 0.02)
    if len(mono) < min_len:
        return None
    mono = mono / (np.max(np.abs(mono)) + 1e-9)  # level-invariant

    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    spectrum[freqs < highpass_hz] = 0.0
    high = np.fft.irfft(spectrum, n=len(mono))

    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, round(sample_rate * hop_ms / 1000.0))
    starts = np.arange(0, max(1, len(high) - frame_len + 1), hop_len)
    if len(starts) < 3:
        return None
    env = np.array([np.sqrt(np.mean(high[s : s + frame_len] ** 2) + 1e-12) for s in starts])
    env_db = 20.0 * np.log10(env + 1e-12)

    # Baseline (noise/quiet level) is taken over the whole clip; the burst is only
    # searched within the onset region so a weak voiced /b/ burst is not skipped in
    # favour of a louder later event (vowel offset, coda release).
    baseline = float(np.percentile(env_db, 20))
    search_frames = np.flatnonzero(starts < int(sample_rate * search_ms / 1000.0))
    if len(search_frames) == 0:
        return None
    lo = int(search_frames[0])
    hi = int(search_frames[-1]) + 1
    window_db = env_db[lo:hi]
    above = np.flatnonzero((window_db - baseline) >= rise_db)
    if len(above):
        frame_index = lo + int(above[0])
    else:
        deltas = np.diff(window_db)
        if len(deltas) == 0:
            return None
        frame_index = lo + int(np.argmax(deltas)) + 1
    return int(starts[frame_index])


def align_clip_to_burst(
    clip: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Trim the front of a clip so the burst sits `onset_preroll_ms` from the start."""
    burst = detect_burst_onset(
        clip, sample_rate, args.burst_highpass_hz, args.burst_rise_db, args.burst_search_ms
    )
    info = {"aligned": False, "burst_offset_sec": ""}
    if burst is None:
        return clip, info
    preroll = int(sample_rate * args.onset_preroll_ms / 1000.0)
    cut = max(0, burst - preroll)
    # Only trim if it leaves a usable clip (keep at least 30 ms after the cut)
    if 0 < cut < len(clip) - int(sample_rate * 0.03):
        info = {"aligned": True, "burst_offset_sec": round(burst / sample_rate, 4)}
        return clip[cut:], info
    info["burst_offset_sec"] = round(burst / sample_rate, 4)
    return clip, info


def normalize_clip_level(
    clip: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    before_stats = waveform_stats(clip)
    before_active_dbfs, mask, threshold_info = active_rms_dbfs(clip, sample_rate, args)
    desired_gain_db = args.clip_target_rms_dbfs - before_active_dbfs
    applied_gain_db = min(desired_gain_db, args.max_clip_gain_db)
    gain = dbfs_to_amplitude(applied_gain_db)
    normalized = np.asarray(clip, dtype=np.float32) * gain

    if len(normalized):
        normalized, peak_info = peak_protect(normalized, max_peak_dbfs=args.max_peak_dbfs)
    else:
        peak_info = {"scale": 1.0, "scale_db": 0.0}

    after_active_dbfs, _, _ = active_rms_dbfs(normalized, sample_rate, args)
    after_stats = waveform_stats(normalized)
    warnings = []
    if before_active_dbfs < args.quiet_warning_dbfs:
        warnings.append("very_quiet_before_clip_norm")
    if desired_gain_db > args.max_clip_gain_db:
        warnings.append("max_clip_gain_reached")
    if abs(after_active_dbfs - args.clip_target_rms_dbfs) > args.clip_rms_tolerance_db:
        warnings.append("post_clip_rms_outside_target_range")
    if float(peak_info.get("scale", 1.0)) < 0.999:
        warnings.append("peak_protection_scaled_clip")

    return normalized.astype(np.float32), {
        "before": before_stats,
        "after": after_stats,
        "before_active_rms_dbfs": before_active_dbfs,
        "after_active_rms_dbfs": after_active_dbfs,
        "threshold_info": threshold_info,
        "active_sample_ratio": round(float(np.count_nonzero(mask) / max(1, len(mask))), 6) if len(mask) else 0.0,
        "desired_gain_db": round(float(desired_gain_db), 3),
        "applied_gain_db": round(float(applied_gain_db), 3),
        "peak_protection": peak_info,
        "warnings": warnings,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_preview_from_clips(clips: list[np.ndarray], sample_rate: int, gap_sec: float) -> np.ndarray:
    gap = np.zeros(round(gap_sec * sample_rate), dtype=np.float32)
    pieces = []
    for clip in clips:
        pieces.append(np.asarray(clip, dtype=np.float32))
        pieces.append(gap)
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    converted_wav = convert_to_16k_mono(args.input, args.experiment_dir, args.overwrite)
    sample_rate, audio = read_wav_float(converted_wav)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz working WAV, got {sample_rate}: {converted_wav}")

    preprocessed, preprocessing_info = preprocess_session(audio, sample_rate, args)
    preprocessed_wav = args.experiment_dir / f"{args.input.stem}_preprocessed_v2.wav"
    write_wav_float(preprocessed_wav, sample_rate, preprocessed)

    expected_words = [("B", word) for word in B_WORDS] + [("P", word) for word in P_WORDS]
    segments, segmentation_info = detect_word_segments(preprocessed, sample_rate, args)
    if len(segments) < len(expected_words):
        raise RuntimeError(
            f"Only detected {len(segments)} segments, but expected {len(expected_words)}. "
            "Try lowering --speech-threshold-dbfs in fixed mode, reducing --merge-gap-sec, "
            "or inspect the preprocessed WAV."
        )

    selected = segments[: len(expected_words)]
    ignored = segments[len(expected_words) :]
    rows = []
    qc_rows = []
    label_counts = {"B": 0, "P": 0}
    normalized_clips = []

    for global_index, ((label, word), segment) in enumerate(zip(expected_words, selected), start=1):
        label_counts[label] += 1
        word_index = label_counts[label]
        label_dir = args.output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / f"{args.speaker}_{label}_{word_index:04d}_{word}.wav"

        raw_clip = preprocessed[int(segment["start_sample"]) : int(segment["end_sample"])]
        if args.align_onset:
            raw_clip, align_info = align_clip_to_burst(raw_clip, sample_rate, args)
        else:
            align_info = {"aligned": False, "burst_offset_sec": ""}
        normalized_clip, clip_info = normalize_clip_level(raw_clip, sample_rate, args)
        write_wav_float(out_path, sample_rate, normalized_clip)
        normalized_clips.append(normalized_clip)

        row_warnings = list(clip_info["warnings"])
        if float(segment["duration_sec"]) < args.min_duration_sec:
            row_warnings.append("segment_too_short")
        rows.append(
            {
                "speaker": args.speaker,
                "label": label,
                "word_index": f"{word_index:04d}",
                "global_index": f"{global_index:04d}",
                "word": word,
                "audio_path": str(out_path),
                "source_audio": str(args.input),
                "converted_wav": str(converted_wav),
                "preprocessed_wav": str(preprocessed_wav),
                "source_segment_index": str(segment["segment_index"]),
                "start_sec": str(segment["start_sec"]),
                "end_sec": str(segment["end_sec"]),
                "duration_sec": str(segment["duration_sec"]),
                "segment_peak_dbfs_before_clip_norm": str(segment["peak_dbfs"]),
                "segment_rms_dbfs_before_clip_norm": str(segment["rms_dbfs"]),
                "clip_active_rms_dbfs_before": str(clip_info["before_active_rms_dbfs"]),
                "clip_gain_db": str(clip_info["applied_gain_db"]),
                "clip_active_rms_dbfs_after": str(clip_info["after_active_rms_dbfs"]),
                "clip_peak_dbfs_after": str(clip_info["after"]["peak_dbfs"]),
                "burst_offset_sec": str(align_info["burst_offset_sec"]),
                "onset_aligned": "true" if align_info["aligned"] else "false",
                "warnings": ";".join(row_warnings),
            }
        )
        qc_rows.append(
            {
                "label": label,
                "word": word,
                "word_index": word_index,
                "duration_sec": float(segment["duration_sec"]),
                "before_active_rms_dbfs": clip_info["before_active_rms_dbfs"],
                "after_active_rms_dbfs": clip_info["after_active_rms_dbfs"],
                "clip_gain_db": clip_info["applied_gain_db"],
                "after_peak_dbfs": clip_info["after"]["peak_dbfs"],
                "warnings": row_warnings,
            }
        )

    write_csv(args.manifest, rows)
    preview = build_preview_from_clips(normalized_clips, sample_rate, args.preview_gap_sec)
    preview_path = args.experiment_dir / "audio_auto_process_labeled_sequence.wav"
    write_wav_float(preview_path, sample_rate, preview)

    warning_counts: dict[str, int] = {}
    for qc_row in qc_rows:
        for warning in qc_row["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

    after_rms = np.array([float(row["after_active_rms_dbfs"]) for row in qc_rows], dtype=np.float64)
    before_rms = np.array([float(row["before_active_rms_dbfs"]) for row in qc_rows], dtype=np.float64)
    summary = {
        "speaker": args.speaker,
        "source_audio": str(args.input),
        "converted_wav": str(converted_wav),
        "preprocessed_wav": str(preprocessed_wav),
        "output_root": str(args.output_root),
        "manifest": str(args.manifest),
        "preview_wav": str(preview_path),
        "expected_word_count": len(expected_words),
        "detected_segment_count": len(segments),
        "assigned_word_count": len(selected),
        "ignored_end_segment_count": len(ignored),
        "label_counts": label_counts,
        "preprocessing": preprocessing_info,
        "segmentation": segmentation_info,
        "clip_level_qc": {
            "target_rms_dbfs": float(args.clip_target_rms_dbfs),
            "before_active_rms_dbfs_mean": round(float(np.mean(before_rms)), 3) if len(before_rms) else None,
            "before_active_rms_dbfs_std": round(float(np.std(before_rms)), 3) if len(before_rms) else None,
            "after_active_rms_dbfs_mean": round(float(np.mean(after_rms)), 3) if len(after_rms) else None,
            "after_active_rms_dbfs_std": round(float(np.std(after_rms)), 3) if len(after_rms) else None,
            "after_active_rms_dbfs_min": round(float(np.min(after_rms)), 3) if len(after_rms) else None,
            "after_active_rms_dbfs_max": round(float(np.max(after_rms)), 3) if len(after_rms) else None,
            "warning_counts": warning_counts,
        },
        "onset_alignment": {
            "enabled": bool(args.align_onset),
            "preroll_ms": float(args.onset_preroll_ms),
            "burst_highpass_hz": float(args.burst_highpass_hz),
            "burst_search_ms": float(args.burst_search_ms),
            "aligned_count": sum(1 for row in rows if row["onset_aligned"] == "true"),
            "fallback_count": sum(1 for row in rows if row["onset_aligned"] == "false"),
        },
    }
    summary_path = args.experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved manifest: {args.manifest}")
    print(f"Saved B clips: {args.output_root / 'B'}")
    print(f"Saved P clips: {args.output_root / 'P'}")
    print(f"Saved preprocessed WAV: {preprocessed_wav}")
    print(f"Saved preview WAV: {preview_path}")


if __name__ == "__main__":
    main()
