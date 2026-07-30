from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.normalization import amplitude_to_dbfs, peak, rms
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments


WORD_GROUPS = [
    ("pronouns", ["i", "you", "he", "she", "we", "they", "it"]),
    ("auxiliaries", ["am", "are", "is", "was", "were", "can", "will"]),
    ("articles", ["a", "an", "the"]),
    ("conjunctions", ["and", "but"]),
    ("determiners", ["my", "your", "his", "her", "their", "this", "that"]),
    ("prepositions_particles", ["at", "for", "from", "in", "of", "on", "to", "up", "with"]),
]

EXPECTED_WORDS = [(category, word) for category, words in WORD_GROUPS for word in words]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split David's connected/function words by fixed list order.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT.parent / "Cleaned Sound" / "David" / "connected words.wav",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed/david_connected_words"),
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=PROJECT_ROOT.parent / "word_library" / "function_words_david",
    )
    parser.add_argument("--threshold-dbfs", type=float, default=-20.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.08)
    parser.add_argument("--padding-sec", type=float, default=0.03)
    parser.add_argument("--block-gap-sec", type=float, default=8.0)
    parser.add_argument("--take-index", type=int, default=1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_blocks(segments: list[dict], block_gap_sec: float) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    current: list[dict] = []
    last_end = None
    for segment in segments:
        if last_end is not None and float(segment["start_sec"]) - last_end > block_gap_sec:
            if current:
                blocks.append(current)
            current = []
        current.append(segment)
        last_end = float(segment["end_sec"])
    if current:
        blocks.append(current)
    return blocks


def merge_closest_segments(segments: list[dict], target_count: int) -> tuple[list[dict], list[dict[str, str]]]:
    merged = [dict(segment) for segment in segments]
    merge_notes: list[dict[str, str]] = []
    while len(merged) > target_count:
        gaps = [
            float(merged[idx + 1]["start_sec"]) - float(merged[idx]["end_sec"])
            for idx in range(len(merged) - 1)
        ]
        merge_idx = int(np.argmin(gaps))
        left = merged[merge_idx]
        right = merged[merge_idx + 1]
        combined = {
            **left,
            "end_sec": right["end_sec"],
            "end_sample": right["end_sample"],
            "duration_sec": round(float(right["end_sec"]) - float(left["start_sec"]), 3),
            "peak_dbfs": max(float(left["peak_dbfs"]), float(right["peak_dbfs"])),
            "rms_dbfs": "",
        }
        merge_notes.append(
            {
                "merged_position_1based": str(merge_idx + 1),
                "left_start_sec": f"{float(left['start_sec']):.3f}",
                "left_end_sec": f"{float(left['end_sec']):.3f}",
                "right_start_sec": f"{float(right['start_sec']):.3f}",
                "right_end_sec": f"{float(right['end_sec']):.3f}",
                "gap_sec": f"{gaps[merge_idx]:.3f}",
            }
        )
        merged[merge_idx : merge_idx + 2] = [combined]
    return merged, merge_notes


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    sample_rate, audio = read_wav_float(args.input)
    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        padding_sec=args.padding_sec,
    )
    segments, segmentation_summary = detect_energy_segments(audio, sample_rate, args.threshold_dbfs, config=config)
    blocks = split_blocks(segments, args.block_gap_sec)
    if not blocks:
        raise RuntimeError("No speech blocks detected.")
    if args.take_index < 1 or args.take_index > len(blocks):
        raise ValueError(f"--take-index must be between 1 and {len(blocks)}.")

    expected_count = len(EXPECTED_WORDS)
    selected_block = blocks[args.take_index - 1]
    if len(selected_block) < expected_count:
        raise RuntimeError(
            f"Selected take has only {len(selected_block)} detected segments; expected {expected_count}."
        )
    labeled_segments, merge_notes = merge_closest_segments(selected_block, expected_count)
    if len(labeled_segments) != expected_count:
        raise RuntimeError(f"Expected {expected_count} labeled segments, got {len(labeled_segments)}.")

    clip_root = args.outdir / "clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    args.library_root.mkdir(parents=True, exist_ok=True)

    rows = []
    preview_parts = []
    preview_gap = np.zeros(round(sample_rate * 0.25), dtype=np.float32)
    for index, ((category, word), segment) in enumerate(zip(EXPECTED_WORDS, labeled_segments), start=1):
        start_sample = int(segment["start_sample"])
        end_sample = int(segment["end_sample"])
        clip = audio[start_sample:end_sample].astype(np.float32)
        clip_name = f"david_function_{index:03d}_{word}.wav"

        clip_path = clip_root / category / word / clip_name
        library_path = args.library_root / category / word / f"{word}.wav"
        write_wav_float(clip_path, sample_rate, clip)
        write_wav_float(library_path, sample_rate, clip)
        preview_parts.extend([clip, preview_gap])

        rows.append(
            {
                "index": f"{index:03d}",
                "speaker": "david",
                "take": str(args.take_index),
                "category": category,
                "word": word,
                "label": "FUNCTION",
                "audio_path": relpath(clip_path),
                "library_audio_path": str(library_path),
                "source_audio": str(args.input),
                "start_sec": f"{start_sample / sample_rate:.3f}",
                "end_sec": f"{end_sample / sample_rate:.3f}",
                "duration_sec": f"{len(clip) / sample_rate:.3f}",
                "rms_dbfs": f"{amplitude_to_dbfs(rms(clip)):.3f}",
                "peak_dbfs": f"{amplitude_to_dbfs(peak(clip)):.3f}",
            }
        )

    manifest_path = args.outdir / "david_connected_words_manifest.csv"
    merge_notes_path = args.outdir / "merge_notes.csv"
    preview_path = args.outdir / "david_connected_words_labeled_sequence.wav"
    summary_path = args.outdir / "summary.json"
    write_csv(manifest_path, rows)
    if merge_notes:
        write_csv(merge_notes_path, merge_notes)
    if preview_parts:
        write_wav_float(preview_path, sample_rate, np.concatenate(preview_parts[:-1]).astype(np.float32))

    summary = {
        "input": str(args.input),
        "sample_rate_hz": sample_rate,
        "duration_sec": round(len(audio) / sample_rate, 3),
        "threshold_dbfs": args.threshold_dbfs,
        "expected_word_count": expected_count,
        "detected_segment_count_total": len(segments),
        "detected_block_count": len(blocks),
        "detected_block_counts": [len(block) for block in blocks],
        "selected_take_index": args.take_index,
        "selected_take_segment_count_before_merge": len(selected_block),
        "labeled_clip_count": len(rows),
        "merge_count": len(merge_notes),
        "manifest_csv": str(manifest_path),
        "merge_notes_csv": str(merge_notes_path) if merge_notes else "",
        "preview_audio": str(preview_path),
        "clip_root": str(clip_root),
        "library_root": str(args.library_root),
        "segmentation_summary": segmentation_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
