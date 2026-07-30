import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments

from render_b_initial_stub_library_demo import (
    adapt_stub_level,
    detect_active_start,
    fit_stub_to_length,
    list_stubs,
    load_stub_for_group,
    prepare_stub_for_insert,
)
from render_bascom_content_heavy_reading_demo import LONG_STORY_LINES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script-guided in-place B/P initial consonant enhancement for a long recording.")
    parser.add_argument("--input", type=Path, required=True, help="Input 16-bit PCM WAV.")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/scripted_long_recording_enhancement"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--p-manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--function-root", type=Path, default=PROJECT_ROOT.parent / "word_library" / "function_words")
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--speaker-for-word-durations", default="bascom")
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.45)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument("--padding-sec", type=float, default=0.03)
    parser.add_argument("--line-padding-sec", type=float, default=0.08)
    parser.add_argument("--line-boundary-gap-weight", type=float, default=0.22)
    parser.add_argument("--crossfade-ms", type=float, default=18.0)
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--stub-fit-mode", choices=["crop", "stretch"], default="crop")
    parser.add_argument("--level-mode", choices=["match_rms", "dave_peak"], default="match_rms")
    parser.add_argument("--stub-rms-ratio", type=float, default=1.18)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--min-target-room-ms", type=float, default=45.0)
    parser.add_argument("--ab-gap-sec", type=float, default=1.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dataset_lookup(rows: list[dict[str, str]], speaker: str) -> dict[str, dict[str, str]]:
    return {
        row["word"].lower(): row
        for row in rows
        if row.get("speaker", "").lower() == speaker.lower()
        and row.get("usable", "").lower() == "true"
        and row.get("word", "")
    }


def subtype_lookup(rows: list[dict[str, str]], speaker: str, subtype_col: str) -> dict[str, str]:
    lookup = {}
    for row in rows:
        if row.get("speaker", "").lower() == speaker.lower() and row.get("word") and row.get(subtype_col):
            lookup[row["word"].lower()] = row[subtype_col]
    return lookup


def function_word_lookup(function_root: Path) -> dict[str, Path]:
    lookup = {}
    for path in sorted(function_root.rglob("*.wav")):
        lookup.setdefault(path.parent.name.lower(), path)
    return lookup


def replacement_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        stub_time_scale=args.stub_time_scale,
        stub_time_mode=args.stub_time_mode,
        stub_fit_mode=args.stub_fit_mode,
        level_mode=args.level_mode,
        stub_rms_ratio=args.stub_rms_ratio,
        crossfade_ms=args.crossfade_ms,
    )


def word_duration_weight(
    word: str,
    data_by_word: dict[str, dict[str, str]],
    function_by_word: dict[str, Path],
) -> float:
    path = None
    if word in data_by_word:
        path = Path(data_by_word[word]["audio_path"])
    elif word in function_by_word:
        path = function_by_word[word]
    if path and path.exists():
        sample_rate, audio = read_wav_float(path)
        return max(0.10, min(0.90, len(audio) / sample_rate))
    return max(0.12, min(0.55, 0.06 * len(word)))


def detect_segments(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[list[dict], dict]:
    _, frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(frame_db)
    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        padding_sec=args.padding_sec,
    )
    segments, summary = detect_energy_segments(audio, sample_rate, threshold_dbfs, config=config)
    return segments, {"threshold_info": threshold_info, "summary": summary}


def group_segments_to_lines(segments: list[dict], line_weights: list[float], args: argparse.Namespace) -> list[tuple[int, int]]:
    n = len(segments)
    m = len(line_weights)
    if n < m:
        raise ValueError(f"Need at least {m} speech segments for {m} script lines, got {n}.")

    observed_total = float(segments[-1]["end_sec"] - segments[0]["start_sec"])
    expected_total = float(sum(line_weights))
    expected_durations = [observed_total * weight / expected_total for weight in line_weights]
    gaps = [float(segments[i + 1]["start_sec"] - segments[i]["end_sec"]) for i in range(n - 1)]

    cost = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    back = np.full((m + 1, n + 1), -1, dtype=np.int32)
    cost[0, 0] = 0.0

    for line_idx in range(1, m + 1):
        remaining_lines = m - line_idx
        for end_count in range(line_idx, n - remaining_lines + 1):
            best_score = np.inf
            best_prev = -1
            for prev_count in range(line_idx - 1, end_count):
                if not np.isfinite(cost[line_idx - 1, prev_count]):
                    continue
                start_seg = prev_count
                end_seg = end_count - 1
                group_duration = float(segments[end_seg]["end_sec"] - segments[start_seg]["start_sec"])
                expected = max(0.5, expected_durations[line_idx - 1])
                duration_error = (group_duration - expected) / expected
                score = cost[line_idx - 1, prev_count] + duration_error * duration_error
                if line_idx < m and end_seg < n - 1:
                    score -= args.line_boundary_gap_weight * min(max(gaps[end_seg], 0.0), 2.0) / 2.0
                if score < best_score:
                    best_score = score
                    best_prev = prev_count
            cost[line_idx, end_count] = best_score
            back[line_idx, end_count] = best_prev

    if not np.isfinite(cost[m, n]):
        raise RuntimeError("Failed to align script lines to detected speech segments.")

    groups = []
    end_count = n
    for line_idx in range(m, 0, -1):
        prev_count = int(back[line_idx, end_count])
        groups.append((prev_count, end_count - 1))
        end_count = prev_count
    return list(reversed(groups))


def allocate_word_intervals(
    line_start: float,
    line_end: float,
    words: list[str],
    data_by_word: dict[str, dict[str, str]],
    function_by_word: dict[str, Path],
) -> list[tuple[str, float, float, float]]:
    weights = [word_duration_weight(word, data_by_word, function_by_word) for word in words]
    total_weight = max(1e-6, sum(weights))
    line_duration = max(0.001, line_end - line_start)
    intervals = []
    cursor = line_start
    for word, weight in zip(words, weights):
        duration = line_duration * weight / total_weight
        start = cursor
        end = min(line_end, start + duration)
        intervals.append((word, start, end, weight))
        cursor = end
    if intervals:
        word, start, _, weight = intervals[-1]
        intervals[-1] = (word, start, line_end, weight)
    return intervals


def crossfade_preserve_length(stub: np.ndarray, source_segment: np.ndarray, sample_rate: int, crossfade_ms: float) -> np.ndarray:
    mask_len = len(source_segment)
    if mask_len == 0:
        return source_segment.astype(np.float32)
    if len(stub) >= mask_len:
        return stub[:mask_len].astype(np.float32)

    out = np.zeros(mask_len, dtype=np.float32)
    stub_len = len(stub)
    out[:stub_len] = stub.astype(np.float32)
    tail_start = stub_len
    if tail_start >= mask_len:
        return out

    fade_len = min(round(sample_rate * crossfade_ms / 1000.0), max(1, stub_len // 2), mask_len - tail_start)
    if fade_len > 0:
        fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float32)
        fade_in = 1.0 - fade_out
        overlap_start = max(0, stub_len - fade_len)
        source_overlap = source_segment[overlap_start : overlap_start + fade_len]
        if len(source_overlap) == fade_len:
            out[overlap_start:stub_len] = out[overlap_start:stub_len] * fade_out + source_overlap * fade_in
        tail_start = stub_len
    out[tail_start:] = source_segment[tail_start:]
    return out.astype(np.float32)


def replace_word_onset_in_place(
    enhanced: np.ndarray,
    sample_rate: int,
    word_start_sec: float,
    word_end_sec: float,
    stub: np.ndarray,
    rargs: SimpleNamespace,
    args: argparse.Namespace,
) -> tuple[bool, int, int, int]:
    word_start = max(0, min(len(enhanced) - 1, round(word_start_sec * sample_rate)))
    word_end = max(word_start + 1, min(len(enhanced), round(word_end_sec * sample_rate)))
    target_room = word_end - word_start
    if target_room < round(sample_rate * args.min_target_room_ms / 1000.0):
        return False, word_start, word_start, 0

    local = enhanced[word_start:word_end]
    active_offset = detect_active_start(local, sample_rate)
    start = min(word_end - 1, word_start + active_offset)
    available = word_end - start
    if available < round(sample_rate * args.min_target_room_ms / 1000.0):
        return False, start, start, 0

    stub_len = max(1, len(stub))
    min_mask_len = round(sample_rate * args.mask_min_ms / 1000.0)
    extra_len = round(sample_rate * args.mask_extra_ms / 1000.0)
    max_mask_len = round(sample_rate * args.mask_max_ms / 1000.0)
    mask_len = min(max(stub_len + extra_len, min_mask_len), max_mask_len, available)
    if mask_len <= 0:
        return False, start, start, 0

    fitted_len = min(stub_len, mask_len)
    fitted_stub = fit_stub_to_length(stub, fitted_len, rargs.stub_fit_mode)
    source_segment = enhanced[start : start + mask_len].copy()
    adapted_stub = adapt_stub_level(
        fitted_stub,
        source_segment,
        rargs.level_mode,
        rms_ratio=args.stub_rms_ratio,
        max_stub_peak=0.9,
    )
    replacement = crossfade_preserve_length(adapted_stub, source_segment, sample_rate, args.crossfade_ms)
    enhanced[start : start + mask_len] = replacement
    return True, start, start + mask_len, len(adapted_stub)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample_rate, audio = read_wav_float(args.input)
    data_by_word = dataset_lookup(read_csv(args.dataset_manifest), args.speaker_for_word_durations)
    b_subtypes = subtype_lookup(read_csv(args.b_manifest), args.speaker_for_word_durations, "b_subtype")
    p_subtypes = subtype_lookup(read_csv(args.p_manifest), args.speaker_for_word_durations, "p_subtype")
    function_by_word = function_word_lookup(args.function_root)
    b_stubs = list_stubs(args.b_stub_root)
    p_stubs = list_stubs(args.p_stub_root)
    rargs = replacement_args(args)

    segments, segmentation_info = detect_segments(audio, sample_rate, args)
    if not segments:
        raise ValueError("No speech segments detected.")

    line_weights = [
        sum(word_duration_weight(word, data_by_word, function_by_word) for word in line)
        for line in LONG_STORY_LINES
    ]
    line_groups = group_segments_to_lines(segments, line_weights, args)

    enhanced = audio.astype(np.float32).copy()
    line_rows = []
    word_rows = []
    replacement_count = 0
    skipped_count = 0

    for line_index, ((start_seg_idx, end_seg_idx), words) in enumerate(zip(line_groups, LONG_STORY_LINES), start=1):
        line_start = max(0.0, float(segments[start_seg_idx]["start_sec"]) - args.line_padding_sec)
        line_end = min(len(audio) / sample_rate, float(segments[end_seg_idx]["end_sec"]) + args.line_padding_sec)
        line_rows.append(
            {
                "line_index": str(line_index),
                "segment_start_index": str(start_seg_idx + 1),
                "segment_end_index": str(end_seg_idx + 1),
                "start_sec": f"{line_start:.3f}",
                "end_sec": f"{line_end:.3f}",
                "duration_sec": f"{line_end - line_start:.3f}",
                "script": " ".join(words),
            }
        )

        for word_index, (word, word_start, word_end, weight) in enumerate(
            allocate_word_intervals(line_start, line_end, words, data_by_word, function_by_word),
            start=1,
        ):
            role = "function"
            label = "FUNCTION"
            subtype = ""
            stub_path = ""
            applied = False
            replace_start = replace_end = fitted_len = 0

            if word in b_subtypes:
                role = "content"
                label = "B"
                subtype = b_subtypes[word]
                stubs = b_stubs
            elif word in p_subtypes:
                role = "content"
                label = "P"
                subtype = p_subtypes[word]
                stubs = p_stubs
            else:
                stubs = {}

            if role == "content" and subtype in stubs:
                raw_stub, raw_stub_path = load_stub_for_group(subtype, stubs, sample_rate)
                prepared_stub = prepare_stub_for_insert(raw_stub, rargs)
                applied, replace_start, replace_end, fitted_len = replace_word_onset_in_place(
                    enhanced,
                    sample_rate,
                    word_start,
                    word_end,
                    prepared_stub,
                    rargs,
                    args,
                )
                stub_path = str(raw_stub_path)
                replacement_count += int(applied)
                skipped_count += int(not applied)
            elif role == "content":
                skipped_count += 1

            word_rows.append(
                {
                    "line_index": str(line_index),
                    "word_index": str(word_index),
                    "word": word,
                    "role": role,
                    "label": label,
                    "subtype": subtype,
                    "estimated_word_start_sec": f"{word_start:.3f}",
                    "estimated_word_end_sec": f"{word_end:.3f}",
                    "estimated_word_duration_sec": f"{word_end - word_start:.3f}",
                    "replacement_applied": str(applied).lower(),
                    "replace_start_sec": f"{replace_start / sample_rate:.3f}" if replace_start else "",
                    "replace_end_sec": f"{replace_end / sample_rate:.3f}" if replace_end else "",
                    "replace_duration_sec": f"{(replace_end - replace_start) / sample_rate:.3f}" if replace_end else "",
                    "fitted_stub_duration_sec": f"{fitted_len / sample_rate:.3f}" if fitted_len else "",
                    "stub_path": stub_path,
                    "duration_weight": f"{weight:.4f}",
                }
            )

    peak = float(np.max(np.abs(enhanced))) if len(enhanced) else 0.0
    if peak > 0.98:
        enhanced = (enhanced / peak * 0.98).astype(np.float32)

    original_path = args.outdir / "david_long_story_original.wav"
    enhanced_path = args.outdir / "david_long_story_enhanced_script_guided.wav"
    ab_path = args.outdir / "david_long_story_A_original_B_enhanced_script_guided.wav"
    line_csv = args.outdir / "alignment_lines.csv"
    word_csv = args.outdir / "alignment_words.csv"

    ab_gap = np.zeros(round(sample_rate * args.ab_gap_sec), dtype=np.float32)
    write_wav_float(original_path, sample_rate, audio)
    write_wav_float(enhanced_path, sample_rate, enhanced)
    write_wav_float(ab_path, sample_rate, np.concatenate([audio, ab_gap, enhanced]).astype(np.float32))
    write_csv(line_csv, line_rows)
    write_csv(word_csv, word_rows)

    summary = {
        "input": str(args.input),
        "sample_rate_hz": sample_rate,
        "duration_sec": round(len(audio) / sample_rate, 3),
        "script_line_count": len(LONG_STORY_LINES),
        "detected_segment_count": len(segments),
        "replacement_count": replacement_count,
        "skipped_content_count": skipped_count,
        "original_audio": str(original_path),
        "enhanced_audio": str(enhanced_path),
        "ab_audio": str(ab_path),
        "alignment_lines_csv": str(line_csv),
        "alignment_words_csv": str(word_csv),
        "segmentation": segmentation_info,
        "method_note": (
            "Script-guided prototype: speech segments are grouped into script lines, then word positions are "
            "estimated inside each line by expected word-duration weights. This is not neural forced alignment."
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
