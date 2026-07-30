from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.normalization import amplitude_to_dbfs, dbfs_to_amplitude, peak, rms

from render_dave_stub_replacement_demo import resample_to


PARAGRAPH_LINES = [
    ["bob", "and", "pat", "are", "at", "the", "beach"],
    ["bob", "can", "pack", "the", "big", "bag", "and", "put", "the", "box", "on", "the", "boat"],
    ["pat", "will", "pick", "the", "pink", "pen", "up", "and", "put", "it", "in", "the", "bag"],
    ["the", "boy", "is", "with", "bob", "and", "he", "can", "pass", "the", "pale", "page", "to", "pat"],
    ["this", "big", "box", "is", "for", "bob", "and", "that", "bad", "bag", "is", "for", "pat"],
    ["bob", "will", "push", "the", "bus", "back", "and", "pull", "the", "bag", "up"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch model-driven David B/P enhanced word clips into a paragraph.")
    parser.add_argument(
        "--replacement-manifest",
        type=Path,
        default=Path("experiments/phone_prototype/david_bp_model_driven_replacement_gate080/model_driven_replacement_manifest.csv"),
    )
    parser.add_argument("--function-root", type=Path, default=PROJECT_ROOT.parent / "word_library" / "function_words")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/david_model_driven_paragraph_gate080"))
    parser.add_argument("--word-gap-ms", type=float, default=55.0)
    parser.add_argument("--line-gap-ms", type=float, default=430.0)
    parser.add_argument("--ab-gap-ms", type=float, default=1400.0)
    parser.add_argument("--content-target-dbfs", type=float, default=-18.0)
    parser.add_argument("--function-target-dbfs", type=float, default=-23.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def function_word_lookup(function_root: Path) -> dict[str, Path]:
    lookup = {}
    for path in sorted(function_root.rglob("*.wav")):
        lookup.setdefault(path.parent.name.lower(), path)
    return lookup


def choose_best_word_clips(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        candidates.setdefault(row["word"].lower(), []).append(row)

    best = {}
    for word, word_rows in candidates.items():
        def score(row: dict[str, str]) -> tuple[int, int, float, int]:
            correct = 1 if row.get("bp_prediction_correct", "").lower() == "true" else 0
            replaced = 1 if row.get("replacement_applied", "").lower() == "true" else 0
            confidence = float(row.get("confidence") or 0.0)
            take = int(row.get("take") or 0)
            return correct, replaced, confidence, take

        best[word] = sorted(word_rows, key=score, reverse=True)[0]
    return best


def load_clip(path: Path, target_rate: int | None = None) -> tuple[int, np.ndarray]:
    sample_rate, audio = read_wav_float(path)
    if target_rate is not None and sample_rate != target_rate:
        audio = resample_to(audio.astype(np.float32), sample_rate, target_rate).astype(np.float32)
        sample_rate = target_rate
    return sample_rate, audio.astype(np.float32)


def peak_protect_audio(audio: np.ndarray, max_peak_dbfs: float) -> np.ndarray:
    max_peak = dbfs_to_amplitude(max_peak_dbfs)
    current_peak = peak(audio)
    if current_peak <= max_peak:
        return audio.astype(np.float32)
    return (audio * (max_peak / max(current_peak, 1e-12))).astype(np.float32)


def normalize_pair_to_original_rms(
    original: np.ndarray,
    enhanced: np.ndarray,
    target_dbfs: float,
    max_peak_dbfs: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    source_rms = rms(original)
    target_rms = dbfs_to_amplitude(target_dbfs)
    gain = target_rms / max(source_rms, 1e-12)
    original_out = peak_protect_audio(original * gain, max_peak_dbfs)
    enhanced_out = peak_protect_audio(enhanced * gain, max_peak_dbfs)
    return original_out, enhanced_out, {
        "input_rms_dbfs": f"{amplitude_to_dbfs(source_rms):.2f}",
        "target_rms_dbfs": f"{target_dbfs:.2f}",
        "gain_db": f"{amplitude_to_dbfs(gain):.2f}",
        "output_original_peak_dbfs": f"{amplitude_to_dbfs(peak(original_out)):.2f}",
        "output_enhanced_peak_dbfs": f"{amplitude_to_dbfs(peak(enhanced_out)):.2f}",
    }


def normalize_single(
    audio: np.ndarray,
    target_dbfs: float,
    max_peak_dbfs: float,
) -> tuple[np.ndarray, dict[str, str]]:
    source_rms = rms(audio)
    target_rms = dbfs_to_amplitude(target_dbfs)
    gain = target_rms / max(source_rms, 1e-12)
    out = peak_protect_audio(audio * gain, max_peak_dbfs)
    return out, {
        "input_rms_dbfs": f"{amplitude_to_dbfs(source_rms):.2f}",
        "target_rms_dbfs": f"{target_dbfs:.2f}",
        "gain_db": f"{amplitude_to_dbfs(gain):.2f}",
        "output_peak_dbfs": f"{amplitude_to_dbfs(peak(out)):.2f}",
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.replacement_manifest)
    clips_by_word = choose_best_word_clips(rows)
    function_by_word = function_word_lookup(args.function_root)
    args.outdir.mkdir(parents=True, exist_ok=True)

    first_word = PARAGRAPH_LINES[0][0]
    first_path = resolve_repo_path(clips_by_word[first_word]["enhanced_audio"])
    target_rate, _ = read_wav_float(first_path)
    word_gap = np.zeros(round(target_rate * args.word_gap_ms / 1000.0), dtype=np.float32)
    line_gap = np.zeros(round(target_rate * args.line_gap_ms / 1000.0), dtype=np.float32)
    ab_gap = np.zeros(round(target_rate * args.ab_gap_ms / 1000.0), dtype=np.float32)

    original_parts = []
    enhanced_parts = []
    detail_rows = []

    for line_index, words in enumerate(PARAGRAPH_LINES, start=1):
        line_original = []
        line_enhanced = []
        for word_index, word in enumerate(words, start=1):
            if word in clips_by_word:
                row = clips_by_word[word]
                _, original = load_clip(resolve_repo_path(row["original_audio"]), target_rate)
                _, enhanced = load_clip(resolve_repo_path(row["enhanced_audio"]), target_rate)
                original, enhanced, level_info = normalize_pair_to_original_rms(
                    original,
                    enhanced,
                    args.content_target_dbfs,
                    args.max_peak_dbfs,
                )
                detail = {
                    "line_index": str(line_index),
                    "word_index": str(word_index),
                    "word": word,
                    "role": "content",
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": row["confidence"],
                    "bp_prediction_correct": row["bp_prediction_correct"],
                    "replacement_applied": row["replacement_applied"],
                    "true_subtype": row["true_subtype"],
                    "predicted_subtype_for_replacement": row["predicted_subtype_for_replacement"],
                    "audio_used": row["enhanced_audio"],
                    **level_info,
                }
            elif word in function_by_word:
                _, rendered = load_clip(function_by_word[word], target_rate)
                rendered, level_info = normalize_single(rendered, args.function_target_dbfs, args.max_peak_dbfs)
                original = rendered
                enhanced = rendered.copy()
                detail = {
                    "line_index": str(line_index),
                    "word_index": str(word_index),
                    "word": word,
                    "role": "function",
                    "true_label": "",
                    "predicted_label": "",
                    "confidence": "",
                    "bp_prediction_correct": "",
                    "replacement_applied": "false",
                    "true_subtype": "",
                    "predicted_subtype_for_replacement": "",
                    "audio_used": str(function_by_word[word]),
                    **level_info,
                }
            else:
                raise ValueError(f"No audio available for word: {word}")

            line_original.extend([original, word_gap])
            line_enhanced.extend([enhanced, word_gap])
            detail_rows.append(detail)

        original_parts.extend([np.concatenate(line_original[:-1]).astype(np.float32), line_gap])
        enhanced_parts.extend([np.concatenate(line_enhanced[:-1]).astype(np.float32), line_gap])

    original_audio = np.concatenate(original_parts[:-1]).astype(np.float32)
    enhanced_audio = np.concatenate(enhanced_parts[:-1]).astype(np.float32)
    ab_audio = np.concatenate([original_audio, ab_gap, enhanced_audio]).astype(np.float32)

    original_path = args.outdir / "david_model_driven_paragraph_original.wav"
    enhanced_path = args.outdir / "david_model_driven_paragraph_enhanced.wav"
    ab_path = args.outdir / "david_model_driven_paragraph_A_original_B_enhanced.wav"
    details_path = args.outdir / "paragraph_word_details.csv"
    summary_path = args.outdir / "summary.json"

    write_wav_float(original_path, target_rate, original_audio)
    write_wav_float(enhanced_path, target_rate, enhanced_audio)
    write_wav_float(ab_path, target_rate, ab_audio)
    write_csv(details_path, detail_rows)

    content_rows = [row for row in detail_rows if row["role"] == "content"]
    replaced = sum(row["replacement_applied"] == "true" for row in content_rows)
    correct = sum(row["bp_prediction_correct"] == "true" for row in content_rows)
    paragraph_text = " ".join(" ".join(words) for words in PARAGRAPH_LINES)
    summary = {
        "paragraph_text": paragraph_text,
        "replacement_manifest": str(args.replacement_manifest),
        "content_word_instances": len(content_rows),
        "content_word_unique": len({row["word"] for row in content_rows}),
        "selected_content_bp_accuracy": round(correct / len(content_rows), 4) if content_rows else None,
        "selected_content_replacement_count": replaced,
        "word_gap_ms": args.word_gap_ms,
        "line_gap_ms": args.line_gap_ms,
        "content_target_dbfs": args.content_target_dbfs,
        "function_target_dbfs": args.function_target_dbfs,
        "max_peak_dbfs": args.max_peak_dbfs,
        "original_rms_dbfs": round(amplitude_to_dbfs(rms(original_audio)), 3),
        "enhanced_rms_dbfs": round(amplitude_to_dbfs(rms(enhanced_audio)), 3),
        "original_peak_dbfs": round(amplitude_to_dbfs(peak(original_audio)), 3),
        "enhanced_peak_dbfs": round(amplitude_to_dbfs(peak(enhanced_audio)), 3),
        "original_audio": str(original_path),
        "enhanced_audio": str(enhanced_path),
        "ab_audio": str(ab_path),
        "details_csv": str(details_path),
        "selection_note": "For each content word, the script selected the best available model-driven clip by correctness, replacement status, confidence, then take.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
