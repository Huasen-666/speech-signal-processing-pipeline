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

from render_b_initial_stub_library_demo import (
    list_stubs,
    load_stub_for_group,
    prepare_stub_for_insert,
    replace_consonant_with_masked_stub,
    trim_word_boundaries,
)
from render_dave_stub_replacement_demo import resample_to


CONTENT_ONLY_LINES = [
    ["back", "pack", "bad", "pad", "bag", "pan", "ban", "past", "bash", "pass", "bat", "pat", "bath", "path"],
    ["bait", "pace", "base", "page", "bay", "pay", "pain", "pale"],
    ["bed", "peg", "bell", "pen", "bend", "pet", "best", "pick"],
    ["bid", "big", "pig", "bill", "pill", "bin", "pink", "bit", "pit"],
    ["bob", "bond", "box", "boat", "bone", "both", "boy"],
    ["bud", "put", "bug", "puff", "bulk", "pump", "bus", "push", "pull"],
    ["beach", "bead", "beat"],
]


LIGHT_CONNECTOR_LINES = [
    ["back", "pack", "bad", "pad", "bag", "pan", "and", "bait", "pace", "base", "page", "bay", "pay"],
    ["bed", "peg", "bell", "pen", "big", "pick", "bill", "pill", "bit", "pit"],
    ["bob", "in", "the", "box", "boy", "on", "the", "boat", "bud", "bug", "bus", "push"],
    ["pat", "put", "it", "in", "the", "bag", "pink", "pen", "is", "in", "the", "box"],
]


LONG_STORY_LINES = [
    ["bob", "and", "pat", "are", "at", "the", "beach"],
    ["bob", "can", "pack", "the", "big", "bag", "and", "put", "the", "box", "on", "the", "boat"],
    ["pat", "will", "pick", "the", "pink", "pen", "up", "and", "put", "it", "in", "the", "bag"],
    ["the", "boy", "is", "with", "bob", "and", "he", "can", "pass", "the", "pale", "page", "to", "pat"],
    ["this", "big", "box", "is", "for", "bob", "and", "that", "bad", "bag", "is", "for", "pat"],
    ["the", "pet", "is", "on", "the", "bed", "but", "the", "bug", "is", "in", "the", "pan"],
    ["bob", "will", "push", "the", "bus", "back", "and", "pull", "the", "bag", "up"],
    ["pat", "can", "pay", "for", "the", "pass", "and", "put", "the", "bait", "in", "the", "box"],
    ["the", "bell", "is", "on", "the", "boat", "and", "the", "bead", "is", "in", "the", "bin"],
    ["bob", "and", "pat", "will", "pick", "the", "best", "path", "from", "the", "bay", "to", "the", "beach"],
    ["his", "pen", "is", "in", "the", "box", "her", "pad", "is", "in", "the", "bag", "and", "their", "pet", "is", "on", "the", "bed"],
    ["they", "can", "put", "the", "peg", "the", "pill", "and", "the", "pad", "in", "the", "box"],
]


READINGS = {
    "long_story": {
        "description": "A coherent long reading with many B/P content words and natural connector words.",
        "lines": LONG_STORY_LINES,
    },
    "content_only": {
        "description": "Dense B/P content-word reading with no human-recorded connector words.",
        "lines": CONTENT_ONLY_LINES,
    },
    "light_connectors": {
        "description": "Content-heavy reading with only a few connector/function words.",
        "lines": LIGHT_CONNECTOR_LINES,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a long Bascom B/P content-heavy consonant enhancement demo.")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--p-manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--function-root", type=Path, default=PROJECT_ROOT.parent / "word_library" / "function_words")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/bascom_content_heavy_reading_demo"))
    parser.add_argument("--speaker", default="bascom")
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.18)
    parser.add_argument("--content-gap-ms", type=float, default=95.0)
    parser.add_argument("--function-gap-ms", type=float, default=55.0)
    parser.add_argument("--line-gap-ms", type=float, default=480.0)
    parser.add_argument("--reading-gap-ms", type=float, default=1200.0)
    parser.add_argument("--ab-gap-ms", type=float, default=1500.0)
    parser.add_argument("--trim-word-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-word-leading-silence", dest="trim_word_leading_silence", action="store_false")
    parser.add_argument("--trim-word-start-offset-ms", type=float, default=5.0)
    parser.add_argument("--trim-word-tail-ms", type=float, default=24.0)
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


def render_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        level_mode="match_rms",
        stub_fit_mode="crop",
        stub_time_scale=args.stub_time_scale,
        stub_time_mode=args.stub_time_mode,
        crossfade_ms=args.crossfade_ms,
        mask_extra_ms=args.mask_extra_ms,
        mask_min_ms=args.mask_min_ms,
        mask_max_ms=args.mask_max_ms,
        stub_rms_ratio=args.stub_rms_ratio,
        masked_tail_mode="tight_join",
        trim_word_leading_silence=args.trim_word_leading_silence,
        trim_word_start_offset_ms=args.trim_word_start_offset_ms,
        trim_word_tail_ms=args.trim_word_tail_ms,
    )


def render_content_word(
    word: str,
    row: dict[str, str],
    subtype: str,
    stubs: dict[str, list[Path]],
    target_rate: int,
    rargs: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    sample_rate, audio = read_wav_float(row["audio_path"])
    enhancement_applied = subtype in stubs
    if enhancement_applied:
        stub, stub_path = load_stub_for_group(subtype, stubs, sample_rate)
        stub = prepare_stub_for_insert(stub, rargs)
        stub_duration_ms = len(stub) / sample_rate * 1000.0
        enhanced, start, end, fitted_len = replace_consonant_with_masked_stub(
            audio.astype(np.float32),
            sample_rate,
            stub,
            stub_duration_ms,
            rargs,
        )
    else:
        stub_path = Path("")
        enhanced = audio.astype(np.float32)
        start = end = fitted_len = 0

    original_word = trim_word_boundaries(audio.astype(np.float32), sample_rate, rargs)
    enhanced_word = trim_word_boundaries(enhanced.astype(np.float32), sample_rate, rargs)
    original_word = resample_to(original_word, sample_rate, target_rate).astype(np.float32)
    enhanced_word = resample_to(enhanced_word, sample_rate, target_rate).astype(np.float32)
    return original_word, enhanced_word, {
        "word": word,
        "role": "content",
        "label": row["label"].upper(),
        "subtype": subtype,
        "enhancement_applied": str(enhancement_applied).lower(),
        "audio_path": row["audio_path"],
        "stub_path": str(stub_path),
        "stub_duration_sec": f"{fitted_len / sample_rate:.4f}",
        "mask_start_sec": f"{start / sample_rate:.4f}",
        "mask_end_sec": f"{end / sample_rate:.4f}",
    }


def render_function_word(word: str, audio_path: Path, target_rate: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    sample_rate, audio = read_wav_float(audio_path)
    rendered = resample_to(audio.astype(np.float32), sample_rate, target_rate).astype(np.float32)
    return rendered, rendered.copy(), {
        "word": word,
        "role": "function",
        "label": "FUNCTION",
        "subtype": "",
        "enhancement_applied": "false",
        "audio_path": str(audio_path),
        "stub_path": "",
        "stub_duration_sec": "",
        "mask_start_sec": "",
        "mask_end_sec": "",
    }


def render_reading(
    name: str,
    reading: dict[str, object],
    args: argparse.Namespace,
    data_by_word: dict[str, dict[str, str]],
    b_subtypes: dict[str, str],
    p_subtypes: dict[str, str],
    function_by_word: dict[str, Path],
    b_stubs: dict[str, list[Path]],
    p_stubs: dict[str, list[Path]],
    target_rate: int,
    rargs: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    content_gap = np.zeros(round(target_rate * args.content_gap_ms / 1000.0), dtype=np.float32)
    function_gap = np.zeros(round(target_rate * args.function_gap_ms / 1000.0), dtype=np.float32)
    line_gap = np.zeros(round(target_rate * args.line_gap_ms / 1000.0), dtype=np.float32)
    original_parts = []
    enhanced_parts = []
    detail_rows = []

    for line_index, line in enumerate(reading["lines"], start=1):
        line_original_parts = []
        line_enhanced_parts = []
        for word_index, word in enumerate(line, start=1):
            if word in data_by_word:
                row = data_by_word[word]
                if row["label"].upper() == "B":
                    subtype = b_subtypes.get(word, "")
                    stubs = b_stubs
                elif row["label"].upper() == "P":
                    subtype = p_subtypes.get(word, "")
                    stubs = p_stubs
                else:
                    subtype = ""
                    stubs = {}
                original_word, enhanced_word, detail = render_content_word(word, row, subtype, stubs, target_rate, rargs)
                gap = content_gap
            elif word in function_by_word:
                original_word, enhanced_word, detail = render_function_word(word, function_by_word[word], target_rate)
                gap = function_gap
            else:
                raise ValueError(f"Unknown word in {name}: {word}")

            detail_rows.append(
                {
                    "reading": name,
                    "line_index": str(line_index),
                    "word_index": str(word_index),
                    **detail,
                }
            )
            line_original_parts.extend([original_word, gap])
            line_enhanced_parts.extend([enhanced_word, gap])

        original_parts.extend([np.concatenate(line_original_parts[:-1]).astype(np.float32), line_gap])
        enhanced_parts.extend([np.concatenate(line_enhanced_parts[:-1]).astype(np.float32), line_gap])

    return (
        np.concatenate(original_parts[:-1]).astype(np.float32),
        np.concatenate(enhanced_parts[:-1]).astype(np.float32),
        detail_rows,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data_by_word = dataset_lookup(read_csv(args.dataset_manifest), args.speaker)
    b_subtypes = subtype_lookup(read_csv(args.b_manifest), args.speaker, "b_subtype")
    p_subtypes = subtype_lookup(read_csv(args.p_manifest), args.speaker, "p_subtype")
    function_by_word = function_word_lookup(args.function_root)
    b_stubs = list_stubs(args.b_stub_root)
    p_stubs = list_stubs(args.p_stub_root)
    rargs = render_args(args)

    all_script_words = [word for reading in READINGS.values() for line in reading["lines"] for word in line]
    missing = sorted({word for word in all_script_words if word not in data_by_word and word not in function_by_word})
    if missing:
        raise ValueError(f"Missing script words: {missing}")

    first_content = next(word for word in all_script_words if word in data_by_word)
    target_rate, _ = read_wav_float(data_by_word[first_content]["audio_path"])
    ab_gap = np.zeros(round(target_rate * args.ab_gap_ms / 1000.0), dtype=np.float32)
    reading_gap = np.zeros(round(target_rate * args.reading_gap_ms / 1000.0), dtype=np.float32)

    all_original_parts = []
    all_enhanced_parts = []
    all_detail_rows = []
    reading_summaries = {}

    for name, reading in READINGS.items():
        original, enhanced, details = render_reading(
            name,
            reading,
            args,
            data_by_word,
            b_subtypes,
            p_subtypes,
            function_by_word,
            b_stubs,
            p_stubs,
            target_rate,
            rargs,
        )
        ab = np.concatenate([original, ab_gap, enhanced]).astype(np.float32)
        write_wav_float(args.outdir / f"{name}_original.wav", target_rate, original)
        write_wav_float(args.outdir / f"{name}_enhanced.wav", target_rate, enhanced)
        write_wav_float(args.outdir / f"{name}_A_original_B_enhanced.wav", target_rate, ab)
        all_original_parts.extend([original, reading_gap])
        all_enhanced_parts.extend([enhanced, reading_gap])
        all_detail_rows.extend(details)
        reading_summaries[name] = {
            "description": reading["description"],
            "line_count": len(reading["lines"]),
            "word_count": len([word for line in reading["lines"] for word in line]),
            "content_word_count": sum(1 for row in details if row["role"] == "content"),
            "function_word_count": sum(1 for row in details if row["role"] == "function"),
            "enhanced_content_words": sum(1 for row in details if row["role"] == "content" and row["enhancement_applied"] == "true"),
        }

    all_original = np.concatenate(all_original_parts[:-1]).astype(np.float32)
    all_enhanced = np.concatenate(all_enhanced_parts[:-1]).astype(np.float32)
    all_ab = np.concatenate([all_original, ab_gap, all_enhanced]).astype(np.float32)
    write_wav_float(args.outdir / "all_readings_original.wav", target_rate, all_original)
    write_wav_float(args.outdir / "all_readings_enhanced.wav", target_rate, all_enhanced)
    write_wav_float(args.outdir / "all_readings_A_original_B_enhanced.wav", target_rate, all_ab)

    details_path = args.outdir / "word_details.csv"
    write_csv(details_path, all_detail_rows)
    summary = {
        "target_sample_rate": target_rate,
        "stub_time_scale": args.stub_time_scale,
        "stub_time_mode": args.stub_time_mode,
        "content_gap_ms": args.content_gap_ms,
        "function_gap_ms": args.function_gap_ms,
        "line_gap_ms": args.line_gap_ms,
        "readings": reading_summaries,
        "details_csv": str(details_path),
        "all_original_audio": str(args.outdir / "all_readings_original.wav"),
        "all_enhanced_audio": str(args.outdir / "all_readings_enhanced.wav"),
        "all_ab_audio": str(args.outdir / "all_readings_A_original_B_enhanced.wav"),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
