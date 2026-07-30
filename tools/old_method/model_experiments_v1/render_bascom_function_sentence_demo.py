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


SENTENCES = [
    {
        "id": "01",
        "text": "Pat put the big pen in the pink box and Bob put it on the big bag.",
        "words": ["pat", "put", "the", "big", "pen", "in", "the", "pink", "box", "and", "bob", "put", "it", "on", "the", "big", "bag"],
    },
    {
        "id": "02",
        "text": "This box is for Bob and that bag is for Pat.",
        "words": ["this", "box", "is", "for", "bob", "and", "that", "bag", "is", "for", "pat"],
    },
    {
        "id": "03",
        "text": "Bob can put the pink box on the bed and Pat can put the big bag in it.",
        "words": ["bob", "can", "put", "the", "pink", "box", "on", "the", "bed", "and", "pat", "can", "put", "the", "big", "bag", "in", "it"],
    },
    {
        "id": "04",
        "text": "My big pen is with Bob and your pink box is with Pat.",
        "words": ["my", "big", "pen", "is", "with", "bob", "and", "your", "pink", "box", "is", "with", "pat"],
    },
    {
        "id": "05",
        "text": "Pat will pick it up from the box and put it in the bag.",
        "words": ["pat", "will", "pick", "it", "up", "from", "the", "box", "and", "put", "it", "in", "the", "bag"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render coherent sentence demos using Bascom content words plus function words.")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--p-manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--function-root", type=Path, default=PROJECT_ROOT.parent / "word_library" / "function_words")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/bascom_function_sentence_5_demo"))
    parser.add_argument("--speaker", default="bascom")
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.18)
    parser.add_argument("--content-gap-ms", type=float, default=70.0)
    parser.add_argument("--function-gap-ms", type=float, default=45.0)
    parser.add_argument("--sentence-gap-ms", type=float, default=650.0)
    parser.add_argument("--ab-gap-ms", type=float, default=1000.0)
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
        word = path.parent.name.lower()
        lookup.setdefault(word, path)
    return lookup


def sentence_slug(sentence: dict[str, object]) -> str:
    return f"{sentence['id']}_{'_'.join(sentence['words'])}"


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

    all_words = [word for sentence in SENTENCES for word in sentence["words"]]
    missing_content = sorted({w for w in all_words if w not in function_by_word and w not in data_by_word})
    missing_function = sorted({w for w in all_words if w in function_by_word and w not in function_by_word})
    if missing_content or missing_function:
        raise ValueError(f"missing_content={missing_content}; missing_function={missing_function}")

    first_content = next(word for word in all_words if word in data_by_word)
    target_rate, _ = read_wav_float(data_by_word[first_content]["audio_path"])
    content_gap = np.zeros(round(target_rate * args.content_gap_ms / 1000.0), dtype=np.float32)
    function_gap = np.zeros(round(target_rate * args.function_gap_ms / 1000.0), dtype=np.float32)
    sentence_gap = np.zeros(round(target_rate * args.sentence_gap_ms / 1000.0), dtype=np.float32)
    ab_gap = np.zeros(round(target_rate * args.ab_gap_ms / 1000.0), dtype=np.float32)

    all_original_parts = []
    all_enhanced_parts = []
    detail_rows = []
    for sentence_index, sentence in enumerate(SENTENCES, start=1):
        sentence_original_parts = []
        sentence_enhanced_parts = []
        for word_index, word in enumerate(sentence["words"], start=1):
            if word in function_by_word:
                original_word, enhanced_word, detail = render_function_word(word, function_by_word[word], target_rate)
                gap = function_gap
            else:
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
            detail_rows.append(
                {
                    "sentence_id": str(sentence["id"]),
                    "sentence_text": str(sentence["text"]),
                    "sentence_index": str(sentence_index),
                    "word_index": str(word_index),
                    **detail,
                }
            )
            sentence_original_parts.extend([original_word, gap])
            sentence_enhanced_parts.extend([enhanced_word, gap])

        sentence_original = np.concatenate(sentence_original_parts[:-1]).astype(np.float32)
        sentence_enhanced = np.concatenate(sentence_enhanced_parts[:-1]).astype(np.float32)
        sentence_ab = np.concatenate([sentence_original, ab_gap, sentence_enhanced]).astype(np.float32)
        stem = sentence_slug(sentence)
        original_path = args.outdir / f"{stem}_original.wav"
        enhanced_path = args.outdir / f"{stem}_enhanced.wav"
        ab_path = args.outdir / f"{stem}_A_original_B_enhanced.wav"
        write_wav_float(original_path, target_rate, sentence_original)
        write_wav_float(enhanced_path, target_rate, sentence_enhanced)
        write_wav_float(ab_path, target_rate, sentence_ab)
        all_original_parts.extend([sentence_original, sentence_gap])
        all_enhanced_parts.extend([sentence_enhanced, sentence_gap])

    original = np.concatenate(all_original_parts[:-1]).astype(np.float32)
    enhanced = np.concatenate(all_enhanced_parts[:-1]).astype(np.float32)
    ab = np.concatenate([original, ab_gap, enhanced]).astype(np.float32)

    original_path = args.outdir / "all_sentences_original.wav"
    enhanced_path = args.outdir / "all_sentences_enhanced.wav"
    ab_path = args.outdir / "all_sentences_A_original_B_enhanced.wav"
    write_wav_float(original_path, target_rate, original)
    write_wav_float(enhanced_path, target_rate, enhanced)
    write_wav_float(ab_path, target_rate, ab)

    details_path = args.outdir / "word_details.csv"
    write_csv(details_path, detail_rows)
    summary = {
        "rendered_texts": [sentence["text"] for sentence in SENTENCES],
        "target_sample_rate": target_rate,
        "stub_time_scale": args.stub_time_scale,
        "original_audio": str(original_path),
        "enhanced_audio": str(enhanced_path),
        "ab_audio": str(ab_path),
        "details_csv": str(details_path),
        "sentence_count": len(SENTENCES),
        "content_gap_ms": args.content_gap_ms,
        "function_gap_ms": args.function_gap_ms,
        "sentence_gap_ms": args.sentence_gap_ms,
        "enhanced_content_words": sum(1 for row in detail_rows if row["enhancement_applied"] == "true"),
        "function_word_count": sum(1 for word in all_words if word in function_by_word),
        "content_word_count": sum(1 for word in all_words if word in data_by_word),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
