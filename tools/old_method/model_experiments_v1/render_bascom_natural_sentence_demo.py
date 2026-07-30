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
        "id": "pat_put_big_bag_back",
        "requested_text": "Pat put the big bag back.",
        "rendered_text": "Pat put big bag back.",
        "words": ["pat", "put", "big", "bag", "back"],
        "notes": "Skipped 'the' because no Bascom patient clip exists.",
    },
    {
        "id": "bob_pack_pink_box",
        "requested_text": "Ben packed the pink box.",
        "rendered_text": "Bob pack pink box.",
        "words": ["bob", "pack", "pink", "box"],
        "notes": "Used available Bascom words: Bob for Ben, pack for packed, skipped 'the'.",
    },
    {
        "id": "bob_pick_big_pen",
        "requested_text": "Bob picked up the big pen.",
        "rendered_text": "Bob pick big pen.",
        "words": ["bob", "pick", "big", "pen"],
        "notes": "Used pick for picked; skipped 'up' and 'the' because no Bascom patient clips exist.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render understandable Bascom sentence demos with B/P onset enhancement.")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--p-manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/bascom_natural_sentence_demo_speed100"))
    parser.add_argument("--speaker", default="bascom")
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.18)
    parser.add_argument("--word-gap-ms", type=float, default=95.0)
    parser.add_argument("--sentence-gap-ms", type=float, default=800.0)
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
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if row.get("speaker", "").lower() != speaker.lower():
            continue
        if row.get("word", "") and row.get(subtype_col, ""):
            lookup[row["word"].lower()] = row[subtype_col]
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


def render_word(
    word: str,
    row: dict[str, str],
    subtype: str,
    stubs: dict[str, list[Path]],
    sample_rate_target: int,
    args: argparse.Namespace,
    rargs: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    sample_rate, audio = read_wav_float(row["audio_path"])
    label = row["label"].upper()
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
    original_word = resample_to(original_word, sample_rate, sample_rate_target).astype(np.float32)
    enhanced_word = resample_to(enhanced_word, sample_rate, sample_rate_target).astype(np.float32)
    detail = {
        "word": word,
        "label": label,
        "subtype": subtype,
        "enhancement_applied": str(enhancement_applied).lower(),
        "stub_path": str(stub_path),
        "stub_duration_sec": f"{fitted_len / sample_rate:.4f}",
        "mask_start_sec": f"{start / sample_rate:.4f}",
        "mask_end_sec": f"{end / sample_rate:.4f}",
        "source_audio": row["audio_path"],
    }
    return original_word, enhanced_word, detail


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    data_by_word = dataset_lookup(read_csv(args.dataset_manifest), args.speaker)
    b_subtypes = subtype_lookup(read_csv(args.b_manifest), args.speaker, "b_subtype")
    p_subtypes = subtype_lookup(read_csv(args.p_manifest), args.speaker, "p_subtype")
    b_stubs = list_stubs(args.b_stub_root)
    p_stubs = list_stubs(args.p_stub_root)
    rargs = render_args(args)

    all_words = [word for sentence in SENTENCES for word in sentence["words"]]
    missing = sorted({word for word in all_words if word not in data_by_word})
    if missing:
        raise ValueError(f"Missing Bascom clips for required rendered words: {missing}")

    first_rate, _ = read_wav_float(data_by_word[SENTENCES[0]["words"][0]]["audio_path"])
    word_gap = np.zeros(round(first_rate * args.word_gap_ms / 1000.0), dtype=np.float32)
    sentence_gap = np.zeros(round(first_rate * args.sentence_gap_ms / 1000.0), dtype=np.float32)
    ab_gap = np.zeros(round(first_rate * args.ab_gap_ms / 1000.0), dtype=np.float32)

    summary_sentences = []
    detail_rows = []
    combined_original_parts = []
    combined_enhanced_parts = []

    for sentence_index, sentence in enumerate(SENTENCES, start=1):
        original_parts = []
        enhanced_parts = []
        sentence_rows = []
        for word_index, word in enumerate(sentence["words"], start=1):
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

            original_word, enhanced_word, detail = render_word(word, row, subtype, stubs, first_rate, args, rargs)
            original_parts.extend([original_word, word_gap])
            enhanced_parts.extend([enhanced_word, word_gap])
            sentence_rows.append(
                {
                    "sentence_id": sentence["id"],
                    "sentence_index": str(sentence_index),
                    "word_index": str(word_index),
                    **detail,
                }
            )

        original_sentence = np.concatenate(original_parts[:-1]).astype(np.float32)
        enhanced_sentence = np.concatenate(enhanced_parts[:-1]).astype(np.float32)
        ab_sentence = np.concatenate([original_sentence, ab_gap, enhanced_sentence]).astype(np.float32)

        original_path = args.outdir / f"{sentence_index:02d}_{sentence['id']}_original.wav"
        enhanced_path = args.outdir / f"{sentence_index:02d}_{sentence['id']}_enhanced.wav"
        ab_path = args.outdir / f"{sentence_index:02d}_{sentence['id']}_A_original_B_enhanced.wav"
        write_wav_float(original_path, first_rate, original_sentence)
        write_wav_float(enhanced_path, first_rate, enhanced_sentence)
        write_wav_float(ab_path, first_rate, ab_sentence)

        detail_rows.extend(sentence_rows)
        combined_original_parts.extend([original_sentence, sentence_gap])
        combined_enhanced_parts.extend([enhanced_sentence, sentence_gap])
        summary_sentences.append(
            {
                **sentence,
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "ab_audio": str(ab_path),
            }
        )

    combined_original = np.concatenate(combined_original_parts[:-1]).astype(np.float32)
    combined_enhanced = np.concatenate(combined_enhanced_parts[:-1]).astype(np.float32)
    combined_ab = np.concatenate([combined_original, ab_gap, combined_enhanced]).astype(np.float32)
    combined_original_path = args.outdir / "all_sentences_original.wav"
    combined_enhanced_path = args.outdir / "all_sentences_enhanced.wav"
    combined_ab_path = args.outdir / "all_sentences_A_original_B_enhanced.wav"
    write_wav_float(combined_original_path, first_rate, combined_original)
    write_wav_float(combined_enhanced_path, first_rate, combined_enhanced)
    write_wav_float(combined_ab_path, first_rate, combined_ab)

    details_path = args.outdir / "sentence_word_details.csv"
    write_csv(details_path, detail_rows)
    summary = {
        "speaker": args.speaker,
        "stub_time_scale": args.stub_time_scale,
        "stub_time_mode": args.stub_time_mode,
        "word_gap_ms": args.word_gap_ms,
        "sentence_gap_ms": args.sentence_gap_ms,
        "available_b_stub_groups": sorted(b_stubs),
        "available_p_stub_groups": sorted(p_stubs),
        "sentences": summary_sentences,
        "combined_original_audio": str(combined_original_path),
        "combined_enhanced_audio": str(combined_enhanced_path),
        "combined_ab_audio": str(combined_ab_path),
        "details_csv": str(details_path),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
