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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Bascom B-word sentence demo in exact word order.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--speaker", default="bascom")
    parser.add_argument("--words", nargs="+", default=["big", "black", "boat"])
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=Path("experiments/ml_baseline/b_hubert_large_context_head_bascom_raw_patient/predictions.csv"),
    )
    parser.add_argument("--label-source", choices=["prediction_csv", "oracle"], default="prediction_csv")
    parser.add_argument(
        "--skip-labels",
        nargs="*",
        default=[],
        help="Labels that should not be enhanced, e.g. B_L. The original word audio is kept.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/bascom_sentence_big_black_boat"))
    parser.add_argument("--sequence-gap-ms", type=float, default=170.0)
    parser.add_argument("--stub-time-scale", type=float, default=0.8)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument(
        "--label-max-stub-ms",
        nargs="*",
        default=[],
        help="Optional per-label cap, e.g. B_L=145 B_OW=180. Useful when a stub contains vowel color.",
    )
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=80.0)
    parser.add_argument("--mask-max-ms", type=float, default=320.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.25)
    parser.add_argument("--trim-word-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-word-leading-silence", dest="trim_word_leading_silence", action="store_false")
    parser.add_argument("--trim-word-start-offset-ms", type=float, default=5.0)
    parser.add_argument("--trim-word-tail-ms", type=float, default=28.0)
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


def find_word_rows(manifest_rows: list[dict[str, str]], speaker: str, words: list[str]) -> list[dict[str, str]]:
    by_word = {
        row["word"].lower(): row
        for row in manifest_rows
        if row.get("speaker", "").lower() == speaker.lower() and row.get("usable", "").lower() == "true"
    }
    missing = [word for word in words if word.lower() not in by_word]
    if missing:
        raise ValueError(f"Missing usable words for {speaker}: {missing}")
    return [by_word[word.lower()] for word in words]


def prediction_lookup(path: Path) -> dict[tuple[str, str], tuple[str, float]]:
    if not path.exists():
        return {}
    lookup: dict[tuple[str, str], tuple[str, float]] = {}
    for row in read_csv(path):
        key = (row.get("speaker", "").lower(), row.get("word", "").lower())
        try:
            confidence = float(row.get("confidence", "0.0"))
        except ValueError:
            confidence = 0.0
        lookup[key] = (row.get("predicted_label", ""), confidence)
    return lookup


def parse_label_caps(items: list[str]) -> dict[str, float]:
    caps = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=MS for --label-max-stub-ms, got {item!r}")
        label, value = item.split("=", 1)
        caps[label.upper()] = float(value)
    return caps


def cap_stub_duration(stub: np.ndarray, sample_rate: int, label: str, caps_ms: dict[str, float]) -> np.ndarray:
    max_ms = caps_ms.get(label.upper())
    if not max_ms:
        return stub
    max_len = max(1, round(sample_rate * max_ms / 1000.0))
    return stub[:max_len].astype(np.float32)


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


def main() -> None:
    args = parse_args()
    rows = find_word_rows(read_csv(args.manifest), args.speaker, args.words)
    stubs = list_stubs(args.stub_root)
    predictions = prediction_lookup(args.prediction_csv)
    label_caps_ms = parse_label_caps(args.label_max_stub_ms)
    skip_labels = {label.upper() for label in args.skip_labels}
    rargs = render_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    first_rate, _ = read_wav_float(rows[0]["audio_path"])
    silence = np.zeros(round(first_rate * args.sequence_gap_ms / 1000.0), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 1.0), dtype=np.float32)
    original_parts = []
    enhanced_parts = []
    detail_rows = []

    for index, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        oracle_label = row["b_subtype"]
        key = (row["speaker"].lower(), row["word"].lower())
        predicted_label, confidence = predictions.get(key, (oracle_label, 1.0))
        if args.label_source == "oracle":
            selected_label = oracle_label
            confidence = 1.0
        else:
            selected_label = predicted_label if predicted_label in stubs else oracle_label

        enhancement_applied = selected_label.upper() not in skip_labels
        if enhancement_applied:
            stub, stub_path = load_stub_for_group(selected_label, stubs, sample_rate)
            stub = prepare_stub_for_insert(stub, rargs)
            stub = cap_stub_duration(stub, sample_rate, selected_label, label_caps_ms)
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
        original_word = resample_to(original_word, sample_rate, first_rate).astype(np.float32)
        enhanced_word = resample_to(enhanced_word, sample_rate, first_rate).astype(np.float32)
        original_parts.extend([original_word, silence])
        enhanced_parts.extend([enhanced_word, silence])

        stem = f"{index:02d}_{row['speaker']}_{row['word']}_oracle-{oracle_label}_pred-{predicted_label}_{confidence:.2f}"
        original_path = args.outdir / "words" / f"{stem}_original.wav"
        enhanced_path = args.outdir / "words" / f"{stem}_enhanced.wav"
        write_wav_float(original_path, first_rate, original_word)
        write_wav_float(enhanced_path, first_rate, enhanced_word)
        detail_rows.append(
            {
                "index": str(index),
                "speaker": row["speaker"],
                "word": row["word"],
                "oracle_label": oracle_label,
                "predicted_label": predicted_label,
                "selected_label": selected_label,
                "confidence": f"{confidence:.4f}",
                "prediction_correct": str(predicted_label == oracle_label).lower(),
                "enhancement_applied": str(enhancement_applied).lower(),
                "stub_path": str(stub_path),
                "source_audio": row["audio_path"],
                "original_word_audio": str(original_path),
                "enhanced_word_audio": str(enhanced_path),
                "mask_start_sec": f"{start / sample_rate:.4f}",
                "mask_end_sec": f"{end / sample_rate:.4f}",
                "inserted_stub_sec": f"{fitted_len / sample_rate:.4f}",
            }
        )

    original_sentence = np.concatenate(original_parts[:-1]).astype(np.float32)
    enhanced_sentence = np.concatenate(enhanced_parts[:-1]).astype(np.float32)
    ab_sentence = np.concatenate([original_sentence, long_silence, enhanced_sentence]).astype(np.float32)

    phrase = "_".join(args.words)
    original_sentence_path = args.outdir / f"{args.speaker}_{phrase}_original_sentence.wav"
    enhanced_sentence_path = args.outdir / f"{args.speaker}_{phrase}_enhanced_sentence.wav"
    ab_sentence_path = args.outdir / f"{args.speaker}_{phrase}_A_original_B_enhanced.wav"
    write_wav_float(original_sentence_path, first_rate, original_sentence)
    write_wav_float(enhanced_sentence_path, first_rate, enhanced_sentence)
    write_wav_float(ab_sentence_path, first_rate, ab_sentence)
    details_path = args.outdir / "sentence_render_details.csv"
    write_csv(details_path, detail_rows)

    summary = {
        "speaker": args.speaker,
        "words": args.words,
        "label_source": args.label_source,
        "prediction_csv": str(args.prediction_csv),
        "stub_time_scale": args.stub_time_scale,
        "stub_time_mode": args.stub_time_mode,
        "label_max_stub_ms": label_caps_ms,
        "skip_labels": sorted(skip_labels),
        "sequence_gap_ms": args.sequence_gap_ms,
        "original_sentence_audio": str(original_sentence_path),
        "enhanced_sentence_audio": str(enhanced_sentence_path),
        "ab_sentence_audio": str(ab_sentence_path),
        "details_csv": str(details_path),
        "rows": detail_rows,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
