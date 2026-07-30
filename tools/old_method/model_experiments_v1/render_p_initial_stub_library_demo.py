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

from render_b_initial_stub_library_demo import (
    list_stubs,
    load_stub_for_group,
    prepare_stub_for_insert,
    replace_consonant_with_masked_stub,
    tempo_suffix,
    trim_word_boundaries,
    write_sequence_with_tempo,
)
from render_dave_stub_replacement_demo import resample_to


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render P-initial replacement demos using the custom P subtype stub library."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/p_initial_stub_library_bascom"))
    parser.add_argument("--include-speakers", nargs="*", default=["bascom"])
    parser.add_argument("--include-words", nargs="*", default=[], help="Optional lowercase word filter.")
    parser.add_argument(
        "--max-per-subtype",
        type=int,
        default=0,
        help="Set >0 to sample this many clips per P subtype; default keeps all matching clips.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.20)
    parser.add_argument("--sequence-gap-ms", type=float, default=160.0)
    parser.add_argument("--sequence-tempo", type=float, default=1.0)
    parser.add_argument("--sequence-time-mode", choices=["tempo", "speed"], default="tempo")
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


def select_rows(
    rows: list[dict[str, str]],
    stubs: dict[str, list[Path]],
    include_speakers: list[str],
    include_words: list[str],
    max_per_subtype: int,
    seed: int,
) -> list[dict[str, str]]:
    allowed_speakers = {speaker.lower() for speaker in include_speakers}
    allowed_words = {word.lower() for word in include_words}
    rng = np.random.default_rng(seed)
    selected: list[dict[str, str]] = []
    subtypes = sorted(stubs)
    for subtype in subtypes:
        candidates = [
            row
            for row in rows
            if row.get("usable", "").lower() == "true"
            and row.get("p_subtype", "") == subtype
            and row.get("speaker", "").lower() in allowed_speakers
            and (not allowed_words or row.get("word", "").lower() in allowed_words)
        ]
        candidates = sorted(candidates, key=lambda row: (row["speaker"], int(row["protocol_index"]), row["word"]))
        if max_per_subtype > 0 and len(candidates) > max_per_subtype:
            indices = sorted(rng.choice(len(candidates), size=max_per_subtype, replace=False).tolist())
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    return selected


def render_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
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
    stubs = list_stubs(args.stub_root)
    if not stubs:
        raise ValueError(f"No P stub WAV files found under {args.stub_root}")
    rows = select_rows(
        read_csv(args.manifest),
        stubs,
        args.include_speakers,
        args.include_words,
        args.max_per_subtype,
        args.seed,
    )
    if not rows:
        raise ValueError("No usable P rows selected for the available stub groups.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rargs = render_args(args)
    first_rate, _ = read_wav_float(rows[0]["audio_path"])
    silence = np.zeros(round(first_rate * 0.42), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 0.75), dtype=np.float32)
    sequence_silence = np.zeros(round(first_rate * args.sequence_gap_ms / 1000.0), dtype=np.float32)

    detail_rows = []
    combined_parts = []
    original_sequence_parts = []
    enhanced_sequence_parts = []

    for index, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        oracle_subtype = row["p_subtype"]
        stub, stub_path = load_stub_for_group(oracle_subtype, stubs, sample_rate)
        stub = prepare_stub_for_insert(stub, rargs)
        stub_duration_ms = len(stub) / sample_rate * 1000.0
        enhanced, start, end, fitted_len = replace_consonant_with_masked_stub(
            audio.astype(np.float32),
            sample_rate,
            stub,
            stub_duration_ms,
            rargs,
        )

        rendered_original = trim_word_boundaries(audio.astype(np.float32), sample_rate, rargs)
        rendered_enhanced = trim_word_boundaries(enhanced.astype(np.float32), sample_rate, rargs)
        silence_for_rate = silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.42), dtype=np.float32)
        long_silence_for_rate = (
            long_silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.75), dtype=np.float32)
        )
        ab_audio = np.concatenate([rendered_original, silence_for_rate, rendered_enhanced]).astype(np.float32)

        stem = f"{index:03d}_{row['speaker']}_{row['word']}_oracle-{oracle_subtype}"
        original_path = args.outdir / "original" / f"{stem}_original.wav"
        enhanced_path = args.outdir / "enhanced" / f"{stem}_enhanced.wav"
        ab_path = args.outdir / "ab" / f"{stem}_A_original_B_enhanced.wav"
        write_wav_float(original_path, sample_rate, rendered_original)
        write_wav_float(enhanced_path, sample_rate, rendered_enhanced)
        write_wav_float(ab_path, sample_rate, ab_audio)

        combined_parts.extend([ab_audio, long_silence_for_rate])
        original_sequence_parts.extend(
            [resample_to(rendered_original, sample_rate, first_rate).astype(np.float32), sequence_silence]
        )
        enhanced_sequence_parts.extend(
            [resample_to(rendered_enhanced, sample_rate, first_rate).astype(np.float32), sequence_silence]
        )

        detail_rows.append(
            {
                "index": str(index),
                "speaker": row["speaker"],
                "word": row["word"],
                "p_subtype": oracle_subtype,
                "stub_path": str(stub_path),
                "stub_duration_sec": f"{fitted_len / sample_rate:.4f}",
                "mask_start_sec": f"{start / sample_rate:.4f}",
                "mask_end_sec": f"{end / sample_rate:.4f}",
                "source_audio": row["audio_path"],
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "ab_audio": str(ab_path),
                "listener_score_1_to_5": "",
                "listener_preference": "",
                "notes": "",
            }
        )

    combined_path = args.outdir / "p_initial_stub_library_oracle_A_original_B_enhanced.wav"
    write_wav_float(combined_path, first_rate, np.concatenate(combined_parts[:-1]).astype(np.float32))

    suffix = tempo_suffix(args.sequence_tempo, args.sequence_time_mode)
    original_sequence_path = args.outdir / f"p_initial_stub_library_oracle_original_sequence{suffix}.wav"
    enhanced_sequence_path = args.outdir / f"p_initial_stub_library_oracle_enhanced_sequence{suffix}.wav"
    sequence_ab_path = args.outdir / f"p_initial_stub_library_oracle_sequence_A_original_B_enhanced{suffix}.wav"

    original_sequence = np.concatenate(original_sequence_parts[:-1]).astype(np.float32)
    enhanced_sequence = np.concatenate(enhanced_sequence_parts[:-1]).astype(np.float32)
    write_sequence_with_tempo(original_sequence_path, first_rate, original_sequence, args.sequence_tempo, args.sequence_time_mode)
    write_sequence_with_tempo(enhanced_sequence_path, first_rate, enhanced_sequence, args.sequence_tempo, args.sequence_time_mode)
    original_rate, original_sequence = read_wav_float(original_sequence_path)
    enhanced_rate, enhanced_sequence = read_wav_float(enhanced_sequence_path)
    if original_rate != enhanced_rate:
        enhanced_sequence = resample_to(enhanced_sequence, enhanced_rate, original_rate).astype(np.float32)
    write_wav_float(
        sequence_ab_path,
        original_rate,
        np.concatenate(
            [
                original_sequence.astype(np.float32),
                np.zeros(round(original_rate * 1.2), dtype=np.float32),
                enhanced_sequence.astype(np.float32),
            ]
        ),
    )

    details_path = args.outdir / "listening_evaluation_template.csv"
    write_csv(details_path, detail_rows)
    summary = {
        "manifest": str(args.manifest),
        "stub_root": str(args.stub_root),
        "available_stub_groups": sorted(stubs),
        "selected_count": len(detail_rows),
        "include_speakers": args.include_speakers,
        "include_words": args.include_words,
        "stub_time_scale": args.stub_time_scale,
        "stub_time_mode": args.stub_time_mode,
        "replacement_mode": "masked",
        "mask_extra_ms": args.mask_extra_ms,
        "mask_min_ms": args.mask_min_ms,
        "mask_max_ms": args.mask_max_ms,
        "stub_rms_ratio": args.stub_rms_ratio,
        "combined_ab_audio": str(combined_path),
        "original_sequence_audio": str(original_sequence_path),
        "enhanced_sequence_audio": str(enhanced_sequence_path),
        "sequence_ab_audio": str(sequence_ab_path),
        "listening_evaluation_template": str(details_path),
        "rows": detail_rows,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved combined A/B audio: {combined_path}")
    print(f"Saved listening template: {details_path}")


if __name__ == "__main__":
    main()
