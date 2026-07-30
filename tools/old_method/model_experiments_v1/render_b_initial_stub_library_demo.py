import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float

from render_dave_stub_replacement_demo import (
    adapt_stub_level,
    crossfade_join,
    load_stub,
    load_tinycnn_model,
    predict_tinycnn_label,
    resample_to,
)
from run_phone_consonant_enhancement_prototype import (
    detect_active_start,
    fit_stub_to_length,
    replace_consonant_with_duration_stub,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render B-initial replacement demos using the custom B stub library.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/b_initial_stub_library"))
    parser.add_argument("--label-source", choices=["oracle", "model"], default="oracle")
    parser.add_argument("--model-dir", type=Path, default=Path("experiments/ml_baseline/b_subtype_tinycnn_v1"))
    parser.add_argument("--include-speakers", nargs="*", default=["bascom", "corrick", "mickey"])
    parser.add_argument("--include-words", nargs="*", default=[], help="Optional lowercase word filter, e.g. big boat.")
    parser.add_argument("--max-per-subtype", type=int, default=4, help="Set <=0 to keep all rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level-mode", choices=["match_rms", "dave_peak"], default="match_rms")
    parser.add_argument("--stub-fit-mode", choices=["crop", "stretch"], default="crop")
    parser.add_argument(
        "--stub-time-scale",
        type=float,
        default=1.0,
        help="Local time scale for the inserted initial consonant stub. Use 0.8 to slow only the stub.",
    )
    parser.add_argument(
        "--stub-time-mode",
        choices=["speed", "tempo"],
        default="speed",
        help=(
            "speed changes stub duration and pitch, like Audacity playback speed; "
            "tempo changes stub duration while trying to preserve pitch."
        ),
    )
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--skip-labels",
        nargs="*",
        default=[],
        help="Predicted/oracle labels that should not be enhanced, e.g. B_L.",
    )
    parser.add_argument(
        "--replacement-mode",
        choices=["direct", "masked"],
        default="direct",
        help="direct replaces only the stub-length region; masked mutes a wider onset region before inserting the stub.",
    )
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=80.0)
    parser.add_argument("--mask-max-ms", type=float, default=320.0)
    parser.add_argument(
        "--stub-rms-ratio",
        type=float,
        default=1.25,
        help="Only used by --replacement-mode masked with --level-mode match_rms.",
    )
    parser.add_argument(
        "--masked-tail-mode",
        choices=["tight_join", "zero_fill"],
        default="tight_join",
        help=(
            "tight_join removes the masked patient onset and crossfades the stub directly into the remaining vowel; "
            "zero_fill preserves original timing by leaving silence after a short stub."
        ),
    )
    parser.add_argument(
        "--sequence-gap-ms",
        type=float,
        default=160.0,
        help="Short gap between words in continuous original/enhanced sequence outputs.",
    )
    parser.add_argument(
        "--sequence-tempo",
        type=float,
        default=1.0,
        help="Tempo factor for sequence outputs. Use 0.85 to slow speech down while preserving pitch.",
    )
    parser.add_argument(
        "--sequence-time-mode",
        choices=["tempo", "speed"],
        default="tempo",
        help=(
            "tempo changes duration while preserving pitch, similar to Audacity Change Tempo; "
            "speed changes both duration and pitch, similar to Audacity Change Speed."
        ),
    )
    parser.add_argument(
        "--write-individual-time-versions",
        action="store_true",
        help="Also write per-word original/enhanced/A-B files with --sequence-tempo applied.",
    )
    parser.add_argument(
        "--trim-word-leading-silence",
        action="store_true",
        help="Trim leading low-energy samples from each word before sequence/time rendering.",
    )
    parser.add_argument(
        "--trim-word-tail-ms",
        type=float,
        default=0.0,
        help="Conservatively trim this many milliseconds from each word tail before sequence/time rendering.",
    )
    parser.add_argument(
        "--trim-word-start-offset-ms",
        type=float,
        default=0.0,
        help="Move the detected trim start later by this many milliseconds.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_stubs(stub_root: Path) -> dict[str, list[Path]]:
    stubs: dict[str, list[Path]] = {}
    for group_dir in sorted(stub_root.iterdir() if stub_root.exists() else []):
        if not group_dir.is_dir():
            continue
        files = sorted(group_dir.glob("*.wav"))
        if files:
            stubs[group_dir.name] = files
    return stubs


def select_rows(
    rows: list[dict[str, str]],
    include_speakers: list[str],
    include_words: list[str],
    max_per_subtype: int,
    seed: int,
) -> list[dict[str, str]]:
    allowed_speakers = {speaker.lower() for speaker in include_speakers}
    allowed_words = {word.lower() for word in include_words}
    rng = np.random.default_rng(seed)
    selected = []
    subtypes = sorted({row["b_subtype"] for row in rows if row["usable"].lower() == "true"})
    for subtype in subtypes:
        candidates = [
            row
            for row in rows
            if row["usable"].lower() == "true"
            and row["b_subtype"] == subtype
            and row["speaker"].lower() in allowed_speakers
            and (not allowed_words or row["word"].lower() in allowed_words)
        ]
        candidates = sorted(candidates, key=lambda row: (row["speaker"], row["protocol_index"], row["word"]))
        if max_per_subtype > 0 and len(candidates) > max_per_subtype:
            indices = sorted(rng.choice(len(candidates), size=max_per_subtype, replace=False).tolist())
            candidates = [candidates[idx] for idx in indices]
        selected.extend(candidates)
    return selected


def load_stub_for_group(
    group: str,
    stubs: dict[str, list[Path]],
    sample_rate: int,
) -> tuple[np.ndarray, Path]:
    if group not in stubs:
        raise KeyError(f"No stub available for {group}")
    path = stubs[group][0]
    return load_stub(path, sample_rate, target_peak=0.5), path


def time_scale_signal(audio: np.ndarray, scale: float, mode: str) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError("--stub-time-scale must be positive.")
    if abs(scale - 1.0) < 1e-6 or len(audio) <= 1:
        return audio.astype(np.float32)

    target_len = max(1, round(len(audio) / scale))
    if mode == "tempo":
        import librosa

        return librosa.effects.time_stretch(audio.astype(np.float32), rate=scale).astype(np.float32)

    source_positions = np.arange(target_len, dtype=np.float64) * scale
    source_positions = np.clip(source_positions, 0.0, len(audio) - 1.0)
    samples = np.interp(source_positions, np.arange(len(audio), dtype=np.float64), audio.astype(np.float64))
    return samples.astype(np.float32)


def prepare_stub_for_insert(stub: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return time_scale_signal(stub, args.stub_time_scale, args.stub_time_mode)


def replace_consonant_with_masked_stub(
    audio: np.ndarray,
    sample_rate: int,
    stub: np.ndarray,
    duration_ms: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int]:
    start = detect_active_start(audio, sample_rate)
    stub_len = max(1, round(sample_rate * duration_ms / 1000.0))
    min_mask_len = round(sample_rate * args.mask_min_ms / 1000.0)
    extra_len = round(sample_rate * args.mask_extra_ms / 1000.0)
    max_mask_len = round(sample_rate * args.mask_max_ms / 1000.0)
    mask_len = min(max(stub_len + extra_len, min_mask_len), max_mask_len)
    end = min(len(audio), start + mask_len)
    mask_len = max(1, end - start)
    stub_len = min(stub_len, mask_len)

    fitted_stub = fit_stub_to_length(stub, stub_len, args.stub_fit_mode)
    source_segment = audio[start:end]
    adapted_stub = adapt_stub_level(
        fitted_stub,
        source_segment,
        args.level_mode,
        rms_ratio=args.stub_rms_ratio,
        max_stub_peak=0.9,
    )

    before = audio[:start].astype(np.float32)
    tail = audio[end:].astype(np.float32)
    if args.masked_tail_mode == "zero_fill":
        masked_body = np.zeros(mask_len, dtype=np.float32)
        masked_body[: len(adapted_stub)] = adapted_stub
        fade_len = min(round(sample_rate * args.crossfade_ms / 1000.0), max(1, mask_len // 2))
        replaced_body = crossfade_join(masked_body, tail, fade_len)
    else:
        fade_len = min(round(sample_rate * args.crossfade_ms / 1000.0), max(1, len(adapted_stub) // 2))
        replaced_body = crossfade_join(adapted_stub, tail, fade_len)
    replaced = np.concatenate([before, replaced_body]).astype(np.float32)

    peak = float(np.max(np.abs(replaced))) if len(replaced) else 0.0
    if peak > 0.98:
        replaced = (replaced / peak * 0.98).astype(np.float32)
    return replaced, start, end, len(adapted_stub)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "speaker",
        "word",
        "oracle_subtype",
        "predicted_subtype",
        "confidence",
        "prediction_correct",
        "enhancement_applied",
        "stub_path",
        "stub_duration_sec",
        "source_audio",
        "original_audio",
        "enhanced_audio",
        "ab_audio",
        "original_time_audio",
        "enhanced_time_audio",
        "ab_time_audio",
        "listener_score_1_to_5",
        "listener_preference",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tempo_suffix(tempo: float, mode: str) -> str:
    if abs(tempo - 1.0) < 1e-6:
        return ""
    return f"_{mode}{tempo:.2f}".replace(".", "")


def write_sequence_with_tempo(path: Path, sample_rate: int, audio: np.ndarray, tempo: float, mode: str) -> None:
    if tempo <= 0.0:
        raise ValueError("--sequence-tempo must be positive.")
    if abs(tempo - 1.0) < 1e-6:
        write_wav_float(path, sample_rate, audio)
        return

    temp_path = path.with_name(path.stem + "_unstretched_tmp.wav")
    write_wav_float(temp_path, sample_rate, audio)
    try:
        if mode == "tempo":
            audio_filter = f"atempo={tempo}"
        else:
            slowed_rate = max(1, round(sample_rate * tempo))
            audio_filter = f"asetrate={slowed_rate},aresample={sample_rate}"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_path), "-filter:a", audio_filter, str(path)],
            check=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def detect_trim_start(audio: np.ndarray, sample_rate: int) -> int:
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return 0

    starts = []
    rms_values = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        starts.append(start)
        rms_values.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))

    if not rms_values or max(rms_values) <= 1e-10:
        return 0

    frame_db = 20.0 * np.log10(np.maximum(rms_values, 1e-12))
    noise_floor = float(np.percentile(frame_db, 10))
    high_energy = float(np.percentile(frame_db, 95))
    threshold = max(noise_floor + 15.0, high_energy - 30.0, -55.0)
    active = np.flatnonzero(frame_db >= threshold)
    return int(starts[active[0]]) if len(active) else 0


def trim_word_boundaries(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> np.ndarray:
    start = detect_trim_start(audio, sample_rate) if args.trim_word_leading_silence else 0
    start += round(sample_rate * args.trim_word_start_offset_ms / 1000.0)
    start = min(max(0, start), max(0, len(audio) - 1))
    tail_len = round(sample_rate * args.trim_word_tail_ms / 1000.0)
    end = max(start + 1, len(audio) - max(0, tail_len))
    return audio[start:end].astype(np.float32)


def main() -> None:
    args = parse_args()
    rows = select_rows(
        read_manifest(args.manifest),
        args.include_speakers,
        args.include_words,
        args.max_per_subtype,
        args.seed,
    )
    stubs = list_stubs(args.stub_root)
    skip_labels = {label.upper() for label in args.skip_labels}
    if not rows:
        raise ValueError("No usable B subtype rows selected.")
    if not stubs:
        raise ValueError(f"No stub WAV files found under {args.stub_root}")

    classifier = load_tinycnn_model(args.model_dir) if args.label_source == "model" else None
    args.outdir.mkdir(parents=True, exist_ok=True)

    demo_rows = []
    combined_parts = []
    original_sequence_parts = []
    enhanced_sequence_parts = []
    first_rate, _ = read_wav_float(rows[0]["audio_path"])
    silence = np.zeros(round(first_rate * 0.45), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 0.75), dtype=np.float32)
    sequence_silence = np.zeros(round(first_rate * args.sequence_gap_ms / 1000.0), dtype=np.float32)
    individual_time_suffix = tempo_suffix(args.sequence_tempo, args.sequence_time_mode)

    for idx, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        oracle_subtype = row["b_subtype"]
        if classifier is None:
            predicted_subtype = oracle_subtype
            confidence = 1.0
        else:
            predicted_subtype, confidence = predict_tinycnn_label(audio, sample_rate, classifier)

        prediction_correct = predicted_subtype == oracle_subtype
        enhancement_applied = (
            confidence >= args.confidence_threshold
            and predicted_subtype in stubs
            and predicted_subtype.upper() not in skip_labels
        )

        if enhancement_applied:
            stub, stub_path = load_stub_for_group(predicted_subtype, stubs, sample_rate)
            stub = prepare_stub_for_insert(stub, args)
            stub_duration_ms = len(stub) / sample_rate * 1000.0
            if args.replacement_mode == "masked":
                enhanced, start, end, fitted_len = replace_consonant_with_masked_stub(
                    audio,
                    sample_rate,
                    stub,
                    stub_duration_ms,
                    args,
                )
            else:
                enhanced, start, end, fitted_len = replace_consonant_with_duration_stub(
                    audio,
                    sample_rate,
                    stub,
                    stub_duration_ms,
                    args,
                )
        else:
            stub_path = Path("")
            fitted_len = 0
            enhanced = audio.astype(np.float32)

        rendered_audio = trim_word_boundaries(audio.astype(np.float32), sample_rate, args)
        rendered_enhanced = trim_word_boundaries(enhanced.astype(np.float32), sample_rate, args)
        silence_for_rate = silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.45), dtype=np.float32)
        long_silence_for_rate = (
            long_silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.75), dtype=np.float32)
        )
        ab_audio = np.concatenate([rendered_audio.astype(np.float32), silence_for_rate, rendered_enhanced])

        stem = (
            f"{idx:03d}_{row['speaker']}_{row['word']}"
            f"_oracle-{oracle_subtype}_pred-{predicted_subtype}_{confidence:.2f}"
        )
        original_path = args.outdir / "original" / f"{stem}_original.wav"
        enhanced_path = args.outdir / "enhanced" / f"{stem}_enhanced.wav"
        ab_path = args.outdir / "ab" / f"{stem}_A_original_B_enhanced.wav"
        write_wav_float(original_path, sample_rate, rendered_audio)
        write_wav_float(enhanced_path, sample_rate, rendered_enhanced)
        write_wav_float(ab_path, sample_rate, ab_audio)

        original_time_path = Path("")
        enhanced_time_path = Path("")
        ab_time_path = Path("")
        if args.write_individual_time_versions and individual_time_suffix:
            original_time_path = args.outdir / "original_time" / f"{stem}_original{individual_time_suffix}.wav"
            enhanced_time_path = args.outdir / "enhanced_time" / f"{stem}_enhanced{individual_time_suffix}.wav"
            ab_time_path = args.outdir / "ab_time" / f"{stem}_A_original_B_enhanced{individual_time_suffix}.wav"
            write_sequence_with_tempo(
                original_time_path,
                sample_rate,
                rendered_audio.astype(np.float32),
                args.sequence_tempo,
                args.sequence_time_mode,
            )
            write_sequence_with_tempo(
                enhanced_time_path,
                sample_rate,
                rendered_enhanced.astype(np.float32),
                args.sequence_tempo,
                args.sequence_time_mode,
            )
            write_sequence_with_tempo(
                ab_time_path,
                sample_rate,
                ab_audio.astype(np.float32),
                args.sequence_tempo,
                args.sequence_time_mode,
            )
        combined_parts.extend([ab_audio, long_silence_for_rate])
        original_sequence_parts.extend(
            [resample_to(rendered_audio.astype(np.float32), sample_rate, first_rate), sequence_silence]
        )
        enhanced_sequence_parts.extend(
            [resample_to(rendered_enhanced.astype(np.float32), sample_rate, first_rate), sequence_silence]
        )

        demo_rows.append(
            {
                "speaker": row["speaker"],
                "word": row["word"],
                "oracle_subtype": oracle_subtype,
                "predicted_subtype": predicted_subtype,
                "confidence": f"{confidence:.4f}",
                "prediction_correct": str(prediction_correct).lower(),
                "enhancement_applied": str(enhancement_applied).lower(),
                "stub_path": str(stub_path),
                "stub_duration_sec": f"{fitted_len / sample_rate:.4f}",
                "source_audio": row["audio_path"],
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "ab_audio": str(ab_path),
                "original_time_audio": str(original_time_path),
                "enhanced_time_audio": str(enhanced_time_path),
                "ab_time_audio": str(ab_time_path),
                "listener_score_1_to_5": "",
                "listener_preference": "",
                "notes": "",
            }
        )

    combined_path = args.outdir / f"b_initial_stub_library_{args.label_source}_A_original_B_enhanced.wav"
    write_wav_float(combined_path, first_rate, np.concatenate(combined_parts))
    suffix = tempo_suffix(args.sequence_tempo, args.sequence_time_mode)
    original_sequence = np.concatenate(original_sequence_parts)
    enhanced_sequence = np.concatenate(enhanced_sequence_parts)
    original_sequence_path = args.outdir / f"b_initial_stub_library_{args.label_source}_original_sequence{suffix}.wav"
    enhanced_sequence_path = args.outdir / f"b_initial_stub_library_{args.label_source}_enhanced_sequence{suffix}.wav"
    sequence_ab_path = args.outdir / f"b_initial_stub_library_{args.label_source}_sequence_A_original_B_enhanced{suffix}.wav"
    write_sequence_with_tempo(
        original_sequence_path,
        first_rate,
        original_sequence,
        args.sequence_tempo,
        args.sequence_time_mode,
    )
    write_sequence_with_tempo(
        enhanced_sequence_path,
        first_rate,
        enhanced_sequence,
        args.sequence_tempo,
        args.sequence_time_mode,
    )
    original_rate, original_sequence = read_wav_float(original_sequence_path)
    enhanced_rate, enhanced_sequence = read_wav_float(enhanced_sequence_path)
    if original_rate != enhanced_rate:
        enhanced_sequence = resample_to(enhanced_sequence, enhanced_rate, original_rate)
        enhanced_rate = original_rate
    write_wav_float(
        sequence_ab_path,
        original_rate,
        np.concatenate(
            [
                original_sequence,
                np.zeros(round(original_rate * 1.2), dtype=np.float32),
                enhanced_sequence,
            ]
        ),
    )
    manifest_path = args.outdir / "listening_evaluation_template.csv"
    write_manifest(manifest_path, demo_rows)

    correct = sum(row["prediction_correct"] == "true" for row in demo_rows)
    summary = {
        "label_source": args.label_source,
        "model_dir": str(args.model_dir) if classifier is not None else None,
        "stub_root": str(args.stub_root),
        "selected_count": len(demo_rows),
        "prediction_accuracy_on_selected": round(correct / len(demo_rows), 4) if demo_rows else None,
        "stub_time_scale": args.stub_time_scale,
        "stub_time_mode": args.stub_time_mode,
        "combined_ab_audio": str(combined_path),
        "original_sequence_audio": str(original_sequence_path),
        "enhanced_sequence_audio": str(enhanced_sequence_path),
        "sequence_ab_audio": str(sequence_ab_path),
        "sequence_gap_ms": args.sequence_gap_ms,
        "sequence_tempo": args.sequence_tempo,
        "sequence_time_mode": args.sequence_time_mode,
        "write_individual_time_versions": args.write_individual_time_versions,
        "trim_word_leading_silence": args.trim_word_leading_silence,
        "trim_word_tail_ms": args.trim_word_tail_ms,
        "trim_word_start_offset_ms": args.trim_word_start_offset_ms,
        "replacement_mode": args.replacement_mode,
        "skip_labels": sorted(skip_labels),
        "mask_extra_ms": args.mask_extra_ms if args.replacement_mode == "masked" else None,
        "mask_min_ms": args.mask_min_ms if args.replacement_mode == "masked" else None,
        "mask_max_ms": args.mask_max_ms if args.replacement_mode == "masked" else None,
        "stub_rms_ratio": args.stub_rms_ratio if args.replacement_mode == "masked" else None,
        "masked_tail_mode": args.masked_tail_mode if args.replacement_mode == "masked" else None,
        "listening_evaluation_template": str(manifest_path),
        "available_stub_groups": sorted(stubs),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved combined A/B audio: {combined_path}")
    print(f"Saved listening template: {manifest_path}")


if __name__ == "__main__":
    main()
