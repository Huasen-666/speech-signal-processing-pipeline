import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float

from render_dave_stub_replacement_demo import (
    adapt_stub_level,
    crossfade_join,
    load_stub,
    load_tinycnn_model,
    predict_tinycnn_label,
    replace_consonant_with_stub,
)


def parse_args() -> argparse.Namespace:
    dave_dir = PROJECT_ROOT.parent / "DAVE"
    parser = argparse.ArgumentParser(
        description="Run a computer-as-phone prototype for B/P consonant cue enhancement."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/bp_consonant_enhancement"))
    parser.add_argument("--model-dir", type=Path, default=Path("experiments/ml_baseline/bp_tinycnn_logmel_mfcc_bascom_corrick"))
    parser.add_argument("--b-stub", type=Path, default=dave_dir / "B consonant DRB 1.wav")
    parser.add_argument("--p-stub", type=Path, default=dave_dir / "P consonant DRB 1.wav")
    parser.add_argument("--include-speakers", nargs="*", default=["bascom", "corrick"])
    parser.add_argument("--labels", nargs="*", default=["B", "P"])
    parser.add_argument("--max-per-label", type=int, default=12, help="Set <=0 to process all selected rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enhancement-mode",
        choices=["stub", "additive"],
        default="stub",
        help="stub uses Dave-style consonant replacement; additive keeps the original audio and boosts consonant cues.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--level-mode", choices=["match_rms", "dave_peak"], default="match_rms")
    parser.add_argument(
        "--stub-duration-mode",
        choices=["full", "adaptive", "word_context"],
        default="adaptive",
        help=(
            "full uses the complete source stub; adaptive estimates cue duration acoustically; "
            "word_context uses the word label and B context table as an oracle duration proxy."
        ),
    )
    parser.add_argument("--b-context-groups", type=Path, default=Path("data/metadata/b_context_groups.csv"))
    parser.add_argument("--stub-fit-mode", choices=["crop", "stretch"], default="crop")
    parser.add_argument("--b-min-stub-ms", type=float, default=45.0)
    parser.add_argument("--b-max-stub-ms", type=float, default=140.0)
    parser.add_argument("--b-default-stub-ms", type=float, default=85.0)
    parser.add_argument("--p-min-stub-ms", type=float, default=65.0)
    parser.add_argument("--p-max-stub-ms", type=float, default=180.0)
    parser.add_argument("--p-default-stub-ms", type=float, default=120.0)
    parser.add_argument("--crossfade-ms", type=float, default=25.0)
    parser.add_argument("--pre-roll-ms", type=float, default=20.0)
    parser.add_argument("--b-low-gain", type=float, default=0.55)
    parser.add_argument("--b-high-gain", type=float, default=0.15)
    parser.add_argument("--p-high-gain", type=float, default=0.85)
    parser.add_argument("--p-low-cut", type=float, default=0.10)
    parser.add_argument("--onset-ms", type=float, default=160.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_b_context_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["word"].lower(): row["suggested_stub_group"] for row in csv.DictReader(handle)}


def select_rows(
    rows: list[dict[str, str]],
    labels: list[str],
    include_speakers: list[str],
    max_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    allowed_labels = {label.upper() for label in labels}
    allowed_speakers = {speaker.lower() for speaker in include_speakers}
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(allowed_labels):
        candidates = [
            row
            for row in rows
            if row["usable"].lower() == "true"
            and row["label"].upper() == label
            and row["speaker"].lower() in allowed_speakers
        ]
        candidates = sorted(candidates, key=lambda row: (row["speaker"], row["protocol_index"], row["word"]))
        if max_per_label > 0 and len(candidates) > max_per_label:
            indices = sorted(rng.choice(len(candidates), size=max_per_label, replace=False).tolist())
            candidates = [candidates[idx] for idx in indices]
        selected.extend(candidates)
    return selected


def detect_active_start(audio: np.ndarray, sample_rate: int) -> int:
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return 0
    energies = []
    starts = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        energies.append(float(np.mean(frame.astype(np.float64) ** 2)))
        starts.append(start)
    if not energies or max(energies) <= 1e-12:
        return 0
    threshold = max(energies) * 0.04
    for start, energy in zip(starts, energies):
        if energy >= threshold:
            return max(0, start - round(sample_rate * 0.02))
    return 0


def safe_sosfiltfilt(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if len(audio) < 32:
        return audio.astype(np.float64)
    return sosfiltfilt(sos, audio.astype(np.float64))


def fade_envelope(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float64)
    env = np.hanning(length * 2)[:length]
    return np.maximum(env, 0.15)


def additive_consonant_enhancement(
    audio: np.ndarray,
    sample_rate: int,
    label: str,
    onset_ms: float,
    b_low_gain: float,
    b_high_gain: float,
    p_high_gain: float,
    p_low_cut: float,
) -> tuple[np.ndarray, int, int]:
    mono = np.asarray(audio, dtype=np.float64)
    if len(mono) == 0:
        return mono.astype(np.float32), 0, 0

    start = detect_active_start(mono, sample_rate)
    end = min(len(mono), start + round(sample_rate * onset_ms / 1000.0))
    if end <= start:
        return mono.astype(np.float32), start, end

    high_sos = butter(2, 1200.0, btype="highpass", fs=sample_rate, output="sos")
    low_sos = butter(2, [80.0, 600.0], btype="bandpass", fs=sample_rate, output="sos")
    high = safe_sosfiltfilt(high_sos, mono)
    low = safe_sosfiltfilt(low_sos, mono)
    env = fade_envelope(end - start)

    enhanced = mono.copy()
    if label == "P":
        enhanced[start:end] += p_high_gain * env * high[start:end]
        enhanced[start:end] -= p_low_cut * env * low[start:end]
    else:
        enhanced[start:end] += b_low_gain * env * low[start:end]
        enhanced[start:end] += b_high_gain * env * high[start:end]

    peak = float(np.max(np.abs(enhanced))) if len(enhanced) else 0.0
    if peak > 0.98:
        enhanced = enhanced / peak * 0.98
    return enhanced.astype(np.float32), start, end


def label_duration_bounds(label: str, args: argparse.Namespace) -> tuple[float, float, float]:
    if label == "P":
        return args.p_min_stub_ms, args.p_max_stub_ms, args.p_default_stub_ms
    return args.b_min_stub_ms, args.b_max_stub_ms, args.b_default_stub_ms


def estimate_consonant_end(
    audio: np.ndarray,
    sample_rate: int,
    start: int,
    label: str,
    args: argparse.Namespace,
) -> int:
    min_ms, max_ms, default_ms = label_duration_bounds(label, args)
    min_len = round(sample_rate * min_ms / 1000.0)
    max_len = round(sample_rate * max_ms / 1000.0)
    default_len = round(sample_rate * default_ms / 1000.0)
    analysis = np.asarray(audio[start : start + max_len], dtype=np.float64)
    if len(analysis) < round(sample_rate * 0.03):
        return min(len(audio), start + min(default_len, len(analysis)))

    frame_len = max(1, round(sample_rate * 0.015))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(analysis) < frame_len:
        return min(len(audio), start + min(default_len, len(analysis)))

    starts = []
    rms_values = []
    for local_start in range(0, len(analysis) - frame_len + 1, hop_len):
        frame = analysis[local_start : local_start + frame_len]
        starts.append(local_start)
        rms_values.append(float(np.sqrt(np.mean(frame**2))))

    if not rms_values or max(rms_values) <= 1e-10:
        return min(len(audio), start + default_len)

    rms_db = 20.0 * np.log10(np.maximum(rms_values, 1e-8))
    high_db = float(np.percentile(rms_db, 90))
    stable_vowel_threshold = high_db - 7.0
    min_start = min_len

    # Look for the first sustained high-energy region after the consonant cue.
    # This is a lightweight proxy for vowel onset until we add forced alignment.
    for idx, local_start in enumerate(starts):
        if local_start < min_start:
            continue
        future = rms_db[idx : idx + 3]
        if len(future) >= 2 and float(np.median(future)) >= stable_vowel_threshold:
            return min(len(audio), start + local_start)

    return min(len(audio), start + min(max(default_len, min_len), max_len))


def fit_stub_to_length(stub: np.ndarray, target_len: int, fit_mode: str) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(1, dtype=np.float32)
    if len(stub) == target_len:
        return stub.astype(np.float32)
    if fit_mode == "crop":
        if len(stub) > target_len:
            fitted = stub[:target_len]
        else:
            fitted = np.pad(stub, (0, target_len - len(stub)))
        return fitted.astype(np.float32)
    return resample(stub.astype(np.float64), target_len).astype(np.float32)


def replace_consonant_with_adaptive_stub(
    audio: np.ndarray,
    sample_rate: int,
    stub: np.ndarray,
    label: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int]:
    start = detect_active_start(audio, sample_rate)
    end = estimate_consonant_end(audio, sample_rate, start, label, args)
    target_len = max(1, end - start)
    fitted_stub = fit_stub_to_length(stub, target_len, args.stub_fit_mode)
    source_segment = audio[start:end]
    adapted_stub = adapt_stub_level(
        fitted_stub,
        source_segment,
        args.level_mode,
        rms_ratio=1.0,
        max_stub_peak=0.9,
    )

    before = audio[:start].astype(np.float32)
    tail = audio[end:].astype(np.float32)
    fade_len = min(round(sample_rate * args.crossfade_ms / 1000.0), max(1, target_len // 2))
    replaced_body = crossfade_join(adapted_stub, tail, fade_len)
    replaced = np.concatenate([before, replaced_body]).astype(np.float32)

    peak = float(np.max(np.abs(replaced))) if len(replaced) else 0.0
    if peak > 0.98:
        replaced = (replaced / peak * 0.98).astype(np.float32)
    return replaced, start, end, target_len


def b_context_duration_ms(group: str, default_ms: float) -> float:
    durations = {
        "B_front_high": 60.0,
        "B_front_high_lax": 60.0,
        "B_front_mid": 65.0,
        "B_front_low": 70.0,
        "B_central": 85.0,
        "B_back_low": 90.0,
        "B_back_rounded": 95.0,
        "B_back_diphthong": 95.0,
        "B_back_high": 95.0,
        "B_back_high_lax": 90.0,
        "B_r_colored": 110.0,
        "B_open_diphthong": 85.0,
        "B_rounded_diphthong": 90.0,
        "B_L_cluster": 120.0,
    }
    return durations.get(group, default_ms)


def word_context_duration_ms(row: dict[str, str], label: str, context_groups: dict[str, str], args: argparse.Namespace) -> float:
    if label == "P":
        return args.p_default_stub_ms
    group = context_groups.get(row["word"].lower(), "")
    return b_context_duration_ms(group, args.b_default_stub_ms)


def replace_consonant_with_duration_stub(
    audio: np.ndarray,
    sample_rate: int,
    stub: np.ndarray,
    duration_ms: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int]:
    start = detect_active_start(audio, sample_rate)
    target_len = max(1, round(sample_rate * duration_ms / 1000.0))
    end = min(len(audio), start + target_len)
    target_len = max(1, end - start)
    fitted_stub = fit_stub_to_length(stub, target_len, args.stub_fit_mode)
    source_segment = audio[start:end]
    adapted_stub = adapt_stub_level(
        fitted_stub,
        source_segment,
        args.level_mode,
        rms_ratio=1.0,
        max_stub_peak=0.9,
    )
    before = audio[:start].astype(np.float32)
    tail = audio[end:].astype(np.float32)
    fade_len = min(round(sample_rate * args.crossfade_ms / 1000.0), max(1, target_len // 2))
    replaced_body = crossfade_join(adapted_stub, tail, fade_len)
    replaced = np.concatenate([before, replaced_body]).astype(np.float32)
    peak = float(np.max(np.abs(replaced))) if len(replaced) else 0.0
    if peak > 0.98:
        replaced = (replaced / peak * 0.98).astype(np.float32)
    return replaced, start, end, target_len


def write_evaluation_template(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "speaker",
        "word",
        "true_label",
        "predicted_label",
        "confidence",
        "prediction_correct",
        "enhancement_mode",
        "enhancement_applied",
        "start_sec",
        "end_sec",
        "fitted_stub_duration_sec",
        "stub_duration_mode",
        "stub_fit_mode",
        "source_audio",
        "original_audio",
        "enhanced_audio",
        "ab_audio",
        "listener_score_1_to_5",
        "listener_preference",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], args: argparse.Namespace) -> dict:
    correct = sum(row["prediction_correct"] == "true" for row in rows)
    applied = sum(row["enhancement_applied"] == "true" for row in rows)
    by_label: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row["true_label"]
        stats = by_label.setdefault(label, {"count": 0, "correct": 0, "enhanced": 0})
        stats["count"] += 1
        stats["correct"] += int(row["prediction_correct"] == "true")
        stats["enhanced"] += int(row["enhancement_applied"] == "true")
    for stats in by_label.values():
        stats["accuracy"] = round(stats["correct"] / stats["count"], 4) if stats["count"] else None
    return {
        "prototype_role": "computer_as_phone_processor",
        "model": "TinyCNN log-mel/MFCC",
        "model_dir": str(args.model_dir),
        "enhancement_mode": args.enhancement_mode,
        "stub_duration_mode": args.stub_duration_mode if args.enhancement_mode == "stub" else None,
        "stub_fit_mode": args.stub_fit_mode if args.enhancement_mode == "stub" else None,
        "confidence_threshold": args.confidence_threshold,
        "selected_count": len(rows),
        "prediction_accuracy_on_selected": round(correct / len(rows), 4) if rows else None,
        "enhanced_count": applied,
        "by_label": by_label,
        "next_human_evaluation": {
            "listener_score_1_to_5": "1=worse, 3=no clear difference, 5=much clearer",
            "listener_preference": "original / enhanced / no_preference",
        },
    }


def main() -> None:
    args = parse_args()
    rows = select_rows(
        read_manifest(args.manifest),
        args.labels,
        args.include_speakers,
        args.max_per_label,
        args.seed,
    )
    if not rows:
        raise ValueError("No usable rows selected.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    classifier = load_tinycnn_model(args.model_dir)
    b_context_groups = read_b_context_groups(args.b_context_groups)
    first_rate, _ = read_wav_float(rows[0]["audio_path"])
    stubs = {
        "B": load_stub(args.b_stub, first_rate, 0.5),
        "P": load_stub(args.p_stub, first_rate, 0.5),
    }

    evaluation_rows = []
    combined_parts = []
    silence = np.zeros(round(first_rate * 0.45), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 0.8), dtype=np.float32)

    for idx, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        predicted_label, confidence = predict_tinycnn_label(audio, sample_rate, classifier)
        prediction_correct = predicted_label == row["label"].upper()
        enhancement_applied = confidence >= args.confidence_threshold and predicted_label in {"B", "P"}

        if sample_rate == first_rate:
            stubs_for_rate = stubs
            silence_for_rate = silence
            long_silence_for_rate = long_silence
        else:
            stubs_for_rate = {
                "B": load_stub(args.b_stub, sample_rate, 0.5),
                "P": load_stub(args.p_stub, sample_rate, 0.5),
            }
            silence_for_rate = np.zeros(round(sample_rate * 0.45), dtype=np.float32)
            long_silence_for_rate = np.zeros(round(sample_rate * 0.8), dtype=np.float32)

        if not enhancement_applied:
            enhanced = audio.astype(np.float32)
            start = 0
            end = 0
            fitted_stub_duration_sec = 0.0
        elif args.enhancement_mode == "stub":
            if args.stub_duration_mode == "word_context":
                duration_ms = word_context_duration_ms(row, predicted_label, b_context_groups, args)
                enhanced, start, end, fitted_stub_len = replace_consonant_with_duration_stub(
                    audio,
                    sample_rate,
                    stubs_for_rate[predicted_label],
                    duration_ms,
                    args,
                )
                fitted_stub_duration_sec = fitted_stub_len / sample_rate
            elif args.stub_duration_mode == "adaptive":
                enhanced, start, end, fitted_stub_len = replace_consonant_with_adaptive_stub(
                    audio,
                    sample_rate,
                    stubs_for_rate[predicted_label],
                    predicted_label,
                    args,
                )
                fitted_stub_duration_sec = fitted_stub_len / sample_rate
            else:
                enhanced, start, end = replace_consonant_with_stub(
                    audio,
                    sample_rate,
                    stubs_for_rate[predicted_label],
                    args.level_mode,
                    rms_ratio=1.0,
                    max_stub_peak=0.9,
                    crossfade_ms=args.crossfade_ms,
                    pre_roll_ms=args.pre_roll_ms,
                )
                fitted_stub_duration_sec = len(stubs_for_rate[predicted_label]) / sample_rate
        else:
            enhanced, start, end = additive_consonant_enhancement(
                audio,
                sample_rate,
                predicted_label,
                args.onset_ms,
                args.b_low_gain,
                args.b_high_gain,
                args.p_high_gain,
                args.p_low_cut,
            )
            fitted_stub_duration_sec = 0.0

        ab_audio = np.concatenate([audio.astype(np.float32), silence_for_rate, enhanced])
        stem = (
            f"{idx:03d}_{row['speaker']}_{row['label']}_{row['word']}"
            f"_pred-{predicted_label}_{confidence:.2f}_{args.enhancement_mode}"
        )
        original_path = args.outdir / "original" / f"{stem}_original.wav"
        enhanced_path = args.outdir / "enhanced" / f"{stem}_enhanced.wav"
        ab_path = args.outdir / "ab" / f"{stem}_A_original_B_enhanced.wav"

        write_wav_float(original_path, sample_rate, audio)
        write_wav_float(enhanced_path, sample_rate, enhanced)
        write_wav_float(ab_path, sample_rate, ab_audio)
        combined_parts.extend([ab_audio, long_silence_for_rate])

        evaluation_rows.append(
            {
                "speaker": row["speaker"],
                "word": row["word"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "confidence": f"{confidence:.4f}",
                "prediction_correct": str(prediction_correct).lower(),
                "enhancement_mode": args.enhancement_mode,
                "enhancement_applied": str(enhancement_applied).lower(),
                "start_sec": f"{start / sample_rate:.4f}",
                "end_sec": f"{end / sample_rate:.4f}",
                "fitted_stub_duration_sec": f"{fitted_stub_duration_sec:.4f}",
                "stub_duration_mode": args.stub_duration_mode if args.enhancement_mode == "stub" else "",
                "stub_fit_mode": args.stub_fit_mode if args.enhancement_mode == "stub" else "",
                "source_audio": row["audio_path"],
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "ab_audio": str(ab_path),
                "listener_score_1_to_5": "",
                "listener_preference": "",
                "notes": "",
            }
        )

    combined_path = args.outdir / f"phone_prototype_all_A_original_B_enhanced_{args.enhancement_mode}.wav"
    write_wav_float(combined_path, first_rate, np.concatenate(combined_parts))
    evaluation_path = args.outdir / "listening_evaluation_template.csv"
    write_evaluation_template(evaluation_path, evaluation_rows)
    summary = summarize(evaluation_rows, args)
    summary["combined_ab_audio"] = str(combined_path)
    summary["listening_evaluation_template"] = str(evaluation_path)
    (args.outdir / "profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved combined A/B audio: {combined_path}")
    print(f"Saved listening template: {evaluation_path}")


if __name__ == "__main__":
    main()
