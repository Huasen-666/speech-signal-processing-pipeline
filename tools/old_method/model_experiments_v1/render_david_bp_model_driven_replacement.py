from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.ml_features import extract_feature_set

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


L2_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render David B/P consonant replacement driven by an onset B/P classifier. "
            "Subtype is still taken from known word context."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/david_bp_dataset_manifest.csv"))
    parser.add_argument("--b-subtype-manifest", type=Path, default=Path("data/metadata/david_b_subtype_manifest.csv"))
    parser.add_argument("--p-subtype-manifest", type=Path, default=Path("data/metadata/david_p_subtype_manifest.csv"))
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/david_bp_model_driven_replacement"))
    parser.add_argument("--feature-set", choices=["onset_only", "bp_onset", "mfcc_logmel_onset"], default="bp_onset")
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--crossfade-ms", type=float, default=20.0)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=80.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--stub-rms-ratio", type=float, default=1.22)
    parser.add_argument("--sequence-gap-ms", type=float, default=160.0)
    parser.add_argument("--sequence-tempo", type=float, default=1.0)
    parser.add_argument("--sequence-time-mode", choices=["tempo", "speed"], default="tempo")
    parser.add_argument("--trim-word-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-word-leading-silence", dest="trim_word_leading_silence", action="store_false")
    parser.add_argument("--trim-word-start-offset-ms", type=float, default=5.0)
    parser.add_argument("--trim-word-tail-ms", type=float, default=24.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_audio_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_energy_onset(audio: np.ndarray, sample_rate: int, frame_ms: float = 8.0, hop_ms: float = 2.0) -> int:
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame:
        return 0
    starts = np.arange(0, len(audio) - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(audio[start : start + frame] ** 2) + 1e-12) for start in starts])
    db = 20.0 * np.log10(rms + 1e-12)
    threshold = float(db.max() - 25.0)
    active = np.flatnonzero(db >= threshold)
    return int(starts[active[0]]) if len(active) else 0


def window_from_onset(audio: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    onset = find_energy_onset(audio, sample_rate)
    n_samples = max(1, round(sample_rate * window_ms / 1000.0))
    segment = audio[onset : onset + n_samples]
    min_len = round(sample_rate * 30.0 / 1000.0)
    if len(segment) < min_len:
        segment = np.pad(segment, (0, min_len - len(segment)))
    return segment.astype(np.float32)


def take_for_row(row: dict[str, str]) -> int:
    file_index = int(row.get("file_index") or row.get("protocol_index") or "0")
    return 1 if file_index <= 50 else 2


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (eval_x - mean) / std, mean, std


def fit_logistic_newton(x: np.ndarray, y: np.ndarray, l2: float, iterations: int = 80) -> np.ndarray:
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    weights = np.zeros(x_aug.shape[1], dtype=np.float64)
    regularizer = np.eye(x_aug.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0

    for _ in range(iterations):
        logits = np.clip(x_aug @ weights, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.clip(prob * (1.0 - prob), 1e-6, None)
        grad = x_aug.T @ (prob - y) / len(y) + regularizer @ weights
        hessian = (x_aug * curvature[:, None]).T @ x_aug / len(y) + regularizer
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, grad, rcond=None)[0]
        weights -= step
    return weights


def predict_prob_p(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    logits = np.clip(x_aug @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def make_train_folds(labels: np.ndarray, folds: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.zeros(len(labels), dtype=int)
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        for idx, row_idx in enumerate(indices):
            fold_ids[row_idx] = idx % folds
    return fold_ids


def tune_l2(train_x: np.ndarray, train_y: np.ndarray) -> float:
    fold_ids = make_train_folds(train_y)
    best_l2 = L2_GRID[0]
    best_acc = -1.0
    for l2 in L2_GRID:
        pred = np.zeros(len(train_y), dtype=int)
        for fold in sorted(set(fold_ids.tolist())):
            fit_idx = fold_ids != fold
            val_idx = fold_ids == fold
            fit_x, val_x, _, _ = standardize(train_x[fit_idx], train_x[val_idx])
            weights = fit_logistic_newton(fit_x, train_y[fit_idx].astype(np.float64), l2=l2)
            prob = predict_prob_p(weights, val_x)
            pred[val_idx] = (prob >= 0.5).astype(int)
        acc = float((pred == train_y).mean())
        if acc > best_acc:
            best_acc = acc
            best_l2 = l2
    return best_l2


def subtype_lookup(path: Path, subtype_column: str) -> dict[tuple[str, str], str]:
    lookup = {}
    for row in read_csv(path):
        if row.get("usable", "").lower() != "true":
            continue
        lookup[(row["label"].split("_", 1)[0].upper(), row["word"].lower())] = row[subtype_column]
    return lookup


def convert_subtype(subtype: str, target_label: str, available: set[str]) -> str:
    suffix = subtype.split("_", 1)[1] if "_" in subtype else ""
    candidate = f"{target_label}_{suffix}" if suffix else ""
    if candidate in available:
        return candidate
    fallback_by_suffix = {
        "L": f"{target_label}_AE",
        "R": f"{target_label}_AH",
        "AA": f"{target_label}_AH",
        "OW": f"{target_label}_UH",
        "OY": f"{target_label}_EY",
        "IY": f"{target_label}_IH",
    }
    fallback = fallback_by_suffix.get(suffix, f"{target_label}_AH")
    if fallback in available:
        return fallback
    return sorted(available)[0]


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
    rows = [
        row
        for row in read_csv(args.manifest)
        if row.get("usable", "").lower() == "true" and row.get("label", "").upper() in {"B", "P"}
    ]
    if not rows:
        raise ValueError("No usable David B/P rows found.")

    b_stubs = list_stubs(args.b_stub_root)
    p_stubs = list_stubs(args.p_stub_root)
    if not b_stubs or not p_stubs:
        raise ValueError("Both B and P stub libraries are required.")
    b_available = set(b_stubs)
    p_available = set(p_stubs)
    subtype_by_word = {}
    subtype_by_word.update(subtype_lookup(args.b_subtype_manifest, "b_subtype"))
    subtype_by_word.update(subtype_lookup(args.p_subtype_manifest, "p_subtype"))

    features = []
    labels = []
    takes = []
    audio_cache = []
    for row in rows:
        sample_rate, audio = read_wav_float(resolve_audio_path(row["audio_path"]))
        segment = window_from_onset(audio, sample_rate, args.window_ms)
        features.append(extract_feature_set(segment, sample_rate, args.feature_set))
        labels.append(1 if row["label"].upper() == "P" else 0)
        takes.append(take_for_row(row))
        audio_cache.append((sample_rate, audio.astype(np.float32)))

    x = np.vstack(features)
    y = np.array(labels, dtype=int)
    takes_np = np.array(takes, dtype=int)
    prob_p = np.zeros(len(rows), dtype=np.float64)
    trained_l2: dict[str, float] = {}

    for train_take, predict_take in [(1, 2), (2, 1)]:
        train_idx = takes_np == train_take
        pred_idx = takes_np == predict_take
        best_l2 = tune_l2(x[train_idx], y[train_idx])
        train_x, pred_x, _, _ = standardize(x[train_idx], x[pred_idx])
        weights = fit_logistic_newton(train_x, y[train_idx].astype(np.float64), l2=best_l2)
        prob_p[pred_idx] = predict_prob_p(weights, pred_x)
        trained_l2[f"train_take_{train_take}_predict_take_{predict_take}"] = best_l2

    pred_labels = np.where(prob_p >= 0.5, "P", "B")
    true_labels = np.array([row["label"].upper() for row in rows])
    confidence = np.maximum(prob_p, 1.0 - prob_p)
    accuracy = float((pred_labels == true_labels).mean())

    args.outdir.mkdir(parents=True, exist_ok=True)
    rargs = render_args(args)
    first_rate = audio_cache[0][0]
    sequence_silence = np.zeros(round(first_rate * args.sequence_gap_ms / 1000.0), dtype=np.float32)
    ab_silence = np.zeros(round(first_rate * 0.42), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 0.72), dtype=np.float32)

    original_sequence_parts = []
    enhanced_sequence_parts = []
    combined_ab_parts = []
    detail_rows = []

    for index, (row, predicted_label, prob, conf, (sample_rate, audio)) in enumerate(
        zip(rows, pred_labels, prob_p, confidence, audio_cache),
        start=1,
    ):
        true_label = row["label"].upper()
        word = row["word"].lower()
        true_subtype = subtype_by_word.get((true_label, word), "")
        available = b_available if predicted_label == "B" else p_available
        stubs = b_stubs if predicted_label == "B" else p_stubs
        predicted_subtype = (
            true_subtype
            if true_subtype.startswith(f"{predicted_label}_")
            else convert_subtype(true_subtype, predicted_label, available)
        )
        should_replace = bool(predicted_subtype in available and conf >= args.confidence_threshold)

        if should_replace:
            stub, stub_path = load_stub_for_group(predicted_subtype, stubs, sample_rate)
            stub = prepare_stub_for_insert(stub, rargs)
            stub_duration_ms = len(stub) / sample_rate * 1000.0
            enhanced, start, end, fitted_len = replace_consonant_with_masked_stub(
                audio,
                sample_rate,
                stub,
                stub_duration_ms,
                rargs,
            )
        else:
            stub_path = Path("")
            enhanced = audio.copy()
            start = end = fitted_len = 0

        rendered_original = trim_word_boundaries(audio.astype(np.float32), sample_rate, rargs)
        rendered_enhanced = trim_word_boundaries(enhanced.astype(np.float32), sample_rate, rargs)
        silence_for_rate = ab_silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.42), dtype=np.float32)
        long_silence_for_rate = (
            long_silence if sample_rate == first_rate else np.zeros(round(sample_rate * 0.72), dtype=np.float32)
        )
        ab_audio = np.concatenate([rendered_original, silence_for_rate, rendered_enhanced]).astype(np.float32)

        stem = (
            f"{index:03d}_{row['file_index']}_{word}_true-{true_label}_{true_subtype}"
            f"_pred-{predicted_label}_{predicted_subtype}_{conf:.2f}"
        )
        original_path = args.outdir / "original" / f"{stem}_original.wav"
        enhanced_path = args.outdir / "enhanced" / f"{stem}_enhanced.wav"
        ab_path = args.outdir / "ab" / f"{stem}_A_original_B_enhanced.wav"
        write_wav_float(original_path, sample_rate, rendered_original)
        write_wav_float(enhanced_path, sample_rate, rendered_enhanced)
        write_wav_float(ab_path, sample_rate, ab_audio)

        combined_ab_parts.extend([resample_to(ab_audio, sample_rate, first_rate), long_silence_for_rate])
        original_sequence_parts.extend([resample_to(rendered_original, sample_rate, first_rate), sequence_silence])
        enhanced_sequence_parts.extend([resample_to(rendered_enhanced, sample_rate, first_rate), sequence_silence])

        detail_rows.append(
            {
                "index": str(index),
                "take": str(take_for_row(row)),
                "word": word,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "prob_p": f"{prob:.4f}",
                "confidence": f"{conf:.4f}",
                "bp_prediction_correct": str(predicted_label == true_label).lower(),
                "true_subtype": true_subtype,
                "predicted_subtype_for_replacement": predicted_subtype,
                "replacement_applied": str(should_replace).lower(),
                "stub_path": str(stub_path),
                "mask_start_sec": f"{start / sample_rate:.4f}" if should_replace else "",
                "mask_end_sec": f"{end / sample_rate:.4f}" if should_replace else "",
                "stub_duration_sec": f"{fitted_len / sample_rate:.4f}" if should_replace else "",
                "source_audio": row["audio_path"],
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "ab_audio": str(ab_path),
                "listener_score_1_to_5": "",
                "notes": "",
            }
        )

    suffix = tempo_suffix(args.sequence_tempo, args.sequence_time_mode)
    original_sequence_path = args.outdir / f"david_bp_model_driven_original_sequence{suffix}.wav"
    enhanced_sequence_path = args.outdir / f"david_bp_model_driven_enhanced_sequence{suffix}.wav"
    sequence_ab_path = args.outdir / f"david_bp_model_driven_sequence_A_original_B_enhanced{suffix}.wav"
    combined_ab_path = args.outdir / "david_bp_model_driven_all_A_original_B_enhanced.wav"

    original_sequence = np.concatenate(original_sequence_parts[:-1]).astype(np.float32)
    enhanced_sequence = np.concatenate(enhanced_sequence_parts[:-1]).astype(np.float32)
    combined_ab = np.concatenate(combined_ab_parts[:-1]).astype(np.float32)
    write_sequence_with_tempo(original_sequence_path, first_rate, original_sequence, args.sequence_tempo, args.sequence_time_mode)
    write_sequence_with_tempo(enhanced_sequence_path, first_rate, enhanced_sequence, args.sequence_tempo, args.sequence_time_mode)
    write_wav_float(combined_ab_path, first_rate, combined_ab)

    original_rate, original_sequence = read_wav_float(original_sequence_path)
    enhanced_rate, enhanced_sequence = read_wav_float(enhanced_sequence_path)
    if enhanced_rate != original_rate:
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

    prediction_manifest = args.outdir / "model_driven_replacement_manifest.csv"
    write_csv(prediction_manifest, detail_rows)

    summary = {
        "method": "cross-take B/P classifier drives replacement; subtype still uses known word context",
        "feature_set": args.feature_set,
        "window_ms": args.window_ms,
        "confidence_threshold": args.confidence_threshold,
        "n_clips": len(rows),
        "bp_accuracy_cross_take_predictions": round(accuracy, 4),
        "take1_accuracy_predicted_by_take2_model": round(
            float((pred_labels[takes_np == 1] == true_labels[takes_np == 1]).mean()), 4
        ),
        "take2_accuracy_predicted_by_take1_model": round(
            float((pred_labels[takes_np == 2] == true_labels[takes_np == 2]).mean()), 4
        ),
        "trained_l2": trained_l2,
        "replacement_applied_count": sum(row["replacement_applied"] == "true" for row in detail_rows),
        "prediction_manifest": str(prediction_manifest),
        "original_sequence_audio": str(original_sequence_path),
        "enhanced_sequence_audio": str(enhanced_sequence_path),
        "sequence_ab_audio": str(sequence_ab_path),
        "combined_ab_audio": str(combined_ab_path),
        "note": (
            "This is not full subtype recognition yet. A deployment-grade version needs a subtype/context model "
            "or a safer B/P-only enhancement mode."
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
