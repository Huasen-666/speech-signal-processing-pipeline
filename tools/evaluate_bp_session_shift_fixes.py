from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from speech_pipeline.ml_features import extract_feature_set, feature_names_for_set

from compare_bp_cross_session_four_arm import (
    AudioItem,
    binary_metrics,
    clip_onset,
    find_energy_onset,
    labels_for_rows,
    load_audio_items,
    load_rows,
    resolve_path,
    write_csv,
)
from compare_hubert_mfcc_cross_session import fit_logistic_newton, predict_prob, tune_l2_on_train


FEATURE_SETS = [
    "mfcc_logmel_onset",
    "bp_onset",
    "onset_logmel_mfcc",
    "relative_bp_onset",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Dave B/P cross-session covariate shift and test fast fixes: "
            "split-level loudness normalization, per-session CMVN, relative onset features, "
            "and small target-session threshold calibration."
        )
    )
    parser.add_argument("--train-manifest", type=Path, default=Path("data/metadata/dave_bp_consecutive_v2_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/metadata/dave_bp_drb_validation_v2_manifest.csv"))
    parser.add_argument("--speaker", default="dave")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/per_patient_bp_eval/session_shift_fixes"))
    parser.add_argument("--windows-ms", type=int, nargs="*", default=[150, 200, 220])
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS)
    parser.add_argument("--audio-normalization", nargs="*", default=["none", "split_rms"], choices=["none", "split_rms"])
    parser.add_argument(
        "--feature-normalization",
        nargs="*",
        default=["train_cmvn", "split_cmvn"],
        choices=["train_cmvn", "split_cmvn"],
    )
    parser.add_argument("--target-rms-dbfs", type=float, default=-20.0)
    parser.add_argument("--calibration-per-class", type=int, nargs="*", default=[0, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="*", default=[13, 29, 42, 71, 97])
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def rms_dbfs(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return -240.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def gain_audio(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10.0 ** (gain_db / 20.0)
    out = audio.astype(np.float32) * gain
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:
        out = out * (0.98 / peak)
    return out.astype(np.float32)


def make_audio_variant(
    rows: list[dict[str, str]],
    audio_items: list[AudioItem],
    mode: str,
    target_rms_dbfs: float,
) -> tuple[list[AudioItem], dict[str, float]]:
    if mode == "none":
        return audio_items, {"train_gain_db": 0.0, "validation_gain_db": 0.0}

    split_rms: dict[str, list[float]] = {"train": [], "validation": []}
    for row, item in zip(rows, audio_items):
        split_rms[row["split"]].append(item.rms_dbfs)
    split_gain = {
        split: target_rms_dbfs - float(np.mean(values))
        for split, values in split_rms.items()
        if values
    }

    normalized_items = []
    for row, item in zip(rows, audio_items):
        audio = gain_audio(item.audio, split_gain.get(row["split"], 0.0))
        onset = find_energy_onset(audio, item.sample_rate)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        normalized_items.append(
            AudioItem(
                audio=audio,
                sample_rate=item.sample_rate,
                onset_sample=onset,
                duration_sec=item.duration_sec,
                peak=peak,
                rms_dbfs=rms_dbfs(audio),
            )
        )
    return normalized_items, {
        "train_gain_db": round(float(split_gain.get("train", 0.0)), 3),
        "validation_gain_db": round(float(split_gain.get("validation", 0.0)), 3),
    }


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "relative_bp_onset":
        return [name for name in feature_names_for_set("bp_onset") if "rms" not in name.lower()]
    return feature_names_for_set(feature_set)


def extract_one_feature(segment: np.ndarray, sample_rate: int, feature_set: str) -> np.ndarray:
    if feature_set == "relative_bp_onset":
        values = extract_feature_set(segment, sample_rate, "bp_onset")
        names = feature_names_for_set("bp_onset")
        keep = [idx for idx, name in enumerate(names) if "rms" not in name.lower()]
        return values[keep]
    return extract_feature_set(segment, sample_rate, feature_set)


def extract_matrix(
    audio_items: list[AudioItem],
    window_ms: int,
    feature_set: str,
) -> np.ndarray:
    rows = []
    for item in audio_items:
        segment = clip_onset(item.audio, item.sample_rate, item.onset_sample, window_ms)
        rows.append(extract_one_feature(segment, item.sample_rate, feature_set))
    return np.vstack(rows).astype(np.float64)


def normalize_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "train_cmvn":
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (train_x - mean) / std, (val_x - mean) / std

    train_mean = train_x.mean(axis=0, keepdims=True)
    train_std = train_x.std(axis=0, keepdims=True)
    val_mean = val_x.mean(axis=0, keepdims=True)
    val_std = val_x.std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    val_std = np.where(val_std < 1e-8, 1.0, val_std)
    return (train_x - train_mean) / train_std, (val_x - val_mean) / val_std


def calibration_indices(labels: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    if per_class <= 0:
        all_idx = np.arange(len(labels), dtype=int)
        return np.array([], dtype=int), all_idx
    selected = []
    for label in [0, 1]:
        label_idx = np.where(labels == label)[0]
        selected.extend(label_idx[: min(per_class, len(label_idx))].tolist())
    selected_arr = np.array(sorted(selected), dtype=int)
    mask = np.ones(len(labels), dtype=bool)
    mask[selected_arr] = False
    return selected_arr, np.where(mask)[0]


def best_threshold(prob: np.ndarray, labels: np.ndarray) -> float:
    if len(prob) == 0:
        return 0.5
    candidates = sorted(set([0.0, 0.5, 1.0] + prob.tolist()))
    best = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        metrics = binary_metrics(labels, pred)
        score = 0.5 * (metrics["B_recall"] + metrics["P_recall"])
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best - 0.5)):
            best = float(threshold)
            best_score = float(score)
    return best


def evaluate_prepared(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    cal_per_class: int,
) -> dict[str, object]:
    best_l2 = tune_l2_on_train(train_x, train_y, seed)
    weights = fit_logistic_newton(train_x, train_y.astype(np.float64), l2=best_l2)
    val_prob = predict_prob(weights, val_x)
    cal_idx, eval_idx = calibration_indices(val_y, cal_per_class)
    threshold = best_threshold(val_prob[cal_idx], val_y[cal_idx]) if len(cal_idx) else 0.5
    eval_prob = val_prob[eval_idx]
    eval_y = val_y[eval_idx]
    pred = (eval_prob >= threshold).astype(int)
    return {
        **binary_metrics(eval_y, pred),
        "eval_count": int(len(eval_idx)),
        "calibration_count": int(len(cal_idx)),
        "threshold": float(threshold),
        "best_l2": float(best_l2),
    }


def summarize_session_qc(
    rows: list[dict[str, str]],
    audio_items: list[AudioItem],
    audio_norm: str,
    gains: dict[str, float],
) -> list[dict[str, object]]:
    out = []
    for split in ["train", "validation"]:
        idx = [i for i, row in enumerate(rows) if row["split"] == split]
        for label in ["all", "B", "P"]:
            label_idx = idx if label == "all" else [i for i in idx if rows[i]["label"] == label]
            if not label_idx:
                continue
            rms_values = np.array([audio_items[i].rms_dbfs for i in label_idx], dtype=float)
            peak_values = np.array([audio_items[i].peak for i in label_idx], dtype=float)
            onset_values = np.array([audio_items[i].onset_sample / audio_items[i].sample_rate for i in label_idx], dtype=float)
            duration_values = np.array([audio_items[i].duration_sec for i in label_idx], dtype=float)
            out.append(
                {
                    "audio_normalization": audio_norm,
                    "split": split,
                    "label": label,
                    "count": len(label_idx),
                    "gain_db": gains.get(f"{split}_gain_db", 0.0),
                    "rms_dbfs_mean": round(float(rms_values.mean()), 3),
                    "rms_dbfs_std": round(float(rms_values.std()), 3),
                    "peak_mean": round(float(peak_values.mean()), 4),
                    "onset_sec_mean": round(float(onset_values.mean()), 4),
                    "onset_sec_std": round(float(onset_values.std()), 4),
                    "duration_sec_mean": round(float(duration_values.mean()), 4),
                }
            )
    return out


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["window_ms"],
            row["feature_set"],
            row["audio_normalization"],
            row["feature_normalization"],
            row["calibration_per_class"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    metrics = ["accuracy", "B_precision", "B_recall", "P_precision", "P_recall", "threshold"]
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        row = {
            "window_ms": key[0],
            "feature_set": key[1],
            "audio_normalization": key[2],
            "feature_normalization": key[3],
            "calibration_per_class": key[4],
            "seed_count": len(group),
            "eval_count": group[0]["eval_count"],
            "best_l2_values": ";".join(sorted(set(str(item["best_l2"]) for item in group))),
        }
        for metric in metrics:
            values = np.array([float(item[metric]) for item in group], dtype=float)
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)
        summary.append(row)
    return summary


def main() -> None:
    args = parse_args()
    args.outdir = resolve_path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_manifest, args.speaker, "train")
    validation_rows = load_rows(args.validation_manifest, args.speaker, "validation")
    rows = train_rows + validation_rows
    labels = labels_for_rows(rows)
    train_idx = np.arange(len(train_rows), dtype=int)
    val_idx = np.arange(len(train_rows), len(rows), dtype=int)
    val_y = labels[val_idx]

    base_audio = load_audio_items(rows)
    qc_rows = []
    seed_rows = []

    log(f"Loaded train={len(train_rows)} validation={len(validation_rows)}")
    for audio_norm in args.audio_normalization:
        audio_items, gains = make_audio_variant(rows, base_audio, audio_norm, args.target_rms_dbfs)
        qc_rows.extend(summarize_session_qc(rows, audio_items, audio_norm, gains))
        log(f"Audio normalization={audio_norm} gains={gains}")

        for window_ms in args.windows_ms:
            for feature_set in args.feature_sets:
                x = extract_matrix(audio_items, window_ms, feature_set)
                train_x_raw = x[train_idx]
                val_x_raw = x[val_idx]
                for feature_norm in args.feature_normalization:
                    train_x, val_x = normalize_features(train_x_raw, val_x_raw, feature_norm)
                    for cal_per_class in args.calibration_per_class:
                        for seed in args.seeds:
                            result = evaluate_prepared(
                                train_x,
                                labels[train_idx],
                                val_x,
                                val_y,
                                seed,
                                cal_per_class,
                            )
                            seed_rows.append(
                                {
                                    "window_ms": window_ms,
                                    "feature_set": feature_set,
                                    "feature_dim": len(feature_names(feature_set)),
                                    "audio_normalization": audio_norm,
                                    "feature_normalization": feature_norm,
                                    "calibration_per_class": cal_per_class,
                                    "seed": seed,
                                    "accuracy": round(float(result["accuracy"]), 4),
                                    "B_precision": round(float(result["B_precision"]), 4),
                                    "B_recall": round(float(result["B_recall"]), 4),
                                    "P_precision": round(float(result["P_precision"]), 4),
                                    "P_recall": round(float(result["P_recall"]), 4),
                                    "threshold": round(float(result["threshold"]), 4),
                                    "best_l2": result["best_l2"],
                                    "eval_count": result["eval_count"],
                                    "calibration_count": result["calibration_count"],
                                }
                            )

    summary_rows = aggregate_rows(seed_rows)
    qc_csv = args.outdir / "session_qc_by_audio_norm.csv"
    seed_csv = args.outdir / "session_shift_fix_by_seed.csv"
    summary_csv = args.outdir / "session_shift_fix_summary.csv"
    write_csv(qc_csv, qc_rows)
    write_csv(seed_csv, seed_rows)
    write_csv(summary_csv, summary_rows)

    best = max(summary_rows, key=lambda row: float(row["accuracy_mean"])) if summary_rows else {}
    payload = {
        "speaker": args.speaker,
        "train_manifest": str(resolve_path(args.train_manifest)),
        "validation_manifest": str(resolve_path(args.validation_manifest)),
        "outputs": {
            "session_qc": str(qc_csv),
            "by_seed": str(seed_csv),
            "summary": str(summary_csv),
        },
        "best_accuracy_row": best,
        "notes": [
            "calibration_per_class uses that many labeled target-session B and P clips to tune only the decision threshold.",
            "split_cmvn uses unlabeled validation-session feature statistics; it tests whether session-level distribution shift is the bottleneck.",
            "split_rms applies one gain per split before onset detection and feature extraction, not per-word gain.",
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Saved QC: {qc_csv}")
    log(f"Saved summary: {summary_csv}")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
