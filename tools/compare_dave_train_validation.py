from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import extract_feature_set

from compare_hubert_mfcc_cross_session import (
    clip_onset,
    evaluate_mfcc_cross_session,
    extract_hubert_features,
    find_energy_onset,
    log,
    top_layer_summary,
    train_weighted_layer_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on Dave's first B/P recording and validate on Dave's second B/P recording."
    )
    parser.add_argument("--train-manifest", type=Path, default=Path("data/metadata/dave_bp_consecutive_v2_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/metadata/dave_bp_drb_validation_v2_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/per_patient_bp_eval/dave_train_validation"))
    parser.add_argument("--speaker", default="dave")
    parser.add_argument("--model", default="facebook/hubert-base-ls960")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--windows-ms", type=int, nargs="*", default=[150, 200, 320])
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def row_is_usable(row: dict[str, str]) -> bool:
    usable = row.get("usable", "")
    return usable == "" or usable.strip().lower() in {"true", "1", "yes", "y"}


def load_rows(manifest: Path, speaker: str, split: str) -> list[dict[str, str]]:
    rows = []
    manifest = resolve_path(manifest)
    for row in read_csv(manifest):
        if not row_is_usable(row):
            continue
        if row.get("speaker", "").lower() != speaker.lower():
            continue
        if row.get("label", "").upper() not in {"B", "P"}:
            continue
        audio_path = resolve_path(Path(row["audio_path"]))
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        normalized = dict(row)
        normalized["audio_path_abs"] = str(audio_path)
        normalized["split"] = split
        normalized.setdefault("file_index", row.get("word_index", row.get("global_index", "")))
        normalized.setdefault("protocol_index", row.get("word_index", ""))
        rows.append(normalized)
    rows.sort(key=lambda item: (item["label"], int(item.get("word_index") or item.get("file_index") or "0")))
    return rows


def load_audio_cache(rows: list[dict[str, str]]) -> list[tuple[np.ndarray, int, int]]:
    cache = []
    for row in rows:
        sample_rate, audio = read_wav_float(Path(row["audio_path_abs"]))
        onset = find_energy_onset(audio, sample_rate)
        cache.append((audio.astype(np.float32), sample_rate, onset))
    return cache


def labels_for_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([1 if row["label"].upper() == "P" else 0 for row in rows], dtype=int)


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.outdir = resolve_path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_manifest, args.speaker, "train")
    validation_rows = load_rows(args.validation_manifest, args.speaker, "validation")
    if not train_rows or not validation_rows:
        raise ValueError("Both train and validation manifests must contain usable B/P rows.")
    all_rows = train_rows + validation_rows
    labels = labels_for_rows(all_rows)
    train_idx = np.arange(len(train_rows), dtype=int)
    val_idx = np.arange(len(train_rows), len(all_rows), dtype=int)
    audio_cache = load_audio_cache(all_rows)

    log(
        f"Loaded train={len(train_rows)} validation={len(validation_rows)} "
        f"for {args.speaker}; train B/P={sum(r['label']=='B' for r in train_rows)}/"
        f"{sum(r['label']=='P' for r in train_rows)}"
    )
    result_rows = []
    prediction_rows = []

    for window_ms in args.windows_ms:
        log(f"Window {window_ms} ms: MFCC/logmel")
        mfcc_features = []
        for audio, sample_rate, onset in audio_cache:
            segment = clip_onset(audio, sample_rate, onset, window_ms)
            mfcc_features.append(extract_feature_set(segment, sample_rate, "mfcc_logmel_onset"))
        mfcc_x = np.vstack(mfcc_features)
        mfcc_result = evaluate_mfcc_cross_session(mfcc_x, labels, train_idx, val_idx, args.seed)
        result_rows.append(
            {
                "feature": "mfcc_logmel_onset",
                "window_ms": window_ms,
                "train_manifest": str(args.train_manifest),
                "validation_manifest": str(args.validation_manifest),
                "validation_accuracy": round(float(mfcc_result["accuracy"]), 4),
                "detail": f"best_l2={mfcc_result['best_l2']}",
            }
        )
        for local_i, original_i in enumerate(val_idx):
            prediction_rows.append(
                {
                    "feature": "mfcc_logmel_onset",
                    "window_ms": window_ms,
                    "word": all_rows[int(original_i)]["word"],
                    "true_label": all_rows[int(original_i)]["label"],
                    "predicted_label": "P" if int(mfcc_result["pred"][local_i]) == 1 else "B",
                    "confidence": round(float(mfcc_result["confidence"][local_i]), 4),
                    "audio_path": all_rows[int(original_i)]["audio_path_abs"],
                }
            )

        log(f"Window {window_ms} ms: HuBERT weighted layers")
        hubert_x, layer_ids = extract_hubert_features(all_rows, audio_cache, window_ms, args)
        hubert_result = train_weighted_layer_model(
            hubert_x[train_idx],
            labels[train_idx],
            hubert_x[val_idx],
            labels[val_idx],
            args,
            seed=args.seed,
        )
        result_rows.append(
            {
                "feature": "hubert_base_weighted_layers",
                "window_ms": window_ms,
                "train_manifest": str(args.train_manifest),
                "validation_manifest": str(args.validation_manifest),
                "validation_accuracy": round(float(hubert_result["accuracy"]), 4),
                "detail": top_layer_summary(hubert_result["layer_weights"], layer_ids),
            }
        )
        for local_i, original_i in enumerate(val_idx):
            prediction_rows.append(
                {
                    "feature": "hubert_base_weighted_layers",
                    "window_ms": window_ms,
                    "word": all_rows[int(original_i)]["word"],
                    "true_label": all_rows[int(original_i)]["label"],
                    "predicted_label": "P" if int(hubert_result["pred"][local_i]) == 1 else "B",
                    "confidence": round(float(hubert_result["confidence"][local_i]), 4),
                    "audio_path": all_rows[int(original_i)]["audio_path_abs"],
                }
            )

    summary_csv = args.outdir / "dave_train_validation_feature_comparison.csv"
    predictions_csv = args.outdir / "dave_train_validation_predictions.csv"
    write_csv(summary_csv, result_rows)
    write_csv(predictions_csv, prediction_rows)
    summary = {
        "speaker": args.speaker,
        "train_manifest": str(resolve_path(args.train_manifest)),
        "validation_manifest": str(resolve_path(args.validation_manifest)),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "windows_ms": args.windows_ms,
        "results": result_rows,
        "summary_csv": str(summary_csv),
        "predictions_csv": str(predictions_csv),
        "interpretation": "These are real cross-recording validation scores: train on Dave take 1, validate on Dave take 2.",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
