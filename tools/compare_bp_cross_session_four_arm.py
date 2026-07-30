from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import extract_feature_set

from compare_hubert_mfcc_cross_session import (
    clip_onset,
    find_energy_onset,
    fit_logistic_newton,
    predict_prob,
    standardize,
    tune_l2_on_train,
)


DEFAULT_ARMS = [
    "mfcc_logmel_onset",
    "hubert_mid_meanstd",
    "hubert_mid_attention",
    "hybrid_attention",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fair cross-session B/P comparison for MFCC/log-mel, HuBERT middle-layer mean/std, "
            "HuBERT temporal attention, and HuBERT+explicit-onset hybrid features."
        )
    )
    parser.add_argument("--train-manifest", type=Path, default=Path("data/metadata/dave_bp_consecutive_v2_manifest.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/metadata/dave_bp_drb_validation_v2_manifest.csv"))
    parser.add_argument("--speaker", default="dave")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/per_patient_bp_eval/bp_four_arm_cross_session"))
    parser.add_argument("--windows-ms", type=int, nargs="*", default=[150, 200, 220])
    parser.add_argument("--arms", nargs="*", default=DEFAULT_ARMS, choices=DEFAULT_ARMS)
    parser.add_argument("--explicit-feature-set", default="bp_onset")
    parser.add_argument("--model", default="facebook/hubert-base-ls960")
    parser.add_argument("--hubert-layers", type=int, nargs="*", default=[4, 5, 6, 7, 8, 9])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="*", default=[13, 29, 42, 71, 97])
    parser.add_argument("--weighted-epochs", type=int, default=260)
    parser.add_argument("--attention-epochs", type=int, default=180)
    parser.add_argument("--weighted-learning-rate", type=float, default=0.03)
    parser.add_argument("--attention-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--attention-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


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


def labels_for_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([1 if row["label"].upper() == "P" else 0 for row in rows], dtype=np.int64)


@dataclass
class AudioItem:
    audio: np.ndarray
    sample_rate: int
    onset_sample: int
    duration_sec: float
    peak: float
    rms_dbfs: float


def load_audio_items(rows: list[dict[str, str]]) -> list[AudioItem]:
    items = []
    for row in rows:
        sample_rate, audio = read_wav_float(Path(row["audio_path_abs"]))
        audio = audio.astype(np.float32)
        onset = find_energy_onset(audio, sample_rate)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)) if len(audio) else 0.0
        items.append(
            AudioItem(
                audio=audio,
                sample_rate=sample_rate,
                onset_sample=onset,
                duration_sec=len(audio) / sample_rate if sample_rate else 0.0,
                peak=peak,
                rms_dbfs=20.0 * np.log10(rms + 1e-12),
            )
        )
    return items


def write_alignment_qc(path: Path, rows: list[dict[str, str]], audio_items: list[AudioItem]) -> None:
    qc_rows = []
    for row, item in zip(rows, audio_items):
        qc_rows.append(
            {
                "split": row["split"],
                "label": row["label"],
                "word": row.get("word", ""),
                "audio_path": row["audio_path_abs"],
                "duration_sec": round(item.duration_sec, 4),
                "onset_sec": round(item.onset_sample / item.sample_rate, 4),
                "peak": round(item.peak, 6),
                "rms_dbfs": round(item.rms_dbfs, 2),
            }
        )
    write_csv(path, qc_rows)


def device_for(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(args.device)


def binary_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"accuracy": float(np.mean(pred == y_true))}
    for label_id, label_name in [(0, "B"), (1, "P")]:
        tp = float(np.sum((pred == label_id) & (y_true == label_id)))
        fp = float(np.sum((pred == label_id) & (y_true != label_id)))
        fn = float(np.sum((pred != label_id) & (y_true == label_id)))
        out[f"{label_name}_precision"] = tp / (tp + fp) if tp + fp > 0 else 0.0
        out[f"{label_name}_recall"] = tp / (tp + fn) if tp + fn > 0 else 0.0
    return out


def evaluate_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> dict[str, object]:
    best_l2 = tune_l2_on_train(features[train_idx], labels[train_idx], seed)
    train_x, val_x = standardize(features[train_idx], features[val_idx])
    weights = fit_logistic_newton(train_x, labels[train_idx].astype(np.float64), l2=best_l2)
    prob = predict_prob(weights, val_x)
    pred = (prob >= 0.5).astype(np.int64)
    confidence = np.maximum(prob, 1.0 - prob)
    return {
        **binary_metrics(labels[val_idx], pred),
        "pred": pred,
        "confidence": confidence,
        "detail": f"best_l2={best_l2}",
    }


def normalize_layer_features(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32)


class WeightedLayerMeanStdClassifier(nn.Module):
    def __init__(self, layer_count: int, feature_dim: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        pooled = torch.sum(x * layer_weights[None, :, None], dim=1)
        return self.classifier(pooled)


def evaluate_weighted_meanstd(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = device_for(args)
    train_x, val_x = normalize_layer_features(features[train_idx], features[val_idx])
    train_y = labels[train_idx]
    model = WeightedLayerMeanStdClassifier(train_x.shape[1], train_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.weighted_learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long)),
        batch_size=min(args.batch_size, len(train_x)),
        shuffle=True,
    )

    for _ in range(args.weighted_epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(val_x, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int64)
        layer_weights = torch.softmax(model.layer_logits, dim=0).detach().cpu().numpy()
    return {
        **binary_metrics(labels[val_idx], pred),
        "pred": pred,
        "confidence": np.max(probs, axis=1),
        "layer_weights": layer_weights,
    }


def normalize_temporal(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = train_mask[:, None, :, None].astype(np.float32)
    denom = np.maximum(valid.sum(axis=(0, 2), keepdims=True), 1.0)
    mean = (train_x * valid).sum(axis=(0, 2), keepdims=True) / denom
    var = (((train_x - mean) * valid) ** 2).sum(axis=(0, 2), keepdims=True) / denom
    std = np.sqrt(np.maximum(var, 1e-6))
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32)


def normalize_plain(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32)


class TemporalAttentionClassifier(nn.Module):
    def __init__(
        self,
        layer_count: int,
        hidden_dim: int,
        explicit_dim: int,
        attention_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim + explicit_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor,
        explicit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_weights = torch.softmax(self.layer_logits, dim=0)
        frames = torch.sum(x * layer_weights[None, :, None, None], dim=1)
        scores = self.attention(frames).squeeze(-1)
        scores = scores.masked_fill(~frame_mask.bool(), -1.0e9)
        attention_weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(frames * attention_weights[:, :, None], dim=1)
        if explicit is not None:
            pooled = torch.cat([pooled, explicit], dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits, layer_weights, attention_weights


def evaluate_attention(
    temporal_features: np.ndarray,
    frame_mask: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    explicit_features: np.ndarray | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = device_for(args)
    train_x, val_x = normalize_temporal(
        temporal_features[train_idx],
        temporal_features[val_idx],
        frame_mask[train_idx],
        frame_mask[val_idx],
    )
    train_exp = None
    val_exp = None
    explicit_dim = 0
    if explicit_features is not None:
        train_exp, val_exp = normalize_plain(explicit_features[train_idx], explicit_features[val_idx])
        explicit_dim = train_exp.shape[1]

    train_y = labels[train_idx]
    model = TemporalAttentionClassifier(
        layer_count=train_x.shape[1],
        hidden_dim=train_x.shape[3],
        explicit_dim=explicit_dim,
        attention_hidden=args.attention_hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.attention_learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    tensors = [
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(frame_mask[train_idx], dtype=torch.bool),
        torch.tensor(train_y, dtype=torch.long),
    ]
    if train_exp is not None:
        tensors.append(torch.tensor(train_exp, dtype=torch.float32))
    loader = DataLoader(TensorDataset(*tensors), batch_size=min(args.batch_size, len(train_x)), shuffle=True)

    for _ in range(args.attention_epochs):
        model.train()
        for batch in loader:
            batch_x = batch[0].to(device)
            batch_mask = batch[1].to(device)
            batch_y = batch[2].to(device)
            batch_exp = batch[3].to(device) if train_exp is not None else None
            optimizer.zero_grad()
            logits, _, _ = model(batch_x, batch_mask, batch_exp)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_exp_tensor = torch.tensor(val_exp, dtype=torch.float32, device=device) if val_exp is not None else None
        logits, layer_weights, attention_weights = model(
            torch.tensor(val_x, dtype=torch.float32, device=device),
            torch.tensor(frame_mask[val_idx], dtype=torch.bool, device=device),
            val_exp_tensor,
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1).astype(np.int64)
        layer_weights_np = layer_weights.detach().cpu().numpy()
        attention_weights_np = attention_weights.detach().cpu().numpy()

    return {
        **binary_metrics(labels[val_idx], pred),
        "pred": pred,
        "confidence": np.max(probs, axis=1),
        "layer_weights": layer_weights_np,
        "attention_peak_frame_mean": float(np.mean(np.argmax(attention_weights_np, axis=1))),
    }


def hubert_temporal_cache_path(row: dict[str, str], window_ms: int, args: argparse.Namespace, cache_dir: Path) -> Path:
    audio_path = Path(row["audio_path_abs"])
    stat = audio_path.stat()
    payload = {
        "audio_path": str(audio_path.resolve()).lower(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "window_ms": window_ms,
        "model": args.model,
        "feature_type": "hidden_states_temporal_all_layers_v1",
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def extract_hubert_temporal_features(
    rows: list[dict[str, str]],
    audio_items: list[AudioItem],
    window_ms: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from transformers import AutoFeatureExtractor, HubertModel

    cache_dir = resolve_path(args.feature_cache_dir) if args.feature_cache_dir else args.outdir / "hubert_temporal_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_only = not args.allow_download
    device = device_for(args)
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=local_only)
    model = HubertModel.from_pretrained(args.model, local_files_only=local_only).to(device)
    model.eval()

    arrays: list[np.ndarray] = []
    layer_ids: list[int] = []
    for index, (row, item) in enumerate(zip(rows, audio_items), start=1):
        path = hubert_temporal_cache_path(row, window_ms, args, cache_dir)
        if path.exists() and not args.force_extract:
            cached = np.load(path)
            arrays.append(cached["features"].astype(np.float32))
            layer_ids = cached["layers"].astype(int).tolist()
            continue

        segment = clip_onset(item.audio, item.sample_rate, item.onset_sample, window_ms)
        inputs = extractor(segment, sampling_rate=item.sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)

        layer_arrays = []
        ids = []
        for layer_idx, hidden in enumerate(output.hidden_states[1:], start=1):
            layer_arrays.append(hidden[0].detach().cpu().numpy().astype(np.float32))
            ids.append(layer_idx)
        arr = np.stack(layer_arrays).astype(np.float32)
        np.savez_compressed(path, features=arr, layers=np.array(ids, dtype=np.int32))
        arrays.append(arr)
        layer_ids = ids
        log(f"HuBERT temporal {window_ms}ms {index}/{len(rows)}: {row.get('word', '')} {row['label']}")

    max_frames = max(arr.shape[1] for arr in arrays)
    padded = np.zeros((len(arrays), arrays[0].shape[0], max_frames, arrays[0].shape[2]), dtype=np.float32)
    frame_mask = np.zeros((len(arrays), max_frames), dtype=bool)
    for row_idx, arr in enumerate(arrays):
        frames = arr.shape[1]
        padded[row_idx, :, :frames, :] = arr
        frame_mask[row_idx, :frames] = True
    return padded, frame_mask, layer_ids


def select_layers(
    temporal: np.ndarray,
    layer_ids: list[int],
    wanted_layers: list[int],
) -> tuple[np.ndarray, list[int]]:
    selected_indices = []
    selected_ids = []
    for layer in wanted_layers:
        if layer in layer_ids:
            selected_indices.append(layer_ids.index(layer))
            selected_ids.append(layer)
    if not selected_indices:
        raise ValueError(f"None of the requested HuBERT layers exist: {wanted_layers}; available={layer_ids}")
    return temporal[:, selected_indices, :, :], selected_ids


def temporal_to_meanstd(temporal: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    valid = frame_mask[:, None, :, None].astype(np.float32)
    denom = np.maximum(valid.sum(axis=2), 1.0)
    mean = (temporal * valid).sum(axis=2) / denom
    var = (((temporal - mean[:, :, None, :]) * valid) ** 2).sum(axis=2) / denom
    std = np.sqrt(np.maximum(var, 1e-6))
    return np.concatenate([mean, std], axis=2).astype(np.float32)


def top_layer_summary(layer_weights: np.ndarray, layer_ids: list[int], k: int = 4) -> str:
    order = np.argsort(layer_weights)[::-1][:k]
    return ";".join(f"L{layer_ids[idx]}={layer_weights[idx]:.3f}" for idx in order)


def add_seed_result(
    rows: list[dict[str, object]],
    arm: str,
    window_ms: int,
    seed: int,
    result: dict[str, object],
    detail: str,
) -> None:
    rows.append(
        {
            "arm": arm,
            "window_ms": window_ms,
            "seed": seed,
            "accuracy": round(float(result["accuracy"]), 4),
            "B_precision": round(float(result["B_precision"]), 4),
            "B_recall": round(float(result["B_recall"]), 4),
            "P_precision": round(float(result["P_precision"]), 4),
            "P_recall": round(float(result["P_recall"]), 4),
            "detail": detail,
        }
    )


def add_prediction_rows(
    rows: list[dict[str, object]],
    arm: str,
    window_ms: int,
    seed: int,
    validation_rows: list[dict[str, str]],
    result: dict[str, object],
) -> None:
    pred = result["pred"]
    confidence = result["confidence"]
    for idx, row in enumerate(validation_rows):
        rows.append(
            {
                "arm": arm,
                "window_ms": window_ms,
                "seed": seed,
                "word": row.get("word", ""),
                "true_label": row["label"],
                "predicted_label": "P" if int(pred[idx]) == 1 else "B",
                "confidence": round(float(confidence[idx]), 4),
                "audio_path": row["audio_path_abs"],
            }
        )


def summarize_seed_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["arm"]), int(row["window_ms"])), []).append(row)

    summary = []
    metric_names = ["accuracy", "B_precision", "B_recall", "P_precision", "P_recall"]
    for (arm, window_ms), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        output: dict[str, object] = {
            "arm": arm,
            "window_ms": window_ms,
            "seed_count": len(rows),
        }
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            output[f"{metric}_mean"] = round(float(values.mean()), 4)
            output[f"{metric}_std"] = round(float(values.std(ddof=0)), 4)
        output["details"] = " | ".join(sorted(set(str(row.get("detail", "")) for row in rows if row.get("detail"))))
        summary.append(output)
    return summary


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
    train_idx = np.arange(len(train_rows), dtype=np.int64)
    val_idx = np.arange(len(train_rows), len(all_rows), dtype=np.int64)
    audio_items = load_audio_items(all_rows)
    write_alignment_qc(args.outdir / "onset_alignment_qc.csv", all_rows, audio_items)

    log(
        f"Loaded train={len(train_rows)} validation={len(validation_rows)} speaker={args.speaker}; "
        f"arms={','.join(args.arms)}"
    )

    seed_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    need_hubert = any(arm.startswith("hubert") or arm.startswith("hybrid") for arm in args.arms)

    for window_ms in args.windows_ms:
        log(f"Window {window_ms} ms")

        if "mfcc_logmel_onset" in args.arms or "hybrid_attention" in args.arms:
            explicit_features = []
            mfcc_features = []
            for item in audio_items:
                segment = clip_onset(item.audio, item.sample_rate, item.onset_sample, window_ms)
                explicit_features.append(extract_feature_set(segment, item.sample_rate, args.explicit_feature_set))
                mfcc_features.append(extract_feature_set(segment, item.sample_rate, "mfcc_logmel_onset"))
            explicit_x = np.vstack(explicit_features).astype(np.float32)
            mfcc_x = np.vstack(mfcc_features).astype(np.float32)
        else:
            explicit_x = None
            mfcc_x = None

        if "mfcc_logmel_onset" in args.arms:
            for seed in args.seeds:
                result = evaluate_logistic(mfcc_x, labels, train_idx, val_idx, seed)
                add_seed_result(seed_rows, "mfcc_logmel_onset", window_ms, seed, result, str(result["detail"]))
                add_prediction_rows(prediction_rows, "mfcc_logmel_onset", window_ms, seed, validation_rows, result)

        if need_hubert:
            temporal_all, frame_mask, all_layer_ids = extract_hubert_temporal_features(
                all_rows, audio_items, window_ms, args
            )
            temporal, selected_layer_ids = select_layers(temporal_all, all_layer_ids, args.hubert_layers)
            meanstd_x = temporal_to_meanstd(temporal, frame_mask)
        else:
            temporal = None
            frame_mask = None
            meanstd_x = None
            selected_layer_ids = []

        if "hubert_mid_meanstd" in args.arms:
            for seed in args.seeds:
                result = evaluate_weighted_meanstd(meanstd_x, labels, train_idx, val_idx, args, seed)
                detail = top_layer_summary(result["layer_weights"], selected_layer_ids)
                add_seed_result(seed_rows, "hubert_mid_meanstd", window_ms, seed, result, detail)
                add_prediction_rows(prediction_rows, "hubert_mid_meanstd", window_ms, seed, validation_rows, result)

        if "hubert_mid_attention" in args.arms:
            for seed in args.seeds:
                result = evaluate_attention(temporal, frame_mask, labels, train_idx, val_idx, args, seed)
                detail = (
                    f"{top_layer_summary(result['layer_weights'], selected_layer_ids)};"
                    f"attention_peak_frame_mean={result['attention_peak_frame_mean']:.2f}"
                )
                add_seed_result(seed_rows, "hubert_mid_attention", window_ms, seed, result, detail)
                add_prediction_rows(prediction_rows, "hubert_mid_attention", window_ms, seed, validation_rows, result)

        if "hybrid_attention" in args.arms:
            for seed in args.seeds:
                result = evaluate_attention(
                    temporal,
                    frame_mask,
                    labels,
                    train_idx,
                    val_idx,
                    args,
                    seed,
                    explicit_features=explicit_x,
                )
                detail = (
                    f"{top_layer_summary(result['layer_weights'], selected_layer_ids)};"
                    f"explicit={args.explicit_feature_set};"
                    f"attention_peak_frame_mean={result['attention_peak_frame_mean']:.2f}"
                )
                add_seed_result(seed_rows, "hybrid_attention", window_ms, seed, result, detail)
                add_prediction_rows(prediction_rows, "hybrid_attention", window_ms, seed, validation_rows, result)

    summary_rows = summarize_seed_rows(seed_rows)
    seed_csv = args.outdir / "four_arm_by_seed.csv"
    summary_csv = args.outdir / "four_arm_summary.csv"
    predictions_csv = args.outdir / "four_arm_predictions.csv"
    write_csv(seed_csv, seed_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(predictions_csv, prediction_rows)

    summary = {
        "speaker": args.speaker,
        "train_manifest": str(resolve_path(args.train_manifest)),
        "validation_manifest": str(resolve_path(args.validation_manifest)),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "windows_ms": args.windows_ms,
        "arms": args.arms,
        "hubert_model": args.model,
        "hubert_layers": args.hubert_layers,
        "explicit_feature_set": args.explicit_feature_set,
        "seeds": args.seeds,
        "outputs": {
            "seed_csv": str(seed_csv),
            "summary_csv": str(summary_csv),
            "predictions_csv": str(predictions_csv),
            "onset_alignment_qc": str(args.outdir / "onset_alignment_qc.csv"),
        },
        "statistical_note": (
            "With about 100 validation clips, treat accuracy gaps below roughly 0.07-0.09 as inconclusive. "
            "Use mean/std across seeds and class precision/recall, not a single best seed."
        ),
        "decision_note": (
            "If all arms remain weak on cross-session validation, prioritize session normalization, alignment QA, "
            "and more sessions before trying larger HuBERT models."
        ),
        "results": summary_rows,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Saved by-seed results: {seed_csv}")
    log(f"Saved summary: {summary_csv}")
    log(f"Saved predictions: {predictions_csv}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
