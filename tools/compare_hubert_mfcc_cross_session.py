from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import extract_feature_set


WINDOWS_MS = [150, 200]
L2_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cheap MFCC/log-mel onset features against frozen HuBERT-base learnable layer weighting "
            "on cross-session B/P classification."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/david_bp_dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/per_patient_bp_eval/hubert_mfcc_cross_session"))
    parser.add_argument("--speaker", default="david")
    parser.add_argument("--model", default="facebook/hubert-base-ls960")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--windows-ms", type=int, nargs="*", default=WINDOWS_MS)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


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


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_rows(manifest: Path, speaker: str) -> list[dict[str, str]]:
    manifest = manifest if manifest.is_absolute() else PROJECT_ROOT / manifest
    rows = []
    for row in read_csv(manifest):
        if row.get("usable", "").lower() != "true":
            continue
        if row.get("speaker", "").lower() != speaker.lower():
            continue
        if row.get("label", "").upper() not in {"B", "P"}:
            continue
        audio_path = resolve_path(row["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        normalized = dict(row)
        normalized["audio_path_abs"] = str(audio_path)
        rows.append(normalized)
    rows.sort(key=lambda item: (item["label"], int(item.get("file_index") or item.get("protocol_index") or "0")))
    return rows


def take_for_row(row: dict[str, str], take_size: int = 50) -> int:
    file_index = int(row.get("file_index") or row.get("protocol_index") or "0")
    return (file_index - 1) // take_size + 1


def find_energy_onset(audio: np.ndarray, sample_rate: int, frame_ms: float = 8.0, hop_ms: float = 2.0) -> int:
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame:
        return 0
    starts = np.arange(0, len(audio) - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(audio[start : start + frame].astype(np.float64) ** 2) + 1e-12) for start in starts])
    db = 20.0 * np.log10(rms + 1e-12)
    threshold = max(float(np.percentile(db, 10)) + 12.0, float(np.percentile(db, 95)) - 30.0, -60.0)
    active = np.flatnonzero(db >= threshold)
    return int(starts[active[0]]) if len(active) else 0


def clip_onset(audio: np.ndarray, sample_rate: int, onset: int, window_ms: int) -> np.ndarray:
    n_samples = max(1, round(sample_rate * window_ms / 1000.0))
    segment = audio[onset : onset + n_samples]
    min_len = round(sample_rate * 30.0 / 1000.0)
    if len(segment) < min_len:
        segment = np.pad(segment, (0, min_len - len(segment)))
    return segment.astype(np.float32)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-8
    return (train_x - mean) / std, (test_x - mean) / std


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


def predict_prob(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    logits = np.clip(x_aug @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def stratified_folds(y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.zeros(len(y), dtype=int)
    for label in sorted(set(y.tolist())):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        for index, row_idx in enumerate(indices):
            fold_ids[row_idx] = index % folds
    return fold_ids


def tune_l2_on_train(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    folds = min(5, max(2, int(np.min(np.bincount(y)))))
    fold_ids = stratified_folds(y, folds, seed)
    best_l2 = L2_GRID[0]
    best_acc = -1.0
    for l2 in L2_GRID:
        pred = np.zeros(len(y), dtype=int)
        for fold in sorted(set(fold_ids.tolist())):
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            train_x, val_x = standardize(x[train_mask], x[val_mask])
            weights = fit_logistic_newton(train_x, y[train_mask].astype(np.float64), l2=l2)
            pred[val_mask] = (predict_prob(weights, val_x) >= 0.5).astype(int)
        acc = float(np.mean(pred == y))
        if acc > best_acc:
            best_acc = acc
            best_l2 = l2
    return best_l2


def evaluate_mfcc_cross_session(features: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> dict:
    best_l2 = tune_l2_on_train(features[train_idx], y[train_idx], seed)
    train_x, test_x = standardize(features[train_idx], features[test_idx])
    weights = fit_logistic_newton(train_x, y[train_idx].astype(np.float64), l2=best_l2)
    prob = predict_prob(weights, test_x)
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(np.mean(pred == y[test_idx])),
        "best_l2": best_l2,
        "pred": pred,
        "confidence": np.maximum(prob, 1.0 - prob),
    }


class WeightedLayerClassifier(nn.Module):
    def __init__(self, layer_count: int, feature_dim: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0)
        pooled = torch.sum(x * weights[None, :, None], dim=1)
        return self.classifier(pooled)


def normalize_hubert(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def train_weighted_layer_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "cuda" else "cpu") if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    train_x, test_x = normalize_hubert(train_x, test_x)
    model = WeightedLayerClassifier(train_x.shape[1], train_x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(train_x, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

    for _ in range(args.epochs):
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
        logits = model(torch.tensor(test_x, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)
        weights = torch.softmax(model.layer_logits, dim=0).detach().cpu().numpy()
    return {
        "accuracy": float(np.mean(pred == test_y)),
        "pred": pred,
        "confidence": np.max(probs, axis=1),
        "layer_weights": weights,
    }


def hubert_cache_path(row: dict[str, str], window_ms: int, args: argparse.Namespace, cache_dir: Path) -> Path:
    audio_path = Path(row["audio_path_abs"])
    stat = audio_path.stat()
    payload = {
        "audio_path": str(audio_path.resolve()).lower(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "window_ms": window_ms,
        "model": args.model,
        "feature_type": "hidden_states_mean_std_all_layers_v1",
    }
    return cache_dir / f"{hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()}.npz"


def extract_hubert_features(
    rows: list[dict[str, str]],
    audio_cache: list[tuple[np.ndarray, int, int]],
    window_ms: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[int]]:
    from transformers import AutoFeatureExtractor, HubertModel

    cache_dir = args.feature_cache_dir or (args.outdir / "hubert_feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_only = not args.allow_download
    device = torch.device("cuda" if args.device == "cuda" else "cpu") if args.device != "auto" else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=local_only)
    model = HubertModel.from_pretrained(args.model, local_files_only=local_only).to(device)
    model.eval()

    features = []
    layers_used = []
    for index, (row, (audio, sample_rate, onset)) in enumerate(zip(rows, audio_cache), start=1):
        path = hubert_cache_path(row, window_ms, args, cache_dir)
        if path.exists() and not args.force_extract:
            data = np.load(path)
            features.append(data["features"].astype(np.float32))
            layers_used = data["layers"].astype(int).tolist()
            continue
        segment = clip_onset(audio, sample_rate, onset, window_ms)
        inputs = extractor(segment, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)
        layer_features = []
        layer_ids = []
        for layer_idx, hidden in enumerate(output.hidden_states[1:], start=1):
            frames = hidden[0].detach().cpu().numpy().astype(np.float32)
            layer_features.append(np.concatenate([frames.mean(axis=0), frames.std(axis=0)]))
            layer_ids.append(layer_idx)
        arr = np.stack(layer_features).astype(np.float32)
        np.savez_compressed(path, features=arr, layers=np.array(layer_ids, dtype=np.int32))
        features.append(arr)
        layers_used = layer_ids
        log(f"HuBERT {window_ms}ms feature {index}/{len(rows)} extracted: {row['word']} {row['label']}")
    return np.stack(features), layers_used


def top_layer_summary(layer_weights: np.ndarray, layer_ids: list[int], k: int = 4) -> str:
    order = np.argsort(layer_weights)[::-1][:k]
    return ";".join(f"L{layer_ids[idx]}={layer_weights[idx]:.3f}" for idx in order)


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest, args.speaker)
    if not rows:
        raise ValueError("No usable B/P rows found.")
    labels = np.array([1 if row["label"].upper() == "P" else 0 for row in rows], dtype=int)
    take_ids = np.array([take_for_row(row) for row in rows], dtype=int)
    take1_idx = np.where(take_ids == 1)[0]
    take2_idx = np.where(take_ids == 2)[0]
    if len(take1_idx) == 0 or len(take2_idx) == 0:
        raise ValueError("Cross-session evaluation requires two takes/sessions.")

    audio_cache = []
    for row in rows:
        sample_rate, audio = read_wav_float(Path(row["audio_path_abs"]))
        onset = find_energy_onset(audio, sample_rate)
        audio_cache.append((audio.astype(np.float32), sample_rate, onset))

    log(f"Loaded {len(rows)} clips for {args.speaker}: B={int(np.sum(labels == 0))}, P={int(np.sum(labels == 1))}")
    result_rows = []
    prediction_rows = []

    for window_ms in args.windows_ms:
        log(f"Evaluating window={window_ms}ms")
        mfcc_features = []
        for audio, sample_rate, onset in audio_cache:
            segment = clip_onset(audio, sample_rate, onset, window_ms)
            mfcc_features.append(extract_feature_set(segment, sample_rate, "mfcc_logmel_onset"))
        mfcc_x = np.vstack(mfcc_features)
        for split_name, train_idx, test_idx in [
            ("take1_to_take2", take1_idx, take2_idx),
            ("take2_to_take1", take2_idx, take1_idx),
        ]:
            result = evaluate_mfcc_cross_session(mfcc_x, labels, train_idx, test_idx, args.seed)
            result_rows.append(
                {
                    "feature": "mfcc_logmel_onset",
                    "window_ms": window_ms,
                    "split": split_name,
                    "accuracy": round(result["accuracy"], 4),
                    "detail": f"best_l2={result['best_l2']}",
                }
            )
            for local_i, original_i in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "feature": "mfcc_logmel_onset",
                        "window_ms": window_ms,
                        "split": split_name,
                        "word": rows[int(original_i)]["word"],
                        "true_label": rows[int(original_i)]["label"],
                        "predicted_label": "P" if int(result["pred"][local_i]) == 1 else "B",
                        "confidence": round(float(result["confidence"][local_i]), 4),
                        "audio_path": rows[int(original_i)]["audio_path_abs"],
                    }
                )

        hubert_x, layer_ids = extract_hubert_features(rows, audio_cache, window_ms, args)
        for split_name, train_idx, test_idx in [
            ("take1_to_take2", take1_idx, take2_idx),
            ("take2_to_take1", take2_idx, take1_idx),
        ]:
            result = train_weighted_layer_model(
                hubert_x[train_idx],
                labels[train_idx],
                hubert_x[test_idx],
                labels[test_idx],
                args,
                seed=args.seed + (11 if split_name == "take2_to_take1" else 0),
            )
            detail = top_layer_summary(result["layer_weights"], layer_ids)
            result_rows.append(
                {
                    "feature": "hubert_base_weighted_layers",
                    "window_ms": window_ms,
                    "split": split_name,
                    "accuracy": round(result["accuracy"], 4),
                    "detail": detail,
                }
            )
            for local_i, original_i in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "feature": "hubert_base_weighted_layers",
                        "window_ms": window_ms,
                        "split": split_name,
                        "word": rows[int(original_i)]["word"],
                        "true_label": rows[int(original_i)]["label"],
                        "predicted_label": "P" if int(result["pred"][local_i]) == 1 else "B",
                        "confidence": round(float(result["confidence"][local_i]), 4),
                        "audio_path": rows[int(original_i)]["audio_path_abs"],
                    }
                )

    summary_csv = args.outdir / "cross_session_feature_comparison.csv"
    predictions_csv = args.outdir / "cross_session_predictions.csv"
    write_csv(summary_csv, result_rows)
    write_csv(predictions_csv, prediction_rows)
    summary = {
        "manifest": str(args.manifest),
        "speaker": args.speaker,
        "model": args.model,
        "rows": len(rows),
        "windows_ms": args.windows_ms,
        "results": result_rows,
        "summary_csv": str(summary_csv),
        "predictions_csv": str(predictions_csv),
        "interpretation": (
            "Use take1_to_take2 and take2_to_take1 as the real cross-session signal. "
            "Do not choose a deployment model from word-grouped CV alone."
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Saved summary: {summary_csv}")
    log(f"Saved predictions: {predictions_csv}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
