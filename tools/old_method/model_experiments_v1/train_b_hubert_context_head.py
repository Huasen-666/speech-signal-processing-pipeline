import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import resample_linear


DEFAULT_MODEL = "facebook/hubert-base-ls960"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small B-vowel/context classifier on frozen HuBERT features. "
            "Features are cached per file so interrupted runs can resume safely."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/ml_baseline/b_hubert_context_head"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads if model is not cached.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set >0 to limit CPU threads.")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=900.0, help="Use only the first active N ms of each word.")
    parser.add_argument("--onset-ms", type=float, default=260.0, help="Extra HuBERT stats over the initial cue window.")
    parser.add_argument("--trim-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-leading-silence", dest="trim_leading_silence", action="store_false")
    parser.add_argument("--leading-preroll-ms", type=float, default=25.0)
    parser.add_argument("--label-column", default="b_subtype", help="Usually b_subtype, e.g. B_IH/B_EY/B_L.")
    parser.add_argument(
        "--label-map",
        choices=["raw", "b5_product", "p_product"],
        default="raw",
        help="Map fine labels into product-control groups before training.",
    )
    parser.add_argument("--include-speakers", nargs="*", default=[])
    parser.add_argument("--include-labels", nargs="*", default=[])
    parser.add_argument("--include-missing-stubs", action="store_true")
    parser.add_argument("--min-samples-per-class", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all selected rows; useful for smoke tests.")
    parser.add_argument("--test-speaker", default="")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Fail if any selected feature is missing from cache.")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def str_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self.handle.close()

    def __call__(self, message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped, flush=True)
        self.handle.write(stamped + "\n")
        self.handle.flush()


@dataclass(frozen=True)
class SelectedRow:
    row: dict[str, str]
    label: str
    cache_path: Path


def cache_key(row: dict[str, str], args: argparse.Namespace) -> str:
    audio_path = Path(row["audio_path"])
    stat = audio_path.stat()
    payload = {
        "audio_path": str(audio_path.resolve()).lower(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "model": args.model,
        "sample_rate": args.target_sample_rate,
        "window_ms": args.window_ms,
        "onset_ms": args.onset_ms,
        "trim_leading_silence": args.trim_leading_silence,
        "leading_preroll_ms": args.leading_preroll_ms,
        "label_column": args.label_column,
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def map_label(raw_label: str, args: argparse.Namespace) -> str:
    label = raw_label.upper()
    if args.label_map == "raw":
        return label
    if args.label_map == "b5_product":
        mapping = {
            "B_AE": "B_FRONT_LOW",
            "B_EH": "B_FRONT_MID",
            "B_EY": "B_FRONT_MID",
            "B_IH": "B_FRONT_HIGH",
            "B_IY": "B_FRONT_HIGH",
            "B_AH": "B_CENTRAL_BACK",
            "B_OW": "B_CENTRAL_BACK",
            "B_OY": "B_CENTRAL_BACK",
            "B_L": "B_L_CLUSTER",
        }
        return mapping.get(label, label)
    if args.label_map == "p_product":
        mapping = {
            "P_AE": "P_FRONT_LOW",
            "P_EH": "P_FRONT_MID",
            "P_EY": "P_FRONT_MID",
            "P_IH": "P_FRONT_HIGH",
            "P_IY": "P_FRONT_HIGH",
            "P_AA": "P_OPEN_BACK",
            "P_AH": "P_OPEN_BACK",
            "P_AY": "P_OPEN_BACK",
            "P_OW": "P_BACK_ROUNDED",
            "P_UH": "P_BACK_ROUNDED",
            "P_UW": "P_BACK_ROUNDED",
            "P_ER": "P_R_COLORED",
            "P_L": "P_L_CLUSTER",
            "P_R": "P_R_CLUSTER",
        }
        return mapping.get(label, label)
    raise ValueError(f"Unknown label map: {args.label_map}")


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace, cache_dir: Path) -> list[SelectedRow]:
    include_speakers = {item.lower() for item in args.include_speakers}
    include_labels = {item.upper() for item in args.include_labels}
    selected: list[SelectedRow] = []
    for row in rows:
        if not str_true(row.get("usable", "")):
            continue
        if not args.include_missing_stubs and not str_true(row.get("stub_available", "")):
            continue
        if include_speakers and row.get("speaker", "").lower() not in include_speakers:
            continue
        raw_label = row.get(args.label_column, "").upper()
        if not raw_label:
            continue
        label = map_label(raw_label, args)
        if include_labels and label not in include_labels:
            continue
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue
        selected.append(SelectedRow(row=row, label=label, cache_path=cache_dir / f"{cache_key(row, args)}.npz"))

    label_counts: dict[str, int] = {}
    for item in selected:
        label_counts[item.label] = label_counts.get(item.label, 0) + 1
    allowed = {label for label, count in label_counts.items() if count >= args.min_samples_per_class}
    filtered = [item for item in selected if item.label in allowed]
    filtered.sort(key=lambda item: (item.row.get("speaker", ""), item.row.get("protocol_index", ""), item.row.get("word", "")))
    if args.max_rows > 0:
        filtered = filtered[: args.max_rows]
    return filtered


def trim_leading_silence(audio: np.ndarray, sample_rate: int, preroll_ms: float) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return audio

    rms = []
    starts = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        rms.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
        starts.append(start)
    if not rms:
        return audio

    frame_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    threshold = max(float(np.percentile(frame_db, 10)) + 12.0, float(np.percentile(frame_db, 95)) - 35.0, -60.0)
    active = np.flatnonzero(frame_db >= threshold)
    if len(active) == 0:
        return audio

    preroll = round(sample_rate * preroll_ms / 1000.0)
    start = max(0, starts[int(active[0])] - preroll)
    return audio[start:]


def load_word_audio(path: Path, args: argparse.Namespace) -> np.ndarray:
    sample_rate, audio = read_wav_float(path)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != args.target_sample_rate:
        audio = resample_linear(audio, sample_rate, args.target_sample_rate).astype(np.float32)
    if args.trim_leading_silence:
        audio = trim_leading_silence(audio, args.target_sample_rate, args.leading_preroll_ms)
    max_len = max(1, round(args.target_sample_rate * args.window_ms / 1000.0))
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def load_hubert_bundle(args: argparse.Namespace, device: torch.device):
    try:
        from transformers import AutoFeatureExtractor, HubertModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing transformers. Activate the win_ai environment or install transformers before running this script."
        ) from exc

    local_only = not args.allow_download
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=local_only)
    model = HubertModel.from_pretrained(args.model, local_files_only=local_only).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return feature_extractor, model


def hubert_pool(hidden: torch.Tensor, sample_rate: int, args: argparse.Namespace, model) -> np.ndarray:
    frames = hidden[0].detach().cpu().numpy().astype(np.float32)
    full_mean = frames.mean(axis=0)
    full_std = frames.std(axis=0)

    stride = float(getattr(model.config, "inputs_to_logits_ratio", 320))
    frame_sec = stride / sample_rate
    onset_frames = max(1, min(len(frames), round((args.onset_ms / 1000.0) / frame_sec)))
    onset = frames[:onset_frames]
    onset_mean = onset.mean(axis=0)
    onset_std = onset.std(axis=0)
    return np.concatenate([full_mean, full_std, onset_mean, onset_std]).astype(np.float32)


def extract_one_feature(item: SelectedRow, args: argparse.Namespace, feature_extractor, model, device: torch.device) -> dict[str, object]:
    audio = load_word_audio(Path(item.row["audio_path"]), args)
    inputs = feature_extractor(audio, sampling_rate=args.target_sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
    feature = hubert_pool(output.last_hidden_state, args.target_sample_rate, args, model)
    item.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        item.cache_path,
        feature=feature,
        label=item.label,
        audio_path=item.row["audio_path"],
        speaker=item.row.get("speaker", ""),
        word=item.row.get("word", ""),
        model=args.model,
    )
    return {
        "audio_path": item.row["audio_path"],
        "speaker": item.row.get("speaker", ""),
        "word": item.row.get("word", ""),
        "label": item.label,
        "cache_path": str(item.cache_path),
        "feature_dim": int(feature.shape[0]),
    }


def extract_features(items: list[SelectedRow], args: argparse.Namespace, logger: RunLogger) -> list[dict[str, object]]:
    if args.train_only:
        missing = [item for item in items if not item.cache_path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cached feature files are missing. Remove --train-only first.")
        return []

    device = choose_device(args.device)
    logger(f"HuBERT feature extraction device: {device}")
    feature_extractor, model = load_hubert_bundle(args, device)
    manifest_rows = []
    start_all = time.perf_counter()
    for idx, item in enumerate(items, start=1):
        if item.cache_path.exists() and not args.force_extract:
            logger(f"feature {idx}/{len(items)} cached: {item.row.get('speaker')} {item.row.get('word')} {item.label}")
            continue
        start = time.perf_counter()
        row = extract_one_feature(item, args, feature_extractor, model, device)
        elapsed = time.perf_counter() - start
        logger(f"feature {idx}/{len(items)} extracted in {elapsed:.2f}s: {row['speaker']} {row['word']} {row['label']}")
        manifest_rows.append(row)
    logger(f"Feature extraction finished in {time.perf_counter() - start_all:.2f}s")
    return manifest_rows


def load_cached_matrix(items: list[SelectedRow]) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
    features = []
    labels = []
    metadata = []
    for item in items:
        if not item.cache_path.exists():
            raise FileNotFoundError(item.cache_path)
        data = np.load(item.cache_path, allow_pickle=False)
        features.append(data["feature"].astype(np.float32))
        labels.append(item.label)
        metadata.append(item.row)
    return np.stack(features), labels, metadata


def label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def stratified_split(labels: np.ndarray, val_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    val_parts = []
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        shuffled = rng.permutation(indices)
        if len(shuffled) <= 1:
            train_parts.append(shuffled)
            continue
        n_val = max(1, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 1)
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])
    train_idx = rng.permutation(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    val_idx = rng.permutation(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    return train_idx, val_idx


def split_indices(rows: list[dict[str, str]], labels: np.ndarray, args: argparse.Namespace, rng: np.random.Generator):
    all_idx = np.arange(len(rows))
    if not args.test_speaker:
        train_idx, val_idx = stratified_split(labels, args.val_ratio, rng)
        return train_idx, val_idx, np.array([], dtype=int)

    test_speaker = args.test_speaker.lower()
    test_idx = np.array([idx for idx, row in enumerate(rows) if row.get("speaker", "").lower() == test_speaker], dtype=int)
    if len(test_idx) == 0:
        raise ValueError(f"No rows selected for test speaker: {args.test_speaker}")
    test_set = set(test_idx.tolist())
    remaining_idx = np.array([idx for idx in all_idx if idx not in test_set], dtype=int)
    local_train, local_val = stratified_split(labels[remaining_idx], args.val_ratio, rng)
    return remaining_idx[local_train], remaining_idx[local_val], test_idx


class ContextHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, class_count: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_train(train_x: np.ndarray, *others: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized_train = (train_x - mean) / std
    normalized_others = [(item - mean) / std for item in others]
    return normalized_train.astype(np.float32), [item.astype(np.float32) for item in normalized_others], mean, std


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int):
    if len(features) == 0:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=np.float32)
    model.eval()
    preds = []
    probs = []
    loader = make_loader(features, np.zeros(len(features), dtype=int), batch_size, shuffle=False)
    with torch.inference_mode():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            batch_probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(batch_probs)
            preds.extend(batch_probs.argmax(axis=1).tolist())
    return np.array(preds, dtype=int), np.vstack(probs)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        matrix[int(actual), int(pred)] += 1
    return matrix


def evaluate(model: nn.Module, features: np.ndarray, labels: np.ndarray, class_names: list[str], device: torch.device, batch_size: int):
    if len(features) == 0:
        return {"count": 0}
    preds, probs = predict(model, features, device, batch_size)
    matrix = confusion_matrix(labels, preds, len(class_names))
    per_class = {}
    for idx, name in enumerate(class_names):
        total = int(matrix[idx].sum())
        correct = int(matrix[idx, idx])
        per_class[name] = {"count": total, "accuracy": round(correct / total, 4) if total else None}
    return {
        "count": int(len(labels)),
        "accuracy": round(float(np.mean(preds == labels)), 4),
        "mean_confidence": round(float(np.mean(np.max(probs, axis=1))), 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def class_weights(labels: np.ndarray, class_count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (class_count * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_head(
    features: np.ndarray,
    label_names: list[str],
    metadata: list[dict[str, str]],
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    class_names = sorted(set(label_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([class_to_idx[name] for name in label_names], dtype=int)
    train_idx, val_idx, test_idx = split_indices(metadata, labels, args, rng)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Train/validation split is empty. Add data or lower filtering constraints.")

    train_x, normalized, mean, std = normalize_train(features[train_idx], features[val_idx], features[test_idx])
    val_x = normalized[0]
    test_x = normalized[1]
    train_y = labels[train_idx]
    val_y = labels[val_idx]
    test_y = labels[test_idx] if len(test_idx) else np.array([], dtype=int)

    model = ContextHead(features.shape[1], args.hidden_size, len(class_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, len(class_names)).to(device))
    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_x, val_y, class_names, device, args.batch_size)
        val_acc = float(val_metrics["accuracy"])
        train_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            logger(f"epoch={epoch:04d} loss={train_loss:.4f} val_accuracy={val_acc:.3f} best={best_val:.3f}")
        if bad_epochs >= args.patience:
            logger(f"early stop at epoch={epoch}; best_val_accuracy={best_val:.3f}")
            break

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate(model, train_x, train_y, class_names, device, args.batch_size),
        "validation": evaluate(model, val_x, val_y, class_names, device, args.batch_size),
        "test": evaluate(model, test_x, test_y, class_names, device, args.batch_size),
    }

    pred_idx, pred_probs = predict(model, np.concatenate([train_x, val_x, test_x], axis=0), device, args.batch_size)
    ordered_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
    split_names = ["train"] * len(train_idx) + ["validation"] * len(val_idx) + ["test"] * len(test_idx)
    prediction_rows = []
    for local_i, original_i in enumerate(ordered_idx):
        row = metadata[int(original_i)]
        prediction_rows.append(
            {
                "split": split_names[local_i],
                "speaker": row.get("speaker", ""),
                "word": row.get("word", ""),
                "true_label": label_names[int(original_i)],
                "predicted_label": class_names[int(pred_idx[local_i])],
                "confidence": round(float(np.max(pred_probs[local_i])), 4),
                "audio_path": row.get("audio_path", ""),
            }
        )

    return {
        "model_state": best_state,
        "normalization_mean": mean.squeeze(0).astype(np.float32),
        "normalization_std": std.squeeze(0).astype(np.float32),
        "class_names": class_names,
        "history": history,
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "sample_counts": {
            "selected_total": int(len(labels)),
            "selected_by_label": label_counts(label_names),
            "train": label_counts([label_names[int(idx)] for idx in train_idx]),
            "validation": label_counts([label_names[int(idx)] for idx in val_idx]),
            "test": label_counts([label_names[int(idx)] for idx in test_idx]),
        },
        "architecture": {
            "feature_source": args.model,
            "feature_dim": int(features.shape[1]),
            "pooling": "full_mean, full_std, onset_mean, onset_std",
            "head": "Linear -> LayerNorm -> GELU -> Dropout -> Linear",
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
        },
    }


def write_training_outputs(result: dict[str, object], args: argparse.Namespace, logger: RunLogger) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(result["model_state"], args.outdir / "model_state.pt")
    np.savez_compressed(
        args.outdir / "normalization_stats.npz",
        mean=result["normalization_mean"],
        std=result["normalization_std"],
        class_names=np.array(result["class_names"]),
    )
    write_csv(args.outdir / "training_history.csv", result["history"], ["epoch", "train_loss", "val_accuracy"])
    write_csv(
        args.outdir / "predictions.csv",
        result["prediction_rows"],
        ["split", "speaker", "word", "true_label", "predicted_label", "confidence", "audio_path"],
    )
    report = {
        "script": str(Path(__file__).resolve()),
        "manifest": str(args.manifest),
        "label_column": args.label_column,
        "label_map": args.label_map,
        "include_speakers": args.include_speakers or "all",
        "include_labels": args.include_labels or "all",
        "include_missing_stubs": args.include_missing_stubs,
        "test_speaker": args.test_speaker or "",
        "target_sample_rate": args.target_sample_rate,
        "window_ms": args.window_ms,
        "onset_ms": args.onset_ms,
        "trim_leading_silence": args.trim_leading_silence,
        "sample_counts": result["sample_counts"],
        "architecture": result["architecture"],
        "metrics": result["metrics"],
    }
    (args.outdir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger(f"Saved model: {args.outdir / 'model_state.pt'}")
    logger(f"Saved report: {args.outdir / 'training_report.json'}")
    logger(f"Saved predictions: {args.outdir / 'predictions.csv'}")


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    cache_dir = args.feature_cache_dir or (args.outdir / "feature_cache")
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.outdir / "run.log")
    try:
        logger(f"Starting clean HuBERT context-head training script")
        logger(f"Model: {args.model}")
        logger(f"Output: {args.outdir}")
        rows = read_csv(args.manifest)
        items = select_rows(rows, args, cache_dir)
        if len(items) < 4:
            raise ValueError("Not enough selected rows. Check usable/stub filters or add more data.")
        logger(f"Selected rows: {len(items)}")
        logger(f"Selected label counts: {label_counts([item.label for item in items])}")
        extracted_rows = extract_features(items, args, logger)
        if extracted_rows:
            write_csv(
                args.outdir / "newly_extracted_features.csv",
                extracted_rows,
                ["audio_path", "speaker", "word", "label", "cache_path", "feature_dim"],
            )
        if args.extract_only:
            logger("Extract-only mode finished; no classifier was trained.")
            return
        features, label_names, metadata = load_cached_matrix(items)
        logger(f"Loaded cached feature matrix: shape={features.shape}")
        result = train_head(features, label_names, metadata, args, logger)
        write_training_outputs(result, args, logger)
        logger(f"Final metrics: {json.dumps(result['metrics'], indent=2)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
