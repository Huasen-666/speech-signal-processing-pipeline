import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import (
    mel_filterbank,
    preemphasis,
    resample_linear,
    trim_by_energy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny log-mel CNN for B/P classification.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/ml_baseline/bp_tinycnn_logmel"))
    parser.add_argument("--labels", nargs="+", default=["B", "P"])
    parser.add_argument("--include-speakers", nargs="*", help="Optional speaker filter, e.g. bascom corrick.")
    parser.add_argument("--test-speaker", help="Hold out one speaker as final test split.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--duration-ms", type=float, default=800.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-mfcc", type=int, default=0, help="Optional MFCC channels; 0 means log-mel only.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance-train", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    labels: list[str],
    include_speakers: list[str] | None,
) -> list[dict[str, str]]:
    allowed_labels = {label.upper() for label in labels}
    allowed_speakers = {speaker.lower() for speaker in include_speakers} if include_speakers else None
    selected = []
    for row in rows:
        if row["usable"].lower() != "true":
            continue
        if row["label"].upper() not in allowed_labels:
            continue
        if allowed_speakers and row["speaker"].lower() not in allowed_speakers:
            continue
        selected.append(row)
    return selected


def prepare_active_signal(audio: np.ndarray, sample_rate: int, target_sample_rate: int, duration_ms: float) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float64)
    if len(mono) == 0:
        return np.zeros(round(target_sample_rate * duration_ms / 1000.0), dtype=np.float64)
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-12:
        active = np.zeros(1, dtype=np.float64)
    else:
        active = mono / peak
        active = resample_linear(active, sample_rate, target_sample_rate)
        active = preemphasis(active)
        active = trim_by_energy(active, frame_size=round(target_sample_rate * 0.02), hop_size=round(target_sample_rate * 0.005))

    target_len = max(1, round(target_sample_rate * duration_ms / 1000.0))
    if len(active) < target_len:
        active = np.pad(active, (0, target_len - len(active)))
    else:
        active = active[:target_len]
    return active.astype(np.float64)


def frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    starts = np.arange(0, len(signal) - frame_len + 1, hop_len)
    return np.stack([signal[start : start + frame_len] for start in starts])


def logmel_image(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
    duration_ms: float,
    n_mels: int,
    n_mfcc: int,
) -> np.ndarray:
    active = prepare_active_signal(audio, sample_rate, target_sample_rate, duration_ms)
    frame_len = round(target_sample_rate * 0.025)
    hop_len = round(target_sample_rate * 0.010)
    fft_size = 512
    frames = frame_signal(active, frame_len, hop_len)
    windowed = frames * np.hamming(frame_len)[None, :]
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_size, axis=1)) ** 2
    filters = mel_filterbank(target_sample_rate, fft_size, n_mels)
    mel_energy = spectrum @ filters.T
    logmel = np.log(np.maximum(mel_energy, 1e-10)).T
    channels = [logmel]
    if n_mfcc > 0:
        mfcc = dct(logmel.T, type=2, axis=1, norm="ortho")[:, :n_mfcc].T
        if n_mfcc < n_mels:
            mfcc = np.pad(mfcc, ((0, n_mels - n_mfcc), (0, 0)))
        channels.append(mfcc[:n_mels])
    return np.stack(channels).astype(np.float32)


def load_images(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, str]]]:
    images = []
    valid_rows = []
    errors = []
    for row in rows:
        try:
            sample_rate, audio = read_wav_float(row["audio_path"])
            images.append(
                logmel_image(
                    audio,
                    sample_rate,
                    args.target_sample_rate,
                    args.duration_ms,
                    args.n_mels,
                    args.n_mfcc,
                )
            )
            valid_rows.append(row)
        except (OSError, ValueError, wave.Error) as exc:
            errors.append({"audio_path": row["audio_path"], "error": str(exc)})
    if errors:
        joined = "\n".join(f"{item['audio_path']}: {item['error']}" for item in errors[:5])
        raise RuntimeError(f"Failed to load {len(errors)} files. First errors:\n{joined}")
    return np.stack(images), valid_rows


def stratified_validation_indices(labels: np.ndarray, val_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    val_parts = []
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        shuffled = rng.permutation(indices)
        n_val = max(1, round(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 1)
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])
    return rng.permutation(np.concatenate(train_parts)), rng.permutation(np.concatenate(val_parts))


def split_indices(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
    test_speaker: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(rows))
    if not test_speaker:
        train_idx, val_idx = stratified_validation_indices(labels, val_ratio, rng)
        return train_idx, val_idx, np.array([], dtype=int)

    test_speaker = test_speaker.lower()
    test_idx = np.array([idx for idx, row in enumerate(rows) if row["speaker"].lower() == test_speaker], dtype=int)
    test_set = set(test_idx.tolist())
    remaining_idx = np.array([idx for idx in all_indices if idx not in test_set], dtype=int)
    if len(test_idx) == 0:
        raise ValueError(f"No usable rows found for test speaker: {test_speaker}")
    local_train, local_val = stratified_validation_indices(labels[remaining_idx], val_ratio, rng)
    return remaining_idx[local_train], remaining_idx[local_val], test_idx


def balance_training_indices(indices: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for label in sorted(set(labels[indices].tolist())):
        label_indices = indices[labels[indices] == label]
        parts.append(rng.permutation(label_indices))
    min_count = min(len(part) for part in parts)
    return rng.permutation(np.concatenate([part[:min_count] for part in parts]))


def normalize_images(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], float, float]:
    mean = float(train.mean())
    std = float(train.std()) if float(train.std()) > 1e-8 else 1.0
    norm_train = (train - mean) / std
    norm_others = [(item - mean) / std for item in others]
    return norm_train, norm_others, mean, std


class TinyLogMelCNN(nn.Module):
    def __init__(self, in_channels: int, class_count: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(48, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    if len(images) == 0:
        return np.array([], dtype=int)
    model.eval()
    preds = []
    loader = make_loader(images, np.zeros(len(images), dtype=int), batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.array(preds, dtype=int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        matrix[int(actual), int(pred)] += 1
    return matrix


def evaluate(model: nn.Module, images: np.ndarray, labels: np.ndarray, class_names: list[str], device: torch.device, batch_size: int) -> dict:
    if len(images) == 0:
        return {"count": 0}
    pred = predict(model, images, device, batch_size)
    matrix = confusion_matrix(labels, pred, len(class_names))
    per_class = {}
    for idx, name in enumerate(class_names):
        total = int(matrix[idx].sum())
        correct = int(matrix[idx, idx])
        per_class[name] = {"count": total, "accuracy": round(correct / total, 4) if total else None}
    return {
        "count": int(len(labels)),
        "accuracy": round(float(np.mean(pred == labels)), 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def class_counts(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        label = rows[int(idx)]["label"].upper()
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [label.upper() for label in args.labels]
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    rows = select_rows(read_manifest(args.manifest), class_names, args.include_speakers)
    if len(rows) < 4:
        raise ValueError("Not enough usable samples after filtering.")

    images, rows = load_images(rows, args)
    labels = np.array([class_to_idx[row["label"].upper()] for row in rows], dtype=int)
    train_idx, val_idx, test_idx = split_indices(rows, labels, args.val_ratio, rng, args.test_speaker)
    if not args.no_balance_train:
        train_idx = balance_training_indices(train_idx, labels, rng)

    train_x, normalized, image_mean, image_std = normalize_images(images[train_idx], images[val_idx], images[test_idx])
    val_x = normalized[0]
    test_x = normalized[1]
    train_y = labels[train_idx]
    val_y = labels[val_idx]
    test_y = labels[test_idx] if len(test_idx) else np.array([], dtype=int)

    model = TinyLogMelCNN(in_channels=train_x.shape[1], class_count=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = -1.0
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
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6), "val_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch == args.epochs or epoch % 20 == 0:
            print(f"epoch={epoch:04d} loss={np.mean(losses):.4f} val_accuracy={val_acc:.3f} best={best_val:.3f}")

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate(model, train_x, train_y, class_names, device, args.batch_size),
        "validation": evaluate(model, val_x, val_y, class_names, device, args.batch_size),
        "test": evaluate(model, test_x, test_y, class_names, device, args.batch_size),
    }

    report = {
        "model": "TinyLogMelCNN",
        "feature_set": "logmel_image" if args.n_mfcc == 0 else "logmel_mfcc_image",
        "architecture": {
            "input_shape": list(train_x.shape[1:]),
            "conv_channels": [16, 32, 48],
            "classes": class_names,
        },
        "manifest": str(args.manifest),
        "include_speakers": args.include_speakers or "all",
        "test_speaker": args.test_speaker,
        "balance_train": not args.no_balance_train,
        "image_mean": image_mean,
        "image_std": image_std,
        "target_sample_rate": args.target_sample_rate,
        "duration_ms": args.duration_ms,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "weight_decay": args.weight_decay,
        "sample_counts": {
            "selected_total": len(rows),
            "train": class_counts(rows, train_idx),
            "validation": class_counts(rows, val_idx),
            "test": class_counts(rows, test_idx),
        },
        "metrics": metrics,
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_history(args.outdir / "training_history.csv", history)
    torch.save(best_state, args.outdir / "model_state.pt")
    print(json.dumps(metrics, indent=2))
    print(f"Saved training report: {args.outdir / 'training_report.json'}")
    print(f"Saved model state: {args.outdir / 'model_state.pt'}")


if __name__ == "__main__":
    main()
