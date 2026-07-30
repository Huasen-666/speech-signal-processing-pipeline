import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float
from speech_pipeline.ml_features import (
    AVAILABLE_FEATURE_SETS,
    extract_feature_set,
    feature_names_for_set,
)
from speech_pipeline.simple_mlp import MLP, MLPConfig, one_hot, standardize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small B/P MLP baseline from dataset_manifest.csv.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/ml_baseline/bp_mlp_parcor"))
    parser.add_argument("--labels", nargs="+", default=["B", "P"])
    parser.add_argument("--feature-set", choices=AVAILABLE_FEATURE_SETS, default="parcor")
    parser.add_argument("--include-speakers", nargs="*", help="Optional speaker filter, e.g. bascom corrick.")
    parser.add_argument("--test-speaker", help="Hold out one speaker as the final test split.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance-train", action="store_true", help="Disable class balancing on the train split.")
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


def load_features(rows: list[dict[str, str]], feature_set: str) -> tuple[np.ndarray, list[str]]:
    feature_names = feature_names_for_set(feature_set)
    features = []
    errors = []
    valid_rows = []
    for row in rows:
        path = Path(row["audio_path"])
        try:
            sample_rate, audio = read_wav_float(path)
            features.append(extract_feature_set(audio, sample_rate, feature_set))
            valid_rows.append(row)
        except (OSError, ValueError, wave.Error) as exc:  # type: ignore[name-defined]
            errors.append({"audio_path": str(path), "error": str(exc)})

    if errors:
        joined = "\n".join(f"{item['audio_path']}: {item['error']}" for item in errors[:5])
        raise RuntimeError(f"Failed to load {len(errors)} audio files. First errors:\n{joined}")
    rows[:] = valid_rows
    return np.vstack(features).astype(np.float64), feature_names


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

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    return train_idx, val_idx


def split_rows(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    val_ratio: float,
    rng: np.random.Generator,
    test_speaker: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(rows))
    if test_speaker:
        test_speaker = test_speaker.lower()
        test_idx = np.array([idx for idx, row in enumerate(rows) if row["speaker"].lower() == test_speaker], dtype=int)
        remaining_idx = np.array([idx for idx in all_indices if idx not in set(test_idx.tolist())], dtype=int)
        if len(test_idx) == 0:
            raise ValueError(f"No usable rows found for test speaker: {test_speaker}")
        local_train, local_val = stratified_validation_indices(labels[remaining_idx], val_ratio, rng)
        return remaining_idx[local_train], remaining_idx[local_val], test_idx

    train_idx, val_idx = stratified_validation_indices(labels, val_ratio, rng)
    return train_idx, val_idx, np.array([], dtype=int)


def balance_training_indices(indices: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    by_label = []
    for label in sorted(set(labels[indices].tolist())):
        label_indices = indices[labels[indices] == label]
        by_label.append(rng.permutation(label_indices))
    min_count = min(len(part) for part in by_label)
    balanced = np.concatenate([part[:min_count] for part in by_label])
    return rng.permutation(balanced)


def class_counts(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        label = rows[int(idx)]["label"].upper()
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def evaluate_split(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    if len(features) == 0:
        return {"count": 0}
    probs = model.predict_proba(features)
    pred = probs.argmax(axis=1)
    matrix = confusion_matrix(labels, pred, len(class_names))
    per_class = {}
    for idx, class_name in enumerate(class_names):
        actual_count = int(matrix[idx].sum())
        correct_count = int(matrix[idx, idx])
        per_class[class_name] = {
            "count": actual_count,
            "accuracy": round(correct_count / actual_count, 4) if actual_count else None,
        }
    return {
        "count": int(len(labels)),
        "accuracy": round(float(np.mean(pred == labels)), 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def write_split_manifest(path: Path, rows: list[dict[str, str]], splits: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["split"] if rows else ["split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, indices in splits.items():
            for idx in indices:
                row = dict(rows[int(idx)])
                row["split"] = split_name
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    class_names = [label.upper() for label in args.labels]
    class_to_index = {label: idx for idx, label in enumerate(class_names)}

    rows = select_rows(read_manifest(args.manifest), class_names, args.include_speakers)
    if len(rows) < 4:
        raise ValueError("Not enough usable samples after filtering.")

    features, feature_names = load_features(rows, args.feature_set)
    labels = np.array([class_to_index[row["label"].upper()] for row in rows], dtype=int)
    train_idx, val_idx, test_idx = split_rows(rows, labels, args.val_ratio, rng, args.test_speaker)
    if not args.no_balance_train:
        train_idx = balance_training_indices(train_idx, labels, rng)

    train_x_raw = features[train_idx]
    val_x_raw = features[val_idx]
    train_x, mean, std = standardize(train_x_raw)
    val_x, _, _ = standardize(val_x_raw, mean, std)
    test_x = np.empty((0, features.shape[1]), dtype=np.float64)
    if len(test_idx):
        test_x, _, _ = standardize(features[test_idx], mean, std)

    train_y = labels[train_idx]
    val_y = labels[val_idx]
    test_y = labels[test_idx] if len(test_idx) else np.array([], dtype=int)

    model = MLP(
        MLPConfig(
            input_dim=features.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=len(class_names),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    )
    encoded_train = one_hot(train_y, len(class_names))
    history = []
    best_accuracy = -1.0
    best_snapshot = model.snapshot()

    for epoch in range(1, args.epochs + 1):
        loss = model.train_epoch(train_x, encoded_train, args.batch_size, rng)
        val_accuracy = float(np.mean(model.predict(val_x) == val_y))
        history.append({"epoch": epoch, "loss": round(loss, 6), "val_accuracy": round(val_accuracy, 6)})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_snapshot = model.snapshot()
        if epoch == 1 or epoch == args.epochs or epoch % 50 == 0:
            print(f"epoch={epoch:04d} loss={loss:.4f} val_accuracy={val_accuracy:.3f} best={best_accuracy:.3f}")

    model.restore(best_snapshot)

    metrics = {
        "train": evaluate_split(model, train_x, train_y, class_names),
        "validation": evaluate_split(model, val_x, val_y, class_names),
        "test": evaluate_split(model, test_x, test_y, class_names),
    }
    report = {
        "model": "MLP",
        "feature_set": args.feature_set,
        "dave_comparable": args.feature_set == "parcor",
        "architecture": [features.shape[1], args.hidden_dim, len(class_names)],
        "classes": class_names,
        "feature_names": feature_names,
        "manifest": str(args.manifest),
        "include_speakers": args.include_speakers or "all",
        "test_speaker": args.test_speaker,
        "balance_train": not args.no_balance_train,
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
    (args.outdir / "model_weights.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    (args.outdir / "feature_normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "feature_names": feature_names}, indent=2),
        encoding="utf-8",
    )
    write_history(args.outdir / "training_history.csv", history)
    write_split_manifest(
        args.outdir / "training_samples.csv",
        rows,
        {"train": train_idx, "validation": val_idx, "test": test_idx},
    )

    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved training report: {args.outdir / 'training_report.json'}")
    print(f"Saved model weights: {args.outdir / 'model_weights.json'}")


if __name__ == "__main__":
    main()
