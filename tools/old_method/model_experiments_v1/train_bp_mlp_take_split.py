import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from speech_pipeline.ml_features import AVAILABLE_FEATURE_SETS, feature_names_for_set
from speech_pipeline.simple_mlp import MLP, MLPConfig, one_hot, standardize
from train_bp_mlp_baseline import (
    balance_training_indices,
    class_counts,
    evaluate_split,
    load_features,
    read_manifest,
    select_rows,
    stratified_validation_indices,
    write_history,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train B/P MLP with take 1 as train/validation and take 2 as test.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--feature-set", choices=AVAILABLE_FEATURE_SETS, default="mfcc_logmel_onset")
    parser.add_argument("--labels", nargs="+", default=["B", "P"])
    parser.add_argument("--include-speakers", nargs="*", default=["david"])
    parser.add_argument("--take-size", type=int, default=50)
    parser.add_argument("--train-take", type=int, default=1)
    parser.add_argument("--test-take", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def take_for_row(row: dict[str, str], take_size: int) -> int:
    file_index = int(row.get("file_index", "0") or "0")
    return (file_index - 1) // take_size + 1


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    indices: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["speaker", "word", "file_index", "true_label", "predicted_label", "confidence", "audio_path"],
        )
        writer.writeheader()
        pred = probs.argmax(axis=1)
        for local_idx, row_idx in enumerate(indices):
            row = rows[int(row_idx)]
            writer.writerow(
                {
                    "speaker": row["speaker"],
                    "word": row["word"],
                    "file_index": row["file_index"],
                    "true_label": class_names[int(labels[local_idx])],
                    "predicted_label": class_names[int(pred[local_idx])],
                    "confidence": round(float(np.max(probs[local_idx])), 4),
                    "audio_path": row["audio_path"],
                }
            )


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
    train_pool = np.array(
        [idx for idx, row in enumerate(rows) if take_for_row(row, args.take_size) == args.train_take],
        dtype=int,
    )
    test_idx = np.array(
        [idx for idx, row in enumerate(rows) if take_for_row(row, args.take_size) == args.test_take],
        dtype=int,
    )
    local_train, local_val = stratified_validation_indices(labels[train_pool], args.val_ratio, rng)
    train_idx = train_pool[local_train]
    val_idx = train_pool[local_val]
    train_idx = balance_training_indices(train_idx, labels, rng)

    train_x_raw = features[train_idx]
    val_x_raw = features[val_idx]
    train_x, mean, std = standardize(train_x_raw)
    val_x, _, _ = standardize(val_x_raw, mean, std)
    test_x, _, _ = standardize(features[test_idx], mean, std)
    train_y = labels[train_idx]
    val_y = labels[val_idx]
    test_y = labels[test_idx]

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
        "test_take": evaluate_split(model, test_x, test_y, class_names),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "training_report.json").write_text(
        json.dumps(
            {
                "model": "MLP",
                "feature_set": args.feature_set,
                "split": f"take_{args.train_take}_train_val_take_{args.test_take}_test",
                "architecture": [features.shape[1], args.hidden_dim, len(class_names)],
                "classes": class_names,
                "feature_names": feature_names,
                "manifest": str(args.manifest),
                "sample_counts": {
                    "selected_total": len(rows),
                    "train": class_counts(rows, train_idx),
                    "validation": class_counts(rows, val_idx),
                    "test_take": class_counts(rows, test_idx),
                },
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.outdir / "model_weights.json").write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    (args.outdir / "feature_normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist(), "feature_names": feature_names}, indent=2),
        encoding="utf-8",
    )
    write_history(args.outdir / "training_history.csv", history)
    write_split_manifest(
        args.outdir / "training_samples.csv",
        rows,
        {"train": train_idx, "validation": val_idx, "test_take": test_idx},
    )
    test_probs = model.predict_proba(test_x)
    write_predictions(args.outdir / "test_take_predictions.csv", rows, test_idx, test_y, test_probs, class_names)

    print(json.dumps(metrics, indent=2))
    print(f"Saved training report: {args.outdir / 'training_report.json'}")
    print(f"Saved model weights: {args.outdir / 'model_weights.json'}")


if __name__ == "__main__":
    main()
