import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from train_b_hubert_context_head import (
    ContextHead,
    RunLogger,
    SelectedRow,
    choose_device,
    class_weights,
    evaluate,
    extract_features,
    label_counts,
    load_cached_matrix,
    predict,
    read_csv,
    split_indices,
    stratified_split,
    write_csv,
    write_training_outputs,
)


DEFAULT_MODEL = "facebook/hubert-large-ll60k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a David-specific HuBERT teacher B/P classifier with a take-split test."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/david_bp_dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/ml_baseline/bp_hubert_large_teacher_david"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=900.0)
    parser.add_argument("--onset-ms", type=float, default=260.0)
    parser.add_argument("--trim-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-leading-silence", dest="trim_leading_silence", action="store_false")
    parser.add_argument("--leading-preroll-ms", type=float, default=25.0)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-map", choices=["raw", "b5_product", "p_product"], default="raw")
    parser.add_argument("--include-speakers", nargs="*", default=["david"])
    parser.add_argument("--include-labels", nargs="*", default=["B", "P"])
    parser.add_argument("--include-missing-stubs", action="store_true", default=True)
    parser.add_argument("--min-samples-per-class", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["random", "take_split", "both"], default="both")
    parser.add_argument("--take-size", type=int, default=50)
    parser.add_argument("--train-take", type=int, default=1)
    parser.add_argument("--test-take", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-speaker", default="")
    return parser.parse_args()


def str_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def cache_key(row: dict[str, str], args: argparse.Namespace) -> str:
    import hashlib

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


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace, cache_dir: Path) -> list[SelectedRow]:
    include_speakers = {item.lower() for item in args.include_speakers}
    include_labels = {item.upper() for item in args.include_labels}
    selected = []
    for row in rows:
        if not str_true(row.get("usable", "")):
            continue
        if include_speakers and row.get("speaker", "").lower() not in include_speakers:
            continue
        label = row.get(args.label_column, "").upper()
        if include_labels and label not in include_labels:
            continue
        audio_path = Path(row.get("audio_path", ""))
        if not audio_path.exists():
            continue
        selected.append(SelectedRow(row=row, label=label, cache_path=cache_dir / f"{cache_key(row, args)}.npz"))

    counts = label_counts([item.label for item in selected])
    allowed = {label for label, count in counts.items() if count >= args.min_samples_per_class}
    selected = [item for item in selected if item.label in allowed]
    selected.sort(key=lambda item: (item.row.get("speaker", ""), item.row.get("label", ""), item.row.get("file_index", "")))
    if args.max_rows > 0:
        selected = selected[: args.max_rows]
    return selected


def normalize_train(train_x: np.ndarray, *others: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized_train = (train_x - mean) / std
    normalized_others = [(item - mean) / std for item in others]
    return normalized_train.astype(np.float32), [item.astype(np.float32) for item in normalized_others], mean, std


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def take_for_row(row: dict[str, str], take_size: int) -> int:
    file_index = int(row.get("file_index", "0") or "0")
    return (file_index - 1) // take_size + 1


def take_split_indices(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pool = np.array(
        [idx for idx, row in enumerate(rows) if take_for_row(row, args.take_size) == args.train_take],
        dtype=int,
    )
    test_idx = np.array(
        [idx for idx, row in enumerate(rows) if take_for_row(row, args.take_size) == args.test_take],
        dtype=int,
    )
    if len(train_pool) == 0 or len(test_idx) == 0:
        raise ValueError("Take split is empty. Check file_index and take-size.")
    local_train, local_val = stratified_split(labels[train_pool], args.val_ratio, rng)
    return train_pool[local_train], train_pool[local_val], test_idx


def train_head_with_indices(
    features: np.ndarray,
    label_names: list[str],
    metadata: list[dict[str, str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    class_names = sorted(set(label_names))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array([class_to_idx[name] for name in label_names], dtype=int)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Train/validation split is empty.")

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
            logger(f"{split_name} epoch={epoch:04d} loss={train_loss:.4f} val_accuracy={val_acc:.3f} best={best_val:.3f}")
        if bad_epochs >= args.patience:
            logger(f"{split_name} early stop at epoch={epoch}; best_val_accuracy={best_val:.3f}")
            break

    model.load_state_dict(best_state)
    metrics = {
        "train": evaluate(model, train_x, train_y, class_names, device, args.batch_size),
        "validation": evaluate(model, val_x, val_y, class_names, device, args.batch_size),
        "test": evaluate(model, test_x, test_y, class_names, device, args.batch_size),
    }

    ordered_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)
    ordered_x = np.concatenate([train_x, val_x, test_x], axis=0)
    split_names = ["train"] * len(train_idx) + ["validation"] * len(val_idx) + ["test"] * len(test_idx)
    pred_idx, pred_probs = predict(model, ordered_x, device, args.batch_size)
    prediction_rows = []
    for local_i, original_i in enumerate(ordered_idx):
        row = metadata[int(original_i)]
        prediction_rows.append(
            {
                "split": split_names[local_i],
                "speaker": row.get("speaker", ""),
                "word": row.get("word", ""),
                "file_index": row.get("file_index", ""),
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
        "split_name": split_name,
    }


def write_outputs(result: dict[str, object], outdir: Path, args: argparse.Namespace, logger: RunLogger) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(result["model_state"], outdir / "model_state.pt")
    np.savez_compressed(
        outdir / "normalization_stats.npz",
        mean=result["normalization_mean"],
        std=result["normalization_std"],
        class_names=np.array(result["class_names"]),
    )
    write_csv(outdir / "training_history.csv", result["history"], ["epoch", "train_loss", "val_accuracy"])
    write_csv(
        outdir / "predictions.csv",
        result["prediction_rows"],
        ["split", "speaker", "word", "file_index", "true_label", "predicted_label", "confidence", "audio_path"],
    )
    report = {
        "script": str(Path(__file__).resolve()),
        "manifest": str(args.manifest),
        "label_column": args.label_column,
        "target_sample_rate": args.target_sample_rate,
        "window_ms": args.window_ms,
        "onset_ms": args.onset_ms,
        "trim_leading_silence": args.trim_leading_silence,
        "split_name": result["split_name"],
        "sample_counts": result["sample_counts"],
        "architecture": result["architecture"],
        "metrics": result["metrics"],
    }
    (outdir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger(f"Saved {result['split_name']} report: {outdir / 'training_report.json'}")


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    cache_dir = args.feature_cache_dir or (args.outdir / "feature_cache")
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.outdir / "run.log")
    try:
        logger("Starting HuBERT teacher B/P training")
        logger(f"Model: {args.model}")
        logger(f"Output: {args.outdir}")
        items = select_rows(read_csv(args.manifest), args, cache_dir)
        if len(items) < 4:
            raise ValueError("Not enough selected rows.")
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
            logger("Extract-only mode finished.")
            return

        features, label_names, metadata = load_cached_matrix(items)
        logger(f"Loaded cached feature matrix: shape={features.shape}")
        rng = np.random.default_rng(args.seed)
        class_names = sorted(set(label_names))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        labels = np.array([class_to_idx[name] for name in label_names], dtype=int)

        results = {}
        if args.split_mode in {"random", "both"}:
            train_idx, val_idx, test_idx = split_indices(metadata, labels, args, rng)
            random_result = train_head_with_indices(
                features, label_names, metadata, train_idx, val_idx, test_idx, "random_validation", args, logger
            )
            write_outputs(random_result, args.outdir / "random_validation", args, logger)
            results["random_validation"] = random_result["metrics"]

        if args.split_mode in {"take_split", "both"}:
            train_idx, val_idx, test_idx = take_split_indices(metadata, labels, args, rng)
            take_result = train_head_with_indices(
                features, label_names, metadata, train_idx, val_idx, test_idx, "take_split", args, logger
            )
            write_outputs(take_result, args.outdir / "take_split", args, logger)
            results["take_split"] = take_result["metrics"]

            rev_train_take = args.test_take
            rev_test_take = args.train_take
            original_train_take, original_test_take = args.train_take, args.test_take
            args.train_take, args.test_take = rev_train_take, rev_test_take
            train_idx, val_idx, test_idx = take_split_indices(metadata, labels, args, rng)
            reverse_result = train_head_with_indices(
                features, label_names, metadata, train_idx, val_idx, test_idx, "reverse_take_split", args, logger
            )
            write_outputs(reverse_result, args.outdir / "reverse_take_split", args, logger)
            results["reverse_take_split"] = reverse_result["metrics"]
            args.train_take, args.test_take = original_train_take, original_test_take

        summary = {
            "model": args.model,
            "manifest": str(args.manifest),
            "selected_rows": len(items),
            "label_counts": label_counts([item.label for item in items]),
            "results": results,
        }
        (args.outdir / "teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger(f"Final summary: {json.dumps(summary, indent=2)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
