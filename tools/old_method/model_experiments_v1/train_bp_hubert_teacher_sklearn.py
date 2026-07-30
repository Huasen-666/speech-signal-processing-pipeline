import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from train_b_hubert_context_head import RunLogger, extract_features, label_counts, load_cached_matrix, read_csv, stratified_split
from train_bp_hubert_teacher_take_split import select_rows, take_for_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sklearn classifiers on frozen HuBERT features.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/david_bp_dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/ml_baseline/bp_hubert_teacher_sklearn"))
    parser.add_argument("--model", default="facebook/hubert-large-ll60k")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=900.0)
    parser.add_argument("--onset-ms", type=float, default=260.0)
    parser.add_argument("--trim-leading-silence", action="store_true", default=True)
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
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--take-size", type=int, default=50)
    parser.add_argument("--train-take", type=int, default=1)
    parser.add_argument("--test-take", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--test-speaker", default="")
    return parser.parse_args()


def encode_labels(label_names: list[str]) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(label_names))
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.array([mapping[label] for label in label_names], dtype=int), classes


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> list[list[int]]:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, pred in zip(y_true, y_pred):
        matrix[int(actual), int(pred)] += 1
    return matrix.tolist()


def metrics_for(model, x: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    if len(y) == 0:
        return {"count": 0}
    pred = model.predict(x)
    matrix = np.array(confusion_matrix(y, pred, len(classes)))
    per_class = {}
    for idx, name in enumerate(classes):
        total = int(matrix[idx].sum())
        correct = int(matrix[idx, idx])
        per_class[name] = {"count": total, "accuracy": round(correct / total, 4) if total else None}
    return {
        "count": int(len(y)),
        "accuracy": round(float(np.mean(pred == y)), 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def split_random(labels: np.ndarray, val_ratio: float, rng: np.random.Generator):
    train_idx, val_idx = stratified_split(labels, val_ratio, rng)
    return train_idx, val_idx, np.array([], dtype=int)


def split_take(metadata: list[dict[str, str]], labels: np.ndarray, args: argparse.Namespace, reverse: bool = False):
    train_take = args.test_take if reverse else args.train_take
    test_take = args.train_take if reverse else args.test_take
    train_pool = np.array(
        [idx for idx, row in enumerate(metadata) if take_for_row(row, args.take_size) == train_take],
        dtype=int,
    )
    test_idx = np.array(
        [idx for idx, row in enumerate(metadata) if take_for_row(row, args.take_size) == test_take],
        dtype=int,
    )
    rng = np.random.default_rng(args.seed + (10 if reverse else 0))
    local_train, local_val = stratified_split(labels[train_pool], args.val_ratio, rng)
    return train_pool[local_train], train_pool[local_val], test_idx


def train_best_classifier(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = []
    n_train = len(train_idx)
    pca_options = [0, 8, 16, 32, 64]
    pca_options = [n for n in pca_options if n == 0 or n < n_train]
    c_options = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    for pca_dim in pca_options:
        for c in c_options:
            steps = [("scaler", StandardScaler())]
            if pca_dim:
                steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
            steps.append(
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                )
            )
            candidates.append((f"logreg_pca{pca_dim}_C{c}", Pipeline(steps)))
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        candidates.append(
            (
                f"ridge_alpha{alpha}",
                Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier(alpha=alpha, class_weight="balanced"))]),
            )
        )

    best = None
    rows = []
    for name, model in candidates:
        model.fit(features[train_idx], labels[train_idx])
        val_pred = model.predict(features[val_idx])
        val_acc = float(np.mean(val_pred == labels[val_idx]))
        rows.append({"candidate": name, "validation_accuracy": round(val_acc, 4)})
        if best is None or val_acc > best[0]:
            best = (val_acc, name, model)
    return best[1], best[2], rows


def prediction_rows(model, metadata: list[dict[str, str]], indices: np.ndarray, labels: np.ndarray, classes: list[str], features: np.ndarray):
    pred = model.predict(features[indices])
    rows = []
    for local_idx, original_idx in enumerate(indices):
        row = metadata[int(original_idx)]
        rows.append(
            {
                "speaker": row.get("speaker", ""),
                "word": row.get("word", ""),
                "file_index": row.get("file_index", ""),
                "true_label": classes[int(labels[int(original_idx)])],
                "predicted_label": classes[int(pred[local_idx])],
                "audio_path": row.get("audio_path", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_split(name: str, features: np.ndarray, labels: np.ndarray, classes: list[str], metadata: list[dict[str, str]], indices, outdir: Path, seed: int):
    train_idx, val_idx, test_idx = indices
    candidate_name, model, candidate_rows = train_best_classifier(features, labels, train_idx, val_idx, seed)
    metrics = {
        "best_candidate": candidate_name,
        "train": metrics_for(model, features[train_idx], labels[train_idx], classes),
        "validation": metrics_for(model, features[val_idx], labels[val_idx], classes),
        "test": metrics_for(model, features[test_idx], labels[test_idx], classes),
    }
    split_dir = outdir / name
    split_dir.mkdir(parents=True, exist_ok=True)
    write_csv(split_dir / "candidate_search.csv", candidate_rows)
    if len(test_idx):
        write_csv(split_dir / "test_predictions.csv", prediction_rows(model, metadata, test_idx, labels, classes, features))
    (split_dir / "training_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.outdir / "run.log")
    try:
        logger(f"Starting sklearn HuBERT teacher: {args.model}")
        cache_dir = args.feature_cache_dir or (args.outdir / "feature_cache")
        items = select_rows(read_csv(args.manifest), args, cache_dir)
        logger(f"Selected rows: {len(items)}; counts={label_counts([item.label for item in items])}")
        extracted = extract_features(items, args, logger)
        if extracted:
            write_csv(args.outdir / "newly_extracted_features.csv", extracted)
        features, label_names, metadata = load_cached_matrix(items)
        labels, classes = encode_labels(label_names)
        rng = np.random.default_rng(args.seed)
        results = {
            "random_validation": run_split(
                "random_validation",
                features,
                labels,
                classes,
                metadata,
                split_random(labels, args.val_ratio, rng),
                args.outdir,
                args.seed,
            ),
            "take_split": run_split(
                "take_split",
                features,
                labels,
                classes,
                metadata,
                split_take(metadata, labels, args, reverse=False),
                args.outdir,
                args.seed,
            ),
            "reverse_take_split": run_split(
                "reverse_take_split",
                features,
                labels,
                classes,
                metadata,
                split_take(metadata, labels, args, reverse=True),
                args.outdir,
                args.seed,
            ),
        }
        summary = {
            "model": args.model,
            "manifest": str(args.manifest),
            "selected_rows": len(items),
            "label_counts": label_counts(label_names),
            "feature_shape": list(features.shape),
            "results": results,
        }
        (args.outdir / "teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger(f"Final summary: {json.dumps(summary, indent=2)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
