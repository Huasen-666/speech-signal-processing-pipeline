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
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from render_hubert_dynamic_bp_replacement import DynamicConsonantHead
from train_b_hubert_context_head import RunLogger, SelectedRow, extract_features, load_cached_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved HuBERT dynamic B/P model on a separate validation manifest."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dave_bp_drb_validation_manifest.csv"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("experiments/phone_prototype/dave_hubert_dynamic_bp_replacement"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/phone_prototype/dave_hubert_dynamic_bp_replacement/eval_drb_validation"),
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=0.0)
    parser.add_argument("--onset-ms", type=float, default=0.0)
    parser.add_argument("--trim-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-leading-silence", dest="trim_leading_silence", action="store_false")
    parser.add_argument("--leading-preroll-ms", type=float, default=20.0)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-map", choices=["raw", "b5_product", "p_product"], default="raw")
    parser.add_argument("--include-speakers", nargs="*", default=["dave"])
    parser.add_argument("--include-labels", nargs="*", default=["B", "P"])
    parser.add_argument("--include-missing-stubs", action="store_true", default=True)
    parser.add_argument("--min-samples-per-class", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def str_true_or_missing(value: str) -> bool:
    if value == "":
        return True
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_defaults(args: argparse.Namespace) -> None:
    summary_path = args.model_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not args.model:
            args.model = summary.get("model", "") or "facebook/hubert-base-ls960"
        if args.window_ms <= 0:
            args.window_ms = float(summary.get("feature_window_ms", 320.0))
        if args.onset_ms <= 0:
            args.onset_ms = float(summary.get("onset_pool_ms", 180.0))

    if not args.model:
        args.model = "facebook/hubert-base-ls960"
    if args.window_ms <= 0:
        args.window_ms = 320.0
    if args.onset_ms <= 0:
        args.onset_ms = 180.0


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
        "eval_script": "evaluate_hubert_dynamic_bp_model_v1",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace, cache_dir: Path) -> list[SelectedRow]:
    include_speakers = {speaker.lower() for speaker in args.include_speakers}
    include_labels = {label.upper() for label in args.include_labels}
    selected = []
    for row in rows:
        if not str_true_or_missing(row.get("usable", "")):
            continue
        if include_speakers and row.get("speaker", "").lower() not in include_speakers:
            continue
        label = row.get(args.label_column, "").upper()
        if include_labels and label not in include_labels:
            continue
        audio_path = resolve_path(Path(row["audio_path"]))
        if not audio_path.exists():
            continue
        normalized = dict(row)
        normalized["audio_path"] = str(audio_path)
        normalized.setdefault("file_index", row.get("word_index", row.get("global_index", "")))
        normalized.setdefault("protocol_index", row.get("word_index", ""))
        selected.append(SelectedRow(row=normalized, label=label, cache_path=cache_dir / f"{cache_key(normalized, args)}.npz"))
    selected.sort(key=lambda item: (item.row.get("label", ""), item.row.get("word_index", ""), item.row.get("word", "")))
    if args.max_rows > 0:
        selected = selected[: args.max_rows]
    return selected


def confusion_matrix(true_ids: np.ndarray, pred_ids: np.ndarray, class_count: int) -> list[list[int]]:
    matrix = np.zeros((class_count, class_count), dtype=int)
    for true_id, pred_id in zip(true_ids, pred_ids):
        matrix[int(true_id), int(pred_id)] += 1
    return matrix.tolist()


def main() -> None:
    args = parse_args()
    args.model_dir = resolve_path(args.model_dir)
    args.outdir = resolve_path(args.outdir)
    args.manifest = resolve_path(args.manifest)
    load_training_defaults(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(args.outdir / "run.log")
    try:
        logger(f"Evaluating model dir: {args.model_dir}")
        logger(f"Validation manifest: {args.manifest}")
        logger(f"HuBERT model: {args.model}; window_ms={args.window_ms}; onset_ms={args.onset_ms}")

        cache_dir = args.feature_cache_dir or (args.outdir / "feature_cache")
        items = select_rows(read_csv(args.manifest), args, cache_dir)
        if not items:
            raise ValueError("No validation rows selected.")
        logger(f"Selected validation rows: {len(items)}")

        extracted = extract_features(items, args, logger)
        if extracted:
            write_csv(args.outdir / "newly_extracted_features.csv", extracted)
        features, label_names, metadata = load_cached_matrix(items)

        stats = np.load(args.model_dir / "dynamic_head_stats.npz", allow_pickle=False)
        class_names = [str(item) for item in stats["class_names"].tolist()]
        mean = stats["feature_mean"].astype(np.float32)
        std = stats["feature_std"].astype(np.float32)
        duration_mean = float(stats["duration_mean"][0])
        duration_std = float(stats["duration_std"][0])

        state = torch.load(args.model_dir / "dynamic_head_model_state.pt", map_location="cpu")
        hidden_size = int(state["backbone.0.weight"].shape[0])
        input_dim = int(mean.shape[0])
        model = DynamicConsonantHead(input_dim=input_dim, hidden_size=hidden_size, class_count=len(class_names), dropout=0.0)
        model.load_state_dict(state)
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        x = ((features - mean[None, :]) / std[None, :]).astype(np.float32)
        with torch.no_grad():
            logits, duration_z = model(torch.tensor(x, dtype=torch.float32, device=device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
            pred_duration_ms = duration_z.cpu().numpy() * duration_std + duration_mean

        class_to_id = {name: idx for idx, name in enumerate(class_names)}
        true_ids = np.array([class_to_id[label.upper()] for label in label_names], dtype=int)
        accuracy = float(np.mean(pred_ids == true_ids))
        matrix = confusion_matrix(true_ids, pred_ids, len(class_names))

        rows = []
        for row, true_label, pred_id, conf, pred_ms in zip(metadata, label_names, pred_ids, confidence, pred_duration_ms):
            rows.append(
                {
                    "speaker": row.get("speaker", ""),
                    "word": row.get("word", ""),
                    "word_index": row.get("word_index", row.get("file_index", "")),
                    "true_label": true_label.upper(),
                    "predicted_label": class_names[int(pred_id)],
                    "confidence": round(float(conf), 4),
                    "correct": str(class_names[int(pred_id)] == true_label.upper()).lower(),
                    "predicted_duration_ms": round(float(pred_ms), 2),
                    "audio_path": row.get("audio_path", ""),
                }
            )
        prediction_csv = args.outdir / "validation_predictions.csv"
        write_csv(prediction_csv, rows)

        summary = {
            "model_dir": str(args.model_dir),
            "manifest": str(args.manifest),
            "selected_rows": len(items),
            "class_names": class_names,
            "accuracy": round(accuracy, 4),
            "confusion_matrix_rows_true_cols_pred": matrix,
            "correct_count": int(np.sum(pred_ids == true_ids)),
            "wrong_count": int(np.sum(pred_ids != true_ids)),
            "mean_confidence": round(float(np.mean(confidence)), 4),
            "mean_predicted_duration_ms": round(float(np.mean(pred_duration_ms)), 2),
            "prediction_csv": str(prediction_csv),
            "feature_shape": list(features.shape),
            "hubert_model": args.model,
            "window_ms": args.window_ms,
            "onset_ms": args.onset_ms,
        }
        (args.outdir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger(f"Evaluation summary: {json.dumps(summary, indent=2)}")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
