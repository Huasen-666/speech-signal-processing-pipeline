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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from speech_pipeline.audio_io import read_wav_float, write_wav_float

from build_david_bp_subtype_manifests import B_WORD_TO_SUBTYPE, P_WORD_TO_SUBTYPE
from render_b_initial_stub_library_demo import (
    prepare_stub_for_insert,
    tempo_suffix,
    trim_word_boundaries,
    write_sequence_with_tempo,
)
from render_dave_stub_replacement_demo import adapt_stub_level, crossfade_join, load_stub, resample_to
from run_phone_consonant_enhancement_prototype import fit_stub_to_length
from train_b_hubert_context_head import (
    RunLogger,
    SelectedRow,
    choose_device,
    extract_features,
    label_counts,
    load_cached_matrix,
    stratified_split,
)


class DynamicConsonantHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, class_count: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, class_count)
        self.duration = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.classifier(hidden), self.duration(hidden).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use frozen HuBERT features to train a B/P classifier plus a consonant-duration head, "
            "then render dynamic initial-consonant replacement audio."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dave_bp_consecutive_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/dave_hubert_dynamic_bp_replacement"))
    parser.add_argument("--model", default="facebook/hubert-base-ls960")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=float, default=320.0)
    parser.add_argument("--onset-ms", type=float, default=180.0)
    parser.add_argument("--trim-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-leading-silence", dest="trim_leading_silence", action="store_false")
    parser.add_argument("--leading-preroll-ms", type=float, default=20.0)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-map", choices=["raw", "b5_product", "p_product"], default="raw")
    parser.add_argument("--include-speakers", nargs="*", default=["dave"])
    parser.add_argument("--include-labels", nargs="*", default=["B", "P"])
    parser.add_argument("--include-missing-stubs", action="store_true", default=True)
    parser.add_argument("--min-samples-per-class", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--duration-loss-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--duration-blend-model", type=float, default=0.65)
    parser.add_argument("--mask-min-ms", type=float, default=55.0)
    parser.add_argument("--mask-max-ms", type=float, default=240.0)
    parser.add_argument("--p-extra-ms", type=float, default=18.0)
    parser.add_argument("--stub-fit-mode", choices=["crop", "stretch"], default="stretch")
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--level-mode", choices=["match_rms", "dave_peak"], default="match_rms")
    parser.add_argument("--stub-rms-ratio", type=float, default=1.2)
    parser.add_argument("--crossfade-ms", type=float, default=18.0)
    parser.add_argument("--sequence-gap-ms", type=float, default=160.0)
    parser.add_argument("--sequence-tempo", type=float, default=1.0)
    parser.add_argument("--sequence-time-mode", choices=["tempo", "speed"], default="tempo")
    parser.add_argument("--trim-word-leading-silence", action="store_true", default=True)
    parser.add_argument("--no-trim-word-leading-silence", dest="trim_word_leading_silence", action="store_false")
    parser.add_argument("--trim-word-start-offset-ms", type=float, default=4.0)
    parser.add_argument("--trim-word-tail-ms", type=float, default=18.0)
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


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_key(row: dict[str, str], args: argparse.Namespace) -> str:
    import hashlib

    audio_path = resolve_path(row["audio_path"])
    stat = audio_path.stat()
    payload = {
        "audio_path": str(audio_path.resolve()).lower(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "model": args.model,
        "target_sample_rate": args.target_sample_rate,
        "window_ms": args.window_ms,
        "onset_ms": args.onset_ms,
        "trim_leading_silence": args.trim_leading_silence,
        "leading_preroll_ms": args.leading_preroll_ms,
        "label_column": args.label_column,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace, cache_dir: Path) -> list[SelectedRow]:
    include_speakers = {speaker.lower() for speaker in args.include_speakers}
    include_labels = {label.upper() for label in args.include_labels}
    selected: list[SelectedRow] = []
    for row in rows:
        if not str_true_or_missing(row.get("usable", "")):
            continue
        if include_speakers and row.get("speaker", "").lower() not in include_speakers:
            continue
        label = row.get(args.label_column, "").upper()
        if include_labels and label not in include_labels:
            continue
        audio_path = resolve_path(row.get("audio_path", ""))
        if not audio_path.exists():
            continue
        normalized = dict(row)
        normalized["audio_path"] = str(audio_path)
        normalized.setdefault("file_index", row.get("word_index", row.get("global_index", "")))
        normalized.setdefault("protocol_index", row.get("word_index", ""))
        selected.append(SelectedRow(row=normalized, label=label, cache_path=cache_dir / f"{cache_key(normalized, args)}.npz"))

    counts = label_counts([item.label for item in selected])
    allowed = {label for label, count in counts.items() if count >= args.min_samples_per_class}
    selected = [item for item in selected if item.label in allowed]
    selected.sort(key=lambda item: (item.row.get("label", ""), item.row.get("word_index", ""), item.row.get("word", "")))
    if args.max_rows > 0:
        selected = selected[: args.max_rows]
    return selected


def list_stubs_normalized(root: Path) -> dict[str, list[Path]]:
    stubs: dict[str, list[Path]] = {}
    for group_dir in sorted(root.iterdir() if root.exists() else []):
        if not group_dir.is_dir():
            continue
        files = sorted(group_dir.glob("*.wav"))
        if not files:
            continue
        key = group_dir.name.strip().upper()
        stubs.setdefault(key, []).extend(files)
    return stubs


def find_energy_onset(audio: np.ndarray, sample_rate: int, frame_ms: float = 8.0, hop_ms: float = 2.0) -> int:
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame:
        return 0
    starts = np.arange(0, len(audio) - frame + 1, hop)
    values = np.array(
        [np.sqrt(np.mean(audio[start : start + frame].astype(np.float64) ** 2) + 1e-12) for start in starts]
    )
    db = 20.0 * np.log10(values + 1e-12)
    threshold = max(float(np.percentile(db, 10)) + 12.0, float(np.percentile(db, 95)) - 32.0, -60.0)
    active = np.flatnonzero(db >= threshold)
    return int(starts[active[0]]) if len(active) else 0


def estimate_vowel_transition_ms(
    audio: np.ndarray,
    sample_rate: int,
    label: str,
    min_ms: float,
    max_ms: float,
) -> tuple[float, dict[str, float]]:
    onset = find_energy_onset(audio, sample_rate)
    lookahead = min(len(audio), onset + round(sample_rate * (max_ms + 80.0) / 1000.0))
    chunk = audio[onset:lookahead].astype(np.float64)
    frame_len = max(32, round(sample_rate * 0.012))
    hop_len = max(8, round(sample_rate * 0.003))
    if len(chunk) < frame_len:
        fallback = 115.0 if label == "P" else 85.0
        return float(np.clip(fallback, min_ms, max_ms)), {"fallback": 1.0, "onset_sample": float(onset)}

    window = np.hanning(frame_len)
    times = []
    rms_db = []
    low_ratio = []
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    low_mask = (freqs >= 100.0) & (freqs <= 1200.0)
    full_mask = (freqs >= 80.0) & (freqs <= 5000.0)
    for start in range(0, len(chunk) - frame_len + 1, hop_len):
        frame = chunk[start : start + frame_len]
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        full_energy = float(np.sum(spectrum[full_mask]) + 1e-12)
        low_energy = float(np.sum(spectrum[low_mask]))
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        times.append((start + frame_len / 2) / sample_rate * 1000.0)
        rms_db.append(20.0 * np.log10(rms + 1e-12))
        low_ratio.append(low_energy / full_energy)

    times_np = np.asarray(times)
    rms_np = np.asarray(rms_db)
    ratio_np = np.asarray(low_ratio)
    high_rms = float(np.percentile(rms_np, 90))
    ratio_floor = float(np.percentile(ratio_np, 35))
    rms_threshold = high_rms - (16.0 if label == "P" else 19.0)
    ratio_threshold = max(0.35, ratio_floor + 0.08)

    earliest = min_ms + (15.0 if label == "P" else 0.0)
    for idx, time_ms in enumerate(times_np):
        if time_ms < earliest:
            continue
        end_idx = min(len(times_np), idx + 4)
        if end_idx - idx < 3:
            continue
        if np.mean(rms_np[idx:end_idx] >= rms_threshold) >= 0.75 and np.mean(ratio_np[idx:end_idx] >= ratio_threshold) >= 0.5:
            duration = float(np.clip(time_ms, min_ms, max_ms))
            return duration, {
                "fallback": 0.0,
                "onset_sample": float(onset),
                "rms_threshold_db": float(rms_threshold),
                "low_ratio_threshold": float(ratio_threshold),
            }

    fallback = 128.0 if label == "P" else 92.0
    return float(np.clip(fallback, min_ms, max_ms)), {
        "fallback": 1.0,
        "onset_sample": float(onset),
        "rms_threshold_db": float(rms_threshold),
        "low_ratio_threshold": float(ratio_threshold),
    }


def subtype_for_word(label: str, word: str) -> str:
    mapping = B_WORD_TO_SUBTYPE if label == "B" else P_WORD_TO_SUBTYPE
    return mapping.get(word.lower(), f"{label}_AH")


def convert_subtype(subtype: str, target_label: str, available: set[str]) -> str:
    suffix = subtype.split("_", 1)[1] if "_" in subtype else ""
    candidate = f"{target_label}_{suffix}" if suffix else ""
    if candidate in available:
        return candidate
    fallback_by_suffix = {
        "L": f"{target_label}_AE",
        "R": f"{target_label}_AH",
        "AA": f"{target_label}_AH",
        "AO": f"{target_label}_AH",
        "OW": f"{target_label}_UH",
        "OY": f"{target_label}_EY",
        "IY": f"{target_label}_IH",
    }
    fallback = fallback_by_suffix.get(suffix, f"{target_label}_AH")
    if fallback in available:
        return fallback
    return sorted(available)[0]


def normalize_train(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean).astype(np.float32) / std.astype(np.float32), (eval_x - mean).astype(np.float32) / std.astype(
        np.float32
    ), mean.astype(np.float32), std.astype(np.float32)


def encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.array([mapping[label] for label in labels], dtype=np.int64), classes


def duration_stats(train_duration_ms: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(train_duration_ms))
    std = float(np.std(train_duration_ms))
    return mean, std if std >= 1e-6 else 1.0


def make_loader(x: np.ndarray, y: np.ndarray, duration_z: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(duration_z, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: DynamicConsonantHead,
    x: np.ndarray,
    y: np.ndarray,
    duration_ms: np.ndarray,
    class_names: list[str],
    duration_mean: float,
    duration_std: float,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0:
        return {"count": 0}, np.array([]), np.array([]), np.array([])
    model.eval()
    pred_ids = []
    pred_conf = []
    pred_duration = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits, dur_z = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            ids = np.argmax(probs, axis=1)
            pred_ids.append(ids)
            pred_conf.append(np.max(probs, axis=1))
            pred_duration.append(dur_z.cpu().numpy() * duration_std + duration_mean)
    pred_ids_np = np.concatenate(pred_ids)
    pred_conf_np = np.concatenate(pred_conf)
    pred_duration_np = np.concatenate(pred_duration)
    accuracy = float(np.mean(pred_ids_np == y))
    mae = float(np.mean(np.abs(pred_duration_np - duration_ms)))
    per_class = {}
    for idx, name in enumerate(class_names):
        mask = y == idx
        per_class[name] = {
            "count": int(np.count_nonzero(mask)),
            "accuracy": round(float(np.mean(pred_ids_np[mask] == y[mask])), 4) if np.any(mask) else None,
        }
    return {
        "count": int(len(x)),
        "accuracy": round(accuracy, 4),
        "duration_mae_ms": round(mae, 2),
        "per_class": per_class,
    }, pred_ids_np, pred_conf_np, pred_duration_np


def train_head(
    features: np.ndarray,
    label_names: list[str],
    duration_ms: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    logger: RunLogger,
    split_name: str,
) -> dict[str, object]:
    device = choose_device(args.device)
    y, class_names = encode_labels(label_names)
    train_x, val_x, mean, std = normalize_train(features[train_idx], features[val_idx])
    train_duration = duration_ms[train_idx]
    val_duration = duration_ms[val_idx]
    dur_mean, dur_std = duration_stats(train_duration)
    train_duration_z = ((train_duration - dur_mean) / dur_std).astype(np.float32)

    model = DynamicConsonantHead(features.shape[1], args.hidden_size, len(class_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    loader = make_loader(train_x, y[train_idx], train_duration_z, args.batch_size, shuffle=True)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = -1e9
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y, batch_duration_z in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_duration_z = batch_duration_z.to(device)
            optimizer.zero_grad()
            logits, pred_duration_z = model(batch_x)
            loss = ce_loss(logits, batch_y) + args.duration_loss_weight * mse_loss(pred_duration_z, batch_duration_z)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _, _, _ = evaluate_model(
            model,
            val_x,
            y[val_idx],
            val_duration,
            class_names,
            dur_mean,
            dur_std,
            device,
            args.batch_size,
        )
        score = float(val_metrics["accuracy"]) - float(val_metrics["duration_mae_ms"]) / 1000.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 6),
                "val_accuracy": val_metrics["accuracy"],
                "val_duration_mae_ms": val_metrics["duration_mae_ms"],
            }
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            logger(
                f"{split_name} epoch={epoch:04d} loss={np.mean(losses):.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} val_dur_mae={val_metrics['duration_mae_ms']:.1f}ms"
            )

    model.load_state_dict(best_state)
    train_metrics, _, _, _ = evaluate_model(
        model,
        train_x,
        y[train_idx],
        train_duration,
        class_names,
        dur_mean,
        dur_std,
        device,
        args.batch_size,
    )
    val_metrics, pred_ids, pred_conf, pred_duration = evaluate_model(
        model,
        val_x,
        y[val_idx],
        val_duration,
        class_names,
        dur_mean,
        dur_std,
        device,
        args.batch_size,
    )
    return {
        "model": model,
        "state_dict": best_state,
        "feature_mean": mean.squeeze(0),
        "feature_std": std.squeeze(0),
        "duration_mean": dur_mean,
        "duration_std": dur_std,
        "class_names": class_names,
        "history": history,
        "metrics": {"train": train_metrics, "validation": val_metrics},
        "val_predictions": {
            "indices": val_idx.tolist(),
            "pred_ids": pred_ids.tolist(),
            "confidence": pred_conf.tolist(),
            "duration_ms": pred_duration.tolist(),
        },
    }


def train_final_head(
    features: np.ndarray,
    label_names: list[str],
    duration_ms: np.ndarray,
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, object]:
    all_idx = np.arange(len(features))
    result = train_head(features, label_names, duration_ms, all_idx, all_idx, args, logger, "final_all_data")
    return result


def predict_final(
    result: dict[str, object],
    features: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    device = choose_device(args.device)
    model: DynamicConsonantHead = result["model"]
    mean = result["feature_mean"][None, :]
    std = result["feature_std"][None, :]
    class_names = list(result["class_names"])
    x = ((features - mean) / std).astype(np.float32)
    y_dummy = np.zeros(len(x), dtype=np.int64)
    dur_dummy = np.zeros(len(x), dtype=np.float32)
    _, pred_ids, conf, dur = evaluate_model(
        model,
        x,
        y_dummy,
        dur_dummy,
        class_names,
        float(result["duration_mean"]),
        float(result["duration_std"]),
        device,
        args.batch_size,
    )
    return [class_names[int(idx)] for idx in pred_ids], conf, dur


def load_stub_for_subtype(subtype: str, stubs: dict[str, list[Path]], sample_rate: int) -> tuple[np.ndarray, Path]:
    key = subtype.strip().upper()
    if key not in stubs:
        raise KeyError(f"No stub for subtype {subtype}")
    path = stubs[key][0]
    return load_stub(path, sample_rate, target_peak=0.5), path


def replace_with_dynamic_stub(
    audio: np.ndarray,
    sample_rate: int,
    stub: np.ndarray,
    mask_duration_ms: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int]:
    start = find_energy_onset(audio, sample_rate)
    mask_len = round(sample_rate * mask_duration_ms / 1000.0)
    mask_len = max(1, min(mask_len, max(1, len(audio) - start)))
    end = min(len(audio), start + mask_len)
    source = audio[start:end]
    stub = prepare_stub_for_insert(stub, args)
    fitted = fit_stub_to_length(stub, max(1, end - start), args.stub_fit_mode)
    adapted = adapt_stub_level(fitted, source, args.level_mode, rms_ratio=args.stub_rms_ratio, max_stub_peak=0.9)
    before = audio[:start].astype(np.float32)
    tail = audio[end:].astype(np.float32)
    fade_len = min(round(sample_rate * args.crossfade_ms / 1000.0), max(1, len(adapted) // 2), max(1, len(tail)))
    body = crossfade_join(adapted, tail, fade_len)
    out = np.concatenate([before, body]).astype(np.float32)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.98:
        out = (out / peak * 0.98).astype(np.float32)
    return out, start, end, len(adapted)


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.outdir / "run.log")
    try:
        logger("Starting HuBERT dynamic B/P replacement")
        logger(f"Model: {args.model}")
        logger(f"Manifest: {args.manifest}")
        cache_dir = args.feature_cache_dir or (args.outdir / "feature_cache")
        items = select_rows(read_csv(args.manifest), args, cache_dir)
        if len(items) < 8:
            raise ValueError("Not enough usable B/P rows selected.")
        logger(f"Selected rows: {len(items)}; counts={label_counts([item.label for item in items])}")

        extracted = extract_features(items, args, logger)
        if extracted:
            write_csv(args.outdir / "newly_extracted_features.csv", extracted)
        if args.extract_only:
            logger("Extract-only mode finished.")
            return

        features, label_names, metadata = load_cached_matrix(items)
        logger(f"Loaded feature matrix: shape={features.shape}")

        pseudo_duration_rows = []
        pseudo_durations = []
        for item in items:
            row = item.row
            sample_rate, audio = read_wav_float(resolve_path(row["audio_path"]))
            label = row["label"].upper()
            min_ms = args.mask_min_ms + (10.0 if label == "P" else 0.0)
            max_ms = args.mask_max_ms
            duration_ms, details = estimate_vowel_transition_ms(audio, sample_rate, label, min_ms, max_ms)
            pseudo_durations.append(duration_ms)
            pseudo_duration_rows.append(
                {
                    "speaker": row.get("speaker", ""),
                    "label": label,
                    "word": row.get("word", ""),
                    "pseudo_duration_ms": round(duration_ms, 2),
                    "fallback_used": int(details.get("fallback", 0.0)),
                    "onset_sec": round(details.get("onset_sample", 0.0) / sample_rate, 4),
                    "audio_path": row["audio_path"],
                }
            )
        duration_ms = np.asarray(pseudo_durations, dtype=np.float32)
        write_csv(args.outdir / "duration_pseudo_labels.csv", pseudo_duration_rows)

        y, _ = encode_labels(label_names)
        rng = np.random.default_rng(args.seed)
        train_idx, val_idx = stratified_split(y, args.val_ratio, rng)
        eval_result = train_head(features, label_names, duration_ms, train_idx, val_idx, args, logger, "random_validation")
        final_result = train_final_head(features, label_names, duration_ms, args, logger)

        torch.save(final_result["state_dict"], args.outdir / "dynamic_head_model_state.pt")
        np.savez_compressed(
            args.outdir / "dynamic_head_stats.npz",
            feature_mean=final_result["feature_mean"],
            feature_std=final_result["feature_std"],
            duration_mean=np.array([final_result["duration_mean"]], dtype=np.float32),
            duration_std=np.array([final_result["duration_std"]], dtype=np.float32),
            class_names=np.array(final_result["class_names"]),
        )
        write_csv(args.outdir / "training_history_random_validation.csv", eval_result["history"])
        write_csv(args.outdir / "training_history_final_all_data.csv", final_result["history"])

        predicted_labels, confidence, model_duration_ms = predict_final(final_result, features, args)
        b_stubs = list_stubs_normalized(args.b_stub_root)
        p_stubs = list_stubs_normalized(args.p_stub_root)
        if not b_stubs or not p_stubs:
            raise ValueError("Both B and P stub libraries must contain WAV files.")

        first_rate, _ = read_wav_float(resolve_path(metadata[0]["audio_path"]))
        sequence_gap = np.zeros(round(first_rate * args.sequence_gap_ms / 1000.0), dtype=np.float32)
        ab_gap = np.zeros(round(first_rate * 0.45), dtype=np.float32)
        long_gap = np.zeros(round(first_rate * 0.85), dtype=np.float32)
        original_parts = []
        enhanced_parts = []
        combined_ab_parts = []
        render_rows = []

        for idx, (row, predicted_label, conf, model_ms, energy_ms) in enumerate(
            zip(metadata, predicted_labels, confidence, model_duration_ms, duration_ms),
            start=1,
        ):
            sample_rate, audio = read_wav_float(resolve_path(row["audio_path"]))
            audio = audio.astype(np.float32)
            true_label = row["label"].upper()
            word = row["word"].lower()
            true_subtype = subtype_for_word(true_label, word)
            available = set(b_stubs if predicted_label == "B" else p_stubs)
            stubs = b_stubs if predicted_label == "B" else p_stubs
            predicted_subtype = true_subtype if true_subtype.startswith(f"{predicted_label}_") else convert_subtype(
                true_subtype, predicted_label, available
            )
            energy_for_pred, _ = estimate_vowel_transition_ms(
                audio,
                sample_rate,
                predicted_label,
                args.mask_min_ms + (10.0 if predicted_label == "P" else 0.0),
                args.mask_max_ms,
            )
            model_ms = float(np.clip(model_ms, args.mask_min_ms, args.mask_max_ms))
            if predicted_label == "P":
                model_ms += args.p_extra_ms
            blend = float(np.clip(args.duration_blend_model, 0.0, 1.0))
            final_ms = blend * model_ms + (1.0 - blend) * energy_for_pred
            final_ms = float(np.clip(final_ms, args.mask_min_ms, args.mask_max_ms))
            should_replace = conf >= args.confidence_threshold and predicted_subtype in available

            if should_replace:
                stub, stub_path = load_stub_for_subtype(predicted_subtype, stubs, sample_rate)
                enhanced, start, end, fitted_len = replace_with_dynamic_stub(audio, sample_rate, stub, final_ms, args)
            else:
                stub_path = Path("")
                enhanced = audio.copy()
                start = end = fitted_len = 0

            rendered_original = trim_word_boundaries(audio, sample_rate, args)
            rendered_enhanced = trim_word_boundaries(enhanced, sample_rate, args)
            ab_audio = np.concatenate(
                [
                    rendered_original,
                    np.zeros(round(sample_rate * 0.45), dtype=np.float32),
                    rendered_enhanced,
                ]
            ).astype(np.float32)

            stem = (
                f"{idx:03d}_{row.get('word_index', row.get('file_index', ''))}_{word}"
                f"_true-{true_label}_{true_subtype}_pred-{predicted_label}_{predicted_subtype}"
                f"_dur{final_ms:.0f}ms_conf{conf:.2f}"
            )
            original_path = args.outdir / "original" / f"{stem}_original.wav"
            enhanced_path = args.outdir / "enhanced" / f"{stem}_enhanced.wav"
            ab_path = args.outdir / "ab" / f"{stem}_A_original_B_enhanced.wav"
            write_wav_float(original_path, sample_rate, rendered_original)
            write_wav_float(enhanced_path, sample_rate, rendered_enhanced)
            write_wav_float(ab_path, sample_rate, ab_audio)

            original_parts.extend([resample_to(rendered_original, sample_rate, first_rate), sequence_gap])
            enhanced_parts.extend([resample_to(rendered_enhanced, sample_rate, first_rate), sequence_gap])
            combined_ab_parts.extend([resample_to(ab_audio, sample_rate, first_rate), long_gap])

            render_rows.append(
                {
                    "index": idx,
                    "speaker": row.get("speaker", ""),
                    "word": word,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": round(float(conf), 4),
                    "bp_correct": str(predicted_label == true_label).lower(),
                    "true_subtype": true_subtype,
                    "predicted_subtype": predicted_subtype,
                    "replacement_applied": str(should_replace).lower(),
                    "energy_pseudo_duration_ms": round(float(energy_ms), 2),
                    "energy_runtime_duration_ms": round(float(energy_for_pred), 2),
                    "model_duration_ms": round(float(model_ms), 2),
                    "final_mask_duration_ms": round(float(final_ms), 2),
                    "mask_start_sec": round(start / sample_rate, 4) if should_replace else "",
                    "mask_end_sec": round(end / sample_rate, 4) if should_replace else "",
                    "fitted_stub_duration_ms": round(fitted_len / sample_rate * 1000.0, 2) if should_replace else "",
                    "stub_path": str(stub_path),
                    "source_audio": row["audio_path"],
                    "original_audio": str(original_path),
                    "enhanced_audio": str(enhanced_path),
                    "ab_audio": str(ab_path),
                }
            )

        suffix = tempo_suffix(args.sequence_tempo, args.sequence_time_mode)
        original_sequence_path = args.outdir / f"dave_hubert_dynamic_original_sequence{suffix}.wav"
        enhanced_sequence_path = args.outdir / f"dave_hubert_dynamic_enhanced_sequence{suffix}.wav"
        ab_sequence_path = args.outdir / f"dave_hubert_dynamic_sequence_A_original_B_enhanced{suffix}.wav"
        combined_ab_path = args.outdir / "dave_hubert_dynamic_all_A_original_B_enhanced.wav"

        original_sequence = np.concatenate(original_parts[:-1]).astype(np.float32)
        enhanced_sequence = np.concatenate(enhanced_parts[:-1]).astype(np.float32)
        combined_ab = np.concatenate(combined_ab_parts[:-1]).astype(np.float32)
        write_sequence_with_tempo(original_sequence_path, first_rate, original_sequence, args.sequence_tempo, args.sequence_time_mode)
        write_sequence_with_tempo(enhanced_sequence_path, first_rate, enhanced_sequence, args.sequence_tempo, args.sequence_time_mode)
        write_wav_float(combined_ab_path, first_rate, combined_ab)

        sr_o, original_sequence_loaded = read_wav_float(original_sequence_path)
        sr_e, enhanced_sequence_loaded = read_wav_float(enhanced_sequence_path)
        if sr_o != sr_e:
            enhanced_sequence_loaded = resample_to(enhanced_sequence_loaded, sr_e, sr_o)
        write_wav_float(
            ab_sequence_path,
            sr_o,
            np.concatenate(
                [
                    original_sequence_loaded.astype(np.float32),
                    np.zeros(round(sr_o * 1.2), dtype=np.float32),
                    enhanced_sequence_loaded.astype(np.float32),
                ]
            ),
        )

        render_manifest = args.outdir / "dynamic_replacement_manifest.csv"
        write_csv(render_manifest, render_rows)
        bp_accuracy = float(np.mean([row["bp_correct"] == "true" for row in render_rows]))
        summary = {
            "method": "HuBERT frozen feature extractor + small multi-task head for B/P and pseudo duration",
            "important_note": (
                "Duration labels are pseudo-labels estimated from onset energy and low-frequency vowel transition. "
                "This is a practical first version, not manually annotated phoneme-boundary ground truth."
            ),
            "model": args.model,
            "feature_window_ms": args.window_ms,
            "onset_pool_ms": args.onset_ms,
            "selected_rows": len(items),
            "label_counts": label_counts(label_names),
            "random_validation_metrics": eval_result["metrics"],
            "final_in_sample_bp_accuracy_for_rendering": round(bp_accuracy, 4),
            "mean_final_mask_duration_ms": round(float(np.mean([row["final_mask_duration_ms"] for row in render_rows])), 2),
            "replacement_applied_count": int(sum(row["replacement_applied"] == "true" for row in render_rows)),
            "render_manifest": str(render_manifest),
            "original_sequence_audio": str(original_sequence_path),
            "enhanced_sequence_audio": str(enhanced_sequence_path),
            "sequence_ab_audio": str(ab_sequence_path),
            "combined_ab_audio": str(combined_ab_path),
            "model_state": str(args.outdir / "dynamic_head_model_state.pt"),
            "model_stats": str(args.outdir / "dynamic_head_stats.npz"),
        }
        (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger(f"Final summary: {json.dumps(summary, indent=2)}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
