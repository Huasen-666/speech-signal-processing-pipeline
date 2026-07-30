from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import numpy as np

from speech_pipeline.ml_features import extract_feature_set


WINDOWS_MS = [80, 100, 150, 200, 300, 500, None]
FEATURE_SETS = ["onset_only", "bp_onset", "mfcc_logmel_onset"]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV: {path}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def find_energy_onset(audio: np.ndarray, sample_rate: int, frame_ms: float = 8.0, hop_ms: float = 2.0) -> int:
    frame = max(1, round(sample_rate * frame_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame:
        return 0

    starts = np.arange(0, len(audio) - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(audio[start : start + frame] ** 2) + 1e-12) for start in starts])
    db = 20.0 * np.log10(rms + 1e-12)
    threshold = float(db.max() - 25.0)
    active = np.flatnonzero(db >= threshold)
    return int(starts[active[0]]) if len(active) else 0


def load_manifest(manifest: Path, repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("usable", "").strip().lower() != "true":
                continue
            if row.get("label", "").strip().upper() not in {"B", "P"}:
                continue
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = repo_root / audio_path
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            row["audio_path_abs"] = str(audio_path)
            rows.append(row)
    return rows


def clip_window(audio: np.ndarray, sample_rate: int, onset: int, window_ms: int | None) -> np.ndarray:
    if window_ms is None:
        return audio
    n_samples = max(1, round(sample_rate * window_ms / 1000.0))
    segment = audio[onset : onset + n_samples]
    min_len = round(sample_rate * 30.0 / 1000.0)
    if len(segment) < min_len:
        segment = np.pad(segment, (0, min_len - len(segment)))
    return segment


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


def predict_proba(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((len(x), 1)), x])
    logits = np.clip(x_aug @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def make_word_group_folds(labels: np.ndarray, words: np.ndarray, folds: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids: dict[tuple[int, str], int] = {}
    for label in sorted(set(labels.tolist())):
        label_words = sorted(set(words[labels == label].tolist()))
        rng.shuffle(label_words)
        for idx, word in enumerate(label_words):
            fold_ids[(int(label), word)] = idx % folds
    return np.array([fold_ids[(int(labels[idx]), str(words[idx]))] for idx in range(len(labels))])


def make_take_split(rows: list[dict[str, str]], train_take: int) -> tuple[np.ndarray, np.ndarray]:
    train = []
    test = []
    for idx, row in enumerate(rows):
        file_index = int(row.get("file_index") or row.get("protocol_index") or "0")
        take = 1 if file_index <= 50 else 2
        if take == train_take:
            train.append(idx)
        else:
            test.append(idx)
    return np.array(train, dtype=int), np.array(test, dtype=int)


def evaluate_folds(x: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> float:
    best_accuracy = -1.0
    for l2 in [0.01, 0.03, 0.1, 0.3, 1.0]:
        pred = np.zeros(len(y), dtype=int)
        for fold in sorted(set(fold_ids.tolist())):
            train_idx = fold_ids != fold
            test_idx = fold_ids == fold
            train_x, test_x = standardize(x[train_idx], x[test_idx])
            weights = fit_logistic_newton(train_x, y[train_idx].astype(np.float64), l2=l2)
            prob = predict_proba(weights, test_x)
            pred[test_idx] = (prob >= 0.5).astype(int)
        accuracy = float((pred == y).mean())
        best_accuracy = max(best_accuracy, accuracy)
    return best_accuracy


def evaluate_take_split(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    best_accuracy = -1.0
    for l2 in [0.01, 0.03, 0.1, 0.3, 1.0]:
        train_x, test_x = standardize(x[train_idx], x[test_idx])
        weights = fit_logistic_newton(train_x, y[train_idx].astype(np.float64), l2=l2)
        prob = predict_proba(weights, test_x)
        pred = (prob >= 0.5).astype(int)
        accuracy = float((pred == y[test_idx]).mean())
        best_accuracy = max(best_accuracy, accuracy)
    return best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B/P accuracy using only the first N ms from speech onset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/david_bp_dataset_manifest.csv"),
        help="CSV manifest with B/P word clips.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/per_patient_bp_eval/david_onset_window_results.csv"),
        help="Where to save the result table.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    out_path = args.out if args.out.is_absolute() else repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest, repo_root)

    labels = np.array([1 if row["label"].strip().upper() == "P" else 0 for row in rows], dtype=int)
    words = np.array([row["word"].strip().lower() for row in rows])
    word_folds = make_word_group_folds(labels, words)
    take1_train, take2_test = make_take_split(rows, train_take=1)
    take2_train, take1_test = make_take_split(rows, train_take=2)

    audio_cache: list[tuple[np.ndarray, int, int]] = []
    for row in rows:
        audio, sample_rate = read_wav(Path(row["audio_path_abs"]))
        onset = find_energy_onset(audio, sample_rate)
        audio_cache.append((audio, sample_rate, onset))

    print(f"Loaded {len(rows)} clips: {(labels == 0).sum()} B / {(labels == 1).sum()} P")
    print("Accuracy: word-grouped CV | take1->take2 | take2->take1")
    print()

    result_rows = []
    for feature_set in FEATURE_SETS:
        print(f"[{feature_set}]")
        for window_ms in WINDOWS_MS:
            features = []
            for audio, sample_rate, onset in audio_cache:
                segment = clip_window(audio, sample_rate, onset, window_ms)
                features.append(extract_feature_set(segment, sample_rate, feature_set))
            x = np.vstack(features)
            word_acc = evaluate_folds(x, labels, word_folds)
            forward_acc = evaluate_take_split(x, labels, take1_train, take2_test)
            reverse_acc = evaluate_take_split(x, labels, take2_train, take1_test)
            label = "full word" if window_ms is None else f"{window_ms:>3d} ms"
            print(f"  {label:9s}: {word_acc:.3f} | {forward_acc:.3f} | {reverse_acc:.3f}")
            result_rows.append(
                {
                    "feature_set": feature_set,
                    "window_ms": "full_word" if window_ms is None else str(window_ms),
                    "word_grouped_cv": f"{word_acc:.3f}",
                    "take1_to_take2": f"{forward_acc:.3f}",
                    "take2_to_take1": f"{reverse_acc:.3f}",
                }
            )
        print()

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_set", "window_ms", "word_grouped_cv", "take1_to_take2", "take2_to_take1"],
        )
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
