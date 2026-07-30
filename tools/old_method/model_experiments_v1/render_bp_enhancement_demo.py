import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.ml_features import extract_feature_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render original/enhanced B/P listening demos from a trained MLP.")
    parser.add_argument("--model-dir", type=Path, default=Path("experiments/ml_baseline/bp_mlp_v2_mfcc_logmel_onset_all_usable"))
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/listening_demos/bp_v2_onset_enhancer"))
    parser.add_argument("--include-speakers", nargs="*", default=["bascom", "corrick"])
    parser.add_argument("--clips-per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--onset-ms", type=float, default=160.0)
    parser.add_argument("--p-high-gain", type=float, default=0.85)
    parser.add_argument("--p-low-cut", type=float, default=0.10)
    parser.add_argument("--b-low-gain", type=float, default=0.45)
    parser.add_argument("--b-high-gain", type=float, default=0.20)
    parser.add_argument("--diff-gain", type=float, default=3.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def predict_proba(features: np.ndarray, model: dict) -> np.ndarray:
    weights = [np.array(item, dtype=np.float64) for item in model["weights"]]
    biases = [np.array(item, dtype=np.float64) for item in model["biases"]]
    hidden = np.maximum(0.0, features @ weights[0] + biases[0])
    return softmax(hidden @ weights[1] + biases[1])


def standardize_feature(feature: np.ndarray, normalization: dict) -> np.ndarray:
    mean = np.array(normalization["mean"], dtype=np.float64)
    std = np.array(normalization["std"], dtype=np.float64)
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (feature - mean) / safe_std


def select_demo_rows(
    rows: list[dict[str, str]],
    classes: list[str],
    include_speakers: list[str],
    clips_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    allowed_speakers = {speaker.lower() for speaker in include_speakers}
    rng = np.random.default_rng(seed)
    selected = []
    for label in classes:
        candidates = [
            row
            for row in rows
            if row["usable"].lower() == "true"
            and row["label"].upper() == label
            and row["speaker"].lower() in allowed_speakers
        ]
        candidates = sorted(candidates, key=lambda row: (row["speaker"], row["protocol_index"], row["word"]))
        if len(candidates) > clips_per_label:
            indices = sorted(rng.choice(len(candidates), size=clips_per_label, replace=False).tolist())
            candidates = [candidates[idx] for idx in indices]
        selected.extend(candidates)
    return selected


def detect_active_start(audio: np.ndarray, sample_rate: int) -> int:
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return 0
    energies = []
    starts = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        energies.append(float(np.mean(frame.astype(np.float64) ** 2)))
        starts.append(start)
    if not energies or max(energies) <= 1e-12:
        return 0
    threshold = max(energies) * 0.04
    for start, energy in zip(starts, energies):
        if energy >= threshold:
            return max(0, start - round(sample_rate * 0.02))
    return 0


def safe_sosfiltfilt(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if len(audio) < 32:
        return audio.astype(np.float64)
    return sosfiltfilt(sos, audio.astype(np.float64))


def fade_envelope(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float64)
    env = np.hanning(length * 2)[:length]
    return np.maximum(env, 0.15)


def enhance_prediction_controlled(
    audio: np.ndarray,
    sample_rate: int,
    predicted_label: str,
    onset_ms: float,
    p_high_gain: float,
    p_low_cut: float,
    b_low_gain: float,
    b_high_gain: float,
) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float64)
    if len(mono) == 0:
        return mono

    start = detect_active_start(mono, sample_rate)
    end = min(len(mono), start + round(sample_rate * onset_ms / 1000.0))
    if end <= start:
        return mono

    high_sos = butter(2, 1200.0, btype="highpass", fs=sample_rate, output="sos")
    low_sos = butter(2, [80.0, 600.0], btype="bandpass", fs=sample_rate, output="sos")
    high = safe_sosfiltfilt(high_sos, mono)
    low = safe_sosfiltfilt(low_sos, mono)
    env = fade_envelope(end - start)

    enhanced = mono.copy()
    if predicted_label == "P":
        enhanced[start:end] += p_high_gain * env * high[start:end]
        enhanced[start:end] -= p_low_cut * env * low[start:end]
    else:
        enhanced[start:end] += b_low_gain * env * low[start:end]
        enhanced[start:end] += b_high_gain * env * high[start:end]

    peak = float(np.max(np.abs(enhanced))) if len(enhanced) else 0.0
    if peak > 0.98:
        enhanced = enhanced / peak * 0.98
    return enhanced.astype(np.float32)


def difference_audio(original: np.ndarray, enhanced: np.ndarray, gain: float) -> np.ndarray:
    diff = (np.asarray(enhanced, dtype=np.float64) - np.asarray(original, dtype=np.float64)) * gain
    peak = float(np.max(np.abs(diff))) if len(diff) else 0.0
    if peak > 0.98:
        diff = diff / peak * 0.98
    return diff.astype(np.float32)


def write_demo_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_audio",
        "speaker",
        "word",
        "true_label",
        "predicted_label",
        "confidence",
        "original_audio",
        "enhanced_audio",
        "difference_audio",
        "ab_audio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report = json.loads((args.model_dir / "training_report.json").read_text(encoding="utf-8"))
    model = json.loads((args.model_dir / "model_weights.json").read_text(encoding="utf-8"))
    normalization = json.loads((args.model_dir / "feature_normalization.json").read_text(encoding="utf-8"))
    classes = [label.upper() for label in report["classes"]]
    feature_set = report["feature_set"]

    rows = select_demo_rows(read_manifest(args.manifest), classes, args.include_speakers, args.clips_per_label, args.seed)
    if not rows:
        raise ValueError("No demo rows selected.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    silence_cache: dict[int, np.ndarray] = {}
    combined_parts = []
    demo_rows = []

    for idx, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        feature = extract_feature_set(audio, sample_rate, feature_set)
        feature_n = standardize_feature(feature, normalization)[None, :]
        probs = predict_proba(feature_n, model)[0]
        pred_idx = int(np.argmax(probs))
        predicted_label = classes[pred_idx]
        confidence = float(probs[pred_idx])

        enhanced = enhance_prediction_controlled(
            audio,
            sample_rate,
            predicted_label,
            args.onset_ms,
            args.p_high_gain,
            args.p_low_cut,
            args.b_low_gain,
            args.b_high_gain,
        )
        diff = difference_audio(audio, enhanced, args.diff_gain)
        silence = silence_cache.setdefault(sample_rate, np.zeros(round(sample_rate * 0.35), dtype=np.float32))
        ab_audio = np.concatenate([audio.astype(np.float32), silence, enhanced])

        stem = f"{idx:02d}_{row['speaker']}_{row['label']}_{row['word']}_pred-{predicted_label}_{confidence:.2f}"
        original_path = args.outdir / f"{stem}_original.wav"
        enhanced_path = args.outdir / f"{stem}_enhanced.wav"
        diff_path = args.outdir / f"{stem}_difference_x{args.diff_gain:.1f}.wav"
        ab_path = args.outdir / f"{stem}_A_original_B_enhanced.wav"
        write_wav_float(original_path, sample_rate, audio)
        write_wav_float(enhanced_path, sample_rate, enhanced)
        write_wav_float(diff_path, sample_rate, diff)
        write_wav_float(ab_path, sample_rate, ab_audio)

        combined_parts.extend([ab_audio, np.zeros(round(sample_rate * 0.7), dtype=np.float32)])
        demo_rows.append(
            {
                "source_audio": row["audio_path"],
                "speaker": row["speaker"],
                "word": row["word"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "confidence": f"{confidence:.4f}",
                "original_audio": str(original_path),
                "enhanced_audio": str(enhanced_path),
                "difference_audio": str(diff_path),
                "ab_audio": str(ab_path),
            }
        )

    combined_path = args.outdir / "bp_v2_demo_all_A_original_B_enhanced.wav"
    write_wav_float(combined_path, sample_rate, np.concatenate(combined_parts))
    manifest_path = args.outdir / "demo_manifest.csv"
    write_demo_manifest(manifest_path, demo_rows)

    print(f"Saved demo manifest: {manifest_path}")
    print(f"Saved combined A/B demo: {combined_path}")
    for row in demo_rows:
        print(
            f"{row['speaker']} {row['word']} true={row['true_label']} "
            f"pred={row['predicted_label']} conf={row['confidence']}"
        )


if __name__ == "__main__":
    main()
