import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.ml_features import extract_feature_set, mel_filterbank, preemphasis, resample_linear, trim_by_energy


def parse_args() -> argparse.Namespace:
    dave_dir = PROJECT_ROOT.parent / "DAVE"
    parser = argparse.ArgumentParser(
        description="Render Dave-style consonant stub replacement demos for B/P word clips."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("experiments/listening_demos/dave_stub_replacement"))
    parser.add_argument("--b-stub", type=Path, default=dave_dir / "B consonant DRB 1.wav")
    parser.add_argument("--p-stub", type=Path, default=dave_dir / "P consonant DRB 1.wav")
    parser.add_argument("--include-speakers", nargs="*", default=["bascom", "corrick"])
    parser.add_argument("--clips-per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-source",
        choices=["true", "model"],
        default="true",
        help="Use manifest labels, or use a trained MLP to choose B/P stubs.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("experiments/ml_baseline/bp_tinycnn_logmel_mfcc_all_usable"),
        help="Used only when --label-source model.",
    )
    parser.add_argument(
        "--model-type",
        choices=["mlp", "tinycnn"],
        default="tinycnn",
        help="Classifier type to use when --label-source model.",
    )
    parser.add_argument(
        "--level-mode",
        choices=["dave_peak", "match_rms"],
        default="match_rms",
        help="dave_peak normalizes stubs to peak 0.5 like Dave's C arrays; match_rms adapts to each word.",
    )
    parser.add_argument("--dave-target-peak", type=float, default=0.5)
    parser.add_argument("--rms-ratio", type=float, default=1.0)
    parser.add_argument("--max-stub-peak", type=float, default=0.9)
    parser.add_argument("--crossfade-ms", type=float, default=25.0)
    parser.add_argument("--pre-roll-ms", type=float, default=20.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_demo_rows(
    rows: list[dict[str, str]],
    include_speakers: list[str],
    clips_per_label: int,
    seed: int,
) -> list[dict[str, str]]:
    allowed_speakers = {speaker.lower() for speaker in include_speakers}
    rng = np.random.default_rng(seed)
    selected = []
    for label in ["B", "P"]:
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


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def load_mlp_model(model_dir: Path) -> dict:
    report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    model = json.loads((model_dir / "model_weights.json").read_text(encoding="utf-8"))
    normalization = json.loads((model_dir / "feature_normalization.json").read_text(encoding="utf-8"))
    return {
        "classes": [label.upper() for label in report["classes"]],
        "feature_set": report["feature_set"],
        "weights": [np.array(item, dtype=np.float64) for item in model["weights"]],
        "biases": [np.array(item, dtype=np.float64) for item in model["biases"]],
        "mean": np.array(normalization["mean"], dtype=np.float64),
        "std": np.array(normalization["std"], dtype=np.float64),
    }


def predict_mlp_label(audio: np.ndarray, sample_rate: int, mlp: dict) -> tuple[str, float]:
    feature = extract_feature_set(audio, sample_rate, mlp["feature_set"])
    safe_std = np.where(mlp["std"] < 1e-8, 1.0, mlp["std"])
    feature = ((feature - mlp["mean"]) / safe_std)[None, :]
    hidden = np.maximum(0.0, feature @ mlp["weights"][0] + mlp["biases"][0])
    probs = softmax(hidden @ mlp["weights"][1] + mlp["biases"][1])[0]
    pred_idx = int(np.argmax(probs))
    return mlp["classes"][pred_idx], float(probs[pred_idx])


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
        active = trim_by_energy(
            active,
            frame_size=round(target_sample_rate * 0.02),
            hop_size=round(target_sample_rate * 0.005),
        )

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


def load_tinycnn_model(model_dir: Path) -> dict:
    report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    classes = [label.upper() for label in report["architecture"]["classes"]]
    input_shape = report["architecture"]["input_shape"]
    model = TinyLogMelCNN(in_channels=int(input_shape[0]), class_count=len(classes))
    state = torch.load(model_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return {
        "classes": classes,
        "model": model,
        "image_mean": float(report["image_mean"]),
        "image_std": float(report["image_std"]) if abs(float(report["image_std"])) > 1e-8 else 1.0,
        "target_sample_rate": int(report["target_sample_rate"]),
        "duration_ms": float(report["duration_ms"]),
        "n_mels": int(report["n_mels"]),
        "n_mfcc": int(report["n_mfcc"]),
    }


def predict_tinycnn_label(audio: np.ndarray, sample_rate: int, tinycnn: dict) -> tuple[str, float]:
    image = logmel_image(
        audio,
        sample_rate,
        tinycnn["target_sample_rate"],
        tinycnn["duration_ms"],
        tinycnn["n_mels"],
        tinycnn["n_mfcc"],
    )
    image = (image - tinycnn["image_mean"]) / tinycnn["image_std"]
    tensor = torch.tensor(image[None, :, :, :], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.softmax(tinycnn["model"](tensor), dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return tinycnn["classes"][pred_idx], float(probs[pred_idx])


def resample_to(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(audio.astype(np.float64), up, down).astype(np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-12:
        return audio.astype(np.float32)
    return (audio / peak * target_peak).astype(np.float32)


def load_stub(path: Path, target_rate: int, target_peak: float) -> np.ndarray:
    stub_rate, stub = read_wav_float(path)
    stub = stub.astype(np.float64)
    stub = stub - float(np.mean(stub))
    stub = resample_to(stub, stub_rate, target_rate)
    return normalize_peak(stub, target_peak)


def frame_rms_db(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        rms = np.array([float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0])
        return np.array([0]), 20.0 * np.log10(np.maximum(rms, 1e-8))

    starts = []
    values = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        starts.append(start)
        values.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
    return np.array(starts), 20.0 * np.log10(np.maximum(values, 1e-8))


def detect_active_start(audio: np.ndarray, sample_rate: int, pre_roll_ms: float) -> int:
    starts, rms_db = frame_rms_db(audio, sample_rate)
    if len(rms_db) == 0:
        return 0
    noise_floor = float(np.percentile(rms_db, 10))
    high_energy = float(np.percentile(rms_db, 95))
    threshold = max(noise_floor + 10.0, high_energy - 30.0)
    threshold = min(max(threshold, -65.0), -30.0)
    active = np.flatnonzero(rms_db >= threshold)
    if not len(active):
        return 0
    pre_roll = round(sample_rate * pre_roll_ms / 1000.0)
    return max(0, int(starts[active[0]]) - pre_roll)


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def adapt_stub_level(
    stub: np.ndarray,
    source_segment: np.ndarray,
    level_mode: str,
    rms_ratio: float,
    max_stub_peak: float,
) -> np.ndarray:
    if level_mode == "dave_peak":
        adapted = stub.astype(np.float64)
    else:
        source_rms = rms(source_segment)
        stub_rms = rms(stub)
        if source_rms > 1e-8 and stub_rms > 1e-8:
            adapted = stub.astype(np.float64) * (source_rms * rms_ratio / stub_rms)
        else:
            adapted = stub.astype(np.float64)

    peak = float(np.max(np.abs(adapted))) if len(adapted) else 0.0
    if peak > max_stub_peak:
        adapted = adapted / peak * max_stub_peak
    return adapted.astype(np.float32)


def crossfade_join(left: np.ndarray, right: np.ndarray, fade_len: int) -> np.ndarray:
    if fade_len <= 0 or len(left) < fade_len or len(right) < fade_len:
        return np.concatenate([left, right]).astype(np.float32)

    fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=False)
    fade_in = 1.0 - fade_out
    overlap = left[-fade_len:].astype(np.float64) * fade_out + right[:fade_len].astype(np.float64) * fade_in
    return np.concatenate([left[:-fade_len], overlap.astype(np.float32), right[fade_len:]]).astype(np.float32)


def replace_consonant_with_stub(
    audio: np.ndarray,
    sample_rate: int,
    stub: np.ndarray,
    level_mode: str,
    rms_ratio: float,
    max_stub_peak: float,
    crossfade_ms: float,
    pre_roll_ms: float,
) -> tuple[np.ndarray, int, int]:
    active_start = detect_active_start(audio, sample_rate, pre_roll_ms)
    stub_len = len(stub)
    source_end = min(len(audio), active_start + stub_len)
    source_segment = audio[active_start:source_end]
    adapted_stub = adapt_stub_level(stub, source_segment, level_mode, rms_ratio, max_stub_peak)

    before = audio[:active_start].astype(np.float32)
    tail = audio[source_end:].astype(np.float32)
    fade_len = round(sample_rate * crossfade_ms / 1000.0)
    replaced_body = crossfade_join(adapted_stub, tail, fade_len)
    replaced = np.concatenate([before, replaced_body]).astype(np.float32)

    peak = float(np.max(np.abs(replaced))) if len(replaced) else 0.0
    if peak > 0.98:
        replaced = (replaced / peak * 0.98).astype(np.float32)
    return replaced, active_start, source_end


def write_demo_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_audio",
        "speaker",
        "word",
        "true_label",
        "control_label",
        "confidence",
        "active_start_sec",
        "source_replace_end_sec",
        "stub_duration_sec",
        "original_audio",
        "stub_replaced_audio",
        "ab_audio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = select_demo_rows(read_manifest(args.manifest), args.include_speakers, args.clips_per_label, args.seed)
    if not rows:
        raise ValueError("No usable B/P rows selected.")

    classifier = None
    if args.label_source == "model":
        if args.model_type == "mlp":
            classifier = load_mlp_model(args.model_dir)
        else:
            classifier = load_tinycnn_model(args.model_dir)

    args.outdir.mkdir(parents=True, exist_ok=True)
    first_rate, _ = read_wav_float(rows[0]["audio_path"])
    stubs = {
        "B": load_stub(args.b_stub, first_rate, args.dave_target_peak),
        "P": load_stub(args.p_stub, first_rate, args.dave_target_peak),
    }

    write_wav_float(args.outdir / "stub_B_resampled_normalized.wav", first_rate, stubs["B"])
    write_wav_float(args.outdir / "stub_P_resampled_normalized.wav", first_rate, stubs["P"])

    demo_rows = []
    combined_parts = []
    silence = np.zeros(round(first_rate * 0.45), dtype=np.float32)
    long_silence = np.zeros(round(first_rate * 0.8), dtype=np.float32)

    for idx, row in enumerate(rows, start=1):
        sample_rate, audio = read_wav_float(row["audio_path"])
        if sample_rate != first_rate:
            stubs_for_rate = {
                "B": load_stub(args.b_stub, sample_rate, args.dave_target_peak),
                "P": load_stub(args.p_stub, sample_rate, args.dave_target_peak),
            }
            silence_for_rate = np.zeros(round(sample_rate * 0.45), dtype=np.float32)
            long_silence_for_rate = np.zeros(round(sample_rate * 0.8), dtype=np.float32)
        else:
            stubs_for_rate = stubs
            silence_for_rate = silence
            long_silence_for_rate = long_silence

        if classifier is None:
            control_label = row["label"].upper()
            confidence = 1.0
        elif args.model_type == "mlp":
            control_label, confidence = predict_mlp_label(audio, sample_rate, classifier)
        else:
            control_label, confidence = predict_tinycnn_label(audio, sample_rate, classifier)

        replaced, active_start, source_end = replace_consonant_with_stub(
            audio,
            sample_rate,
            stubs_for_rate[control_label],
            args.level_mode,
            args.rms_ratio,
            args.max_stub_peak,
            args.crossfade_ms,
            args.pre_roll_ms,
        )
        ab_audio = np.concatenate([audio.astype(np.float32), silence_for_rate, replaced])

        stem = (
            f"{idx:02d}_{row['speaker']}_{row['label']}_{row['word']}"
            f"_stub-{control_label}_{args.model_type}_{args.level_mode}"
        )
        original_path = args.outdir / f"{stem}_original.wav"
        replaced_path = args.outdir / f"{stem}_stub_replaced.wav"
        ab_path = args.outdir / f"{stem}_A_original_B_stub_replaced.wav"

        write_wav_float(original_path, sample_rate, audio)
        write_wav_float(replaced_path, sample_rate, replaced)
        write_wav_float(ab_path, sample_rate, ab_audio)

        combined_parts.extend([ab_audio, long_silence_for_rate])
        demo_rows.append(
            {
                "source_audio": row["audio_path"],
                "speaker": row["speaker"],
                "word": row["word"],
                "true_label": row["label"],
                "control_label": control_label,
                "confidence": f"{confidence:.4f}",
                "active_start_sec": f"{active_start / sample_rate:.4f}",
                "source_replace_end_sec": f"{source_end / sample_rate:.4f}",
                "stub_duration_sec": f"{len(stubs_for_rate[control_label]) / sample_rate:.4f}",
                "original_audio": str(original_path),
                "stub_replaced_audio": str(replaced_path),
                "ab_audio": str(ab_path),
            }
        )

    combined_path = args.outdir / (
        f"dave_stub_replacement_all_A_original_B_stub_replaced_{args.model_type}_{args.level_mode}.wav"
    )
    write_wav_float(combined_path, first_rate, np.concatenate(combined_parts))
    manifest_path = args.outdir / "demo_manifest.csv"
    write_demo_manifest(manifest_path, demo_rows)

    print(f"Saved demo manifest: {manifest_path}")
    print(f"Saved combined A/B demo: {combined_path}")
    print(f"Saved normalized B stub: {args.outdir / 'stub_B_resampled_normalized.wav'}")
    print(f"Saved normalized P stub: {args.outdir / 'stub_P_resampled_normalized.wav'}")
    for row in demo_rows:
        print(
            f"{row['speaker']} {row['word']} true={row['true_label']} "
            f"stub={row['control_label']} conf={row['confidence']} "
            f"replace={row['active_start_sec']}..{row['source_replace_end_sec']}s"
        )


if __name__ == "__main__":
    main()
