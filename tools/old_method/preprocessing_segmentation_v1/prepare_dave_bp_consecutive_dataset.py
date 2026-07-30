import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments

from build_dataset_manifest import B_WORDS, P_WORDS


SOURCE_AUDIO = (
    PROJECT_ROOT.parent
    / "DAVE"
    / "new"
    / "Sound files for Huasen  B and P"
    / "b words p words consecutively.wav"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "dave_bp_consecutive" / "Dave"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "dave_bp_consecutive_manifest.csv"

MANIFEST_FIELDS = [
    "speaker",
    "label",
    "word_index",
    "global_index",
    "word",
    "audio_path",
    "source_audio",
    "converted_wav",
    "source_segment_index",
    "start_sec",
    "end_sec",
    "duration_sec",
    "peak_dbfs",
    "rms_dbfs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dave's single consecutive B/P word recording to 16 kHz mono, "
            "split it into 50 B words and 50 P words, and write labeled WAV clips."
        )
    )
    parser.add_argument("--input", type=Path, default=SOURCE_AUDIO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/dave_bp_consecutive_dataset"))
    parser.add_argument("--speaker", default="dave")
    parser.add_argument("--threshold-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--speech-threshold-dbfs", type=float, default=-40.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-duration-sec", type=float, default=0.08)
    parser.add_argument("--padding-sec", type=float, default=0.03)
    parser.add_argument("--preview-gap-sec", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wav_is_16k_mono(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getframerate() == 16000
                and handle.getsampwidth() == 2
            )
    except (wave.Error, EOFError, OSError):
        return False


def convert_to_16k_mono(source: Path, experiment_dir: Path, overwrite: bool) -> Path:
    if wav_is_16k_mono(source):
        return source

    experiment_dir.mkdir(parents=True, exist_ok=True)
    converted = experiment_dir / f"{source.stem}_16k_mono.wav"
    if converted.exists() and not overwrite:
        return converted

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(converted),
    ]
    subprocess.run(command, check=True)
    return converted


def detect_word_segments(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[list[dict], dict]:
    _, frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    if args.threshold_mode == "adaptive":
        threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(frame_db)
    else:
        threshold_dbfs = args.speech_threshold_dbfs
        threshold_info = {"method": "fixed", "threshold_dbfs": float(threshold_dbfs)}

    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        padding_sec=args.padding_sec,
    )
    segments, summary = detect_energy_segments(audio, sample_rate, threshold_dbfs, config=config)
    return segments, {"threshold_info": threshold_info, "summary": summary}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_preview(audio: np.ndarray, segments: list[dict], sample_rate: int, gap_sec: float) -> np.ndarray:
    gap = np.zeros(round(gap_sec * sample_rate), dtype=np.float32)
    pieces = []
    for segment in segments:
        chunk = audio[int(segment["start_sample"]) : int(segment["end_sample"])]
        pieces.append(chunk.astype(np.float32))
        pieces.append(gap)
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    converted_wav = convert_to_16k_mono(args.input, args.experiment_dir, args.overwrite)
    sample_rate, audio = read_wav_float(converted_wav)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz working WAV, got {sample_rate}: {converted_wav}")

    expected_words = [("B", word) for word in B_WORDS] + [("P", word) for word in P_WORDS]
    segments, segmentation_info = detect_word_segments(audio, sample_rate, args)
    if len(segments) < len(expected_words):
        raise RuntimeError(
            f"Only detected {len(segments)} segments, but expected {len(expected_words)}. "
            "Try lowering --speech-threshold-dbfs in fixed mode or reducing --merge-gap-sec."
        )

    selected = segments[: len(expected_words)]
    ignored = segments[len(expected_words) :]
    rows = []
    label_counts = {"B": 0, "P": 0}

    for global_index, ((label, word), segment) in enumerate(zip(expected_words, selected), start=1):
        label_counts[label] += 1
        word_index = label_counts[label]
        label_dir = args.output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / f"{args.speaker}_{label}_{word_index:04d}_{word}.wav"
        chunk = audio[int(segment["start_sample"]) : int(segment["end_sample"])]
        write_wav_float(out_path, sample_rate, chunk)
        rows.append(
            {
                "speaker": args.speaker,
                "label": label,
                "word_index": f"{word_index:04d}",
                "global_index": f"{global_index:04d}",
                "word": word,
                "audio_path": str(out_path),
                "source_audio": str(args.input),
                "converted_wav": str(converted_wav),
                "source_segment_index": str(segment["segment_index"]),
                "start_sec": str(segment["start_sec"]),
                "end_sec": str(segment["end_sec"]),
                "duration_sec": str(segment["duration_sec"]),
                "peak_dbfs": str(segment["peak_dbfs"]),
                "rms_dbfs": str(segment["rms_dbfs"]),
            }
        )

    write_csv(args.manifest, rows)
    preview = build_preview(audio, selected, sample_rate, args.preview_gap_sec)
    preview_path = args.experiment_dir / "dave_bp_consecutive_labeled_sequence.wav"
    write_wav_float(preview_path, sample_rate, preview)

    summary = {
        "speaker": args.speaker,
        "source_audio": str(args.input),
        "converted_wav": str(converted_wav),
        "output_root": str(args.output_root),
        "manifest": str(args.manifest),
        "preview_wav": str(preview_path),
        "expected_word_count": len(expected_words),
        "detected_segment_count": len(segments),
        "assigned_word_count": len(selected),
        "ignored_end_segment_count": len(ignored),
        "label_counts": label_counts,
        "first_assigned_word": rows[0]["word"] if rows else "",
        "last_assigned_word": rows[-1]["word"] if rows else "",
        "segmentation": segmentation_info,
    }
    summary_path = args.experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved manifest: {args.manifest}")
    print(f"Saved B clips: {args.output_root / 'B'}")
    print(f"Saved P clips: {args.output_root / 'P'}")
    print(f"Saved preview WAV: {preview_path}")


if __name__ == "__main__":
    main()
