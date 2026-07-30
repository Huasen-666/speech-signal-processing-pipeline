import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments


MANIFEST_FIELDS = [
    "segment_id",
    "recording_id",
    "audio_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "start_sample",
    "end_sample",
    "peak_dbfs",
    "rms_dbfs",
    "threshold_dbfs",
]


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candidate speech segments using energy VAD.")
    parser.add_argument("--input", type=Path, required=True, help="Input 16-bit PCM WAV.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for segment WAVs and manifests.")
    parser.add_argument("--recording-id", help="Stable recording ID. Defaults to input stem.")
    parser.add_argument("--threshold-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--speech-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--adaptive-noise-margin-db", type=float, default=15.0)
    parser.add_argument("--adaptive-speech-margin-db", type=float, default=25.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.2)
    parser.add_argument("--min-duration-sec", type=float, default=0.15)
    parser.add_argument("--padding-sec", type=float, default=0.05)
    parser.add_argument("--max-segments", type=int, help="Optional cap for quick inspection exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    recording_id = args.recording_id or args.input.stem
    wav_dir = args.outdir / "segments_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    sample_rate, audio = read_wav_float(args.input)
    _, frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    if args.threshold_mode == "adaptive":
        threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(
            frame_db,
            noise_margin_db=args.adaptive_noise_margin_db,
            speech_margin_db=args.adaptive_speech_margin_db,
        )
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
    if args.max_segments is not None:
        segments = segments[: args.max_segments]

    manifest_rows = []
    for row in segments:
        segment_id = f"{recording_id}_{row['segment_index']:04d}"
        audio_path = wav_dir / f"{segment_id}_{row['start_sec']:.2f}_{row['end_sec']:.2f}.wav"
        chunk = audio[row["start_sample"] : row["end_sample"]]
        write_wav_float(audio_path, sample_rate, chunk)

        manifest_rows.append(
            {
                "segment_id": segment_id,
                "recording_id": recording_id,
                "audio_path": str(audio_path),
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "duration_sec": row["duration_sec"],
                "start_sample": row["start_sample"],
                "end_sample": row["end_sample"],
                "peak_dbfs": row["peak_dbfs"],
                "rms_dbfs": row["rms_dbfs"],
                "threshold_dbfs": threshold_dbfs,
            }
        )

    manifest_csv = args.outdir / "manifest_segments.csv"
    manifest_json = args.outdir / "manifest_segments.json"
    report_json = args.outdir / "segmentation_report.json"

    write_manifest_csv(manifest_csv, manifest_rows)
    manifest_json.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    report = {
        "input": str(args.input),
        "recording_id": recording_id,
        "sample_rate_hz": sample_rate,
        "threshold_mode": args.threshold_mode,
        "threshold_info": threshold_info,
        "summary": summary,
        "exported_segment_count": len(manifest_rows),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "segments_wav_dir": str(wav_dir),
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved segment WAVs: {wav_dir}")
    print(f"Saved manifest CSV: {manifest_csv}")
    print(f"Saved report: {report_json}")


if __name__ == "__main__":
    main()
