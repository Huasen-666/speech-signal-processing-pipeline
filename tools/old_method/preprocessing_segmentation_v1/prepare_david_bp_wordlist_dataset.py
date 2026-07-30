import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments

from build_dataset_manifest import B_WORDS, P_WORDS


SOURCE_ROOT = PROJECT_ROOT.parent / "Cleaned Sound" / "David"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "david_wordlist" / "David"


RECORDINGS = [
    {
        "label": "B",
        "take": 1,
        "source": SOURCE_ROOT / "Word list - B words.m4a",
        "recording_id": "david_b_take1",
        "words": B_WORDS,
    },
    {
        "label": "B",
        "take": 2,
        "source": SOURCE_ROOT / "Word list - B words (2).m4a",
        "recording_id": "david_b_take2",
        "words": B_WORDS,
    },
    {
        "label": "P",
        "take": 1,
        "source": SOURCE_ROOT / "Word list - P Words.m4a",
        "recording_id": "david_p_take1",
        "words": P_WORDS,
    },
    {
        "label": "P",
        "take": 2,
        "source": SOURCE_ROOT / "Word list - P words (2).m4a",
        "recording_id": "david_p_take2",
        "words": P_WORDS,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert David's B/P m4a word-list recordings, split the first 50 word segments, and label them."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/david_bp_wordlist_dataset"))
    parser.add_argument("--speaker", default="david")
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.6)
    parser.add_argument("--min-duration-sec", type=float, default=0.12)
    parser.add_argument("--padding-sec", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_to_wav(source: Path, overwrite: bool) -> Path:
    wav = source.with_suffix(".16k_mono.wav")
    if wav.exists() and not overwrite:
        return wav
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
        str(wav),
    ]
    subprocess.run(command, check=True)
    return wav


def detect_word_segments(
    audio: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    _, frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(frame_db)
    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
        padding_sec=args.padding_sec,
    )
    segments, summary = detect_energy_segments(audio, sample_rate, threshold_dbfs, config=config)
    return segments, {"threshold_info": threshold_info, "summary": summary}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    assignment_rows = []
    summary_rows = []

    for recording in RECORDINGS:
        source = recording["source"]
        if not source.exists():
            raise FileNotFoundError(source)
        label = recording["label"]
        take = int(recording["take"])
        words = list(recording["words"])
        wav_path = convert_to_wav(source, args.overwrite)
        sample_rate, audio = read_wav_float(wav_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz after conversion: {wav_path}")

        segments, segmentation_info = detect_word_segments(audio, sample_rate, args)
        if len(segments) < len(words):
            raise RuntimeError(
                f"{recording['recording_id']} produced only {len(segments)} segments for {len(words)} words."
            )

        selected = segments[: len(words)]
        ignored = segments[len(words) :]
        label_dir = args.output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for word_index, (word, segment) in enumerate(zip(words, selected), start=1):
            global_index = (take - 1) * len(words) + word_index
            out_name = f"{args.speaker}_{label}_{global_index:04d}_{word}.wav"
            out_path = label_dir / out_name
            chunk = audio[int(segment["start_sample"]) : int(segment["end_sample"])]
            write_wav_float(out_path, sample_rate, chunk)
            assignment_rows.append(
                {
                    "speaker": args.speaker,
                    "label": label,
                    "take": str(take),
                    "word_index": str(word_index),
                    "global_index": f"{global_index:04d}",
                    "word": word,
                    "audio_path": str(out_path),
                    "source_m4a": str(source),
                    "converted_wav": str(wav_path),
                    "source_segment_index": str(segment["segment_index"]),
                    "start_sec": str(segment["start_sec"]),
                    "end_sec": str(segment["end_sec"]),
                    "duration_sec": str(segment["duration_sec"]),
                    "peak_dbfs": str(segment["peak_dbfs"]),
                    "rms_dbfs": str(segment["rms_dbfs"]),
                }
            )

        summary_rows.append(
            {
                "recording_id": recording["recording_id"],
                "label": label,
                "take": take,
                "source_m4a": str(source),
                "converted_wav": str(wav_path),
                "detected_segments": len(segments),
                "assigned_word_segments": len(selected),
                "ignored_end_segments": len(ignored),
                "first_word": words[0],
                "last_word": words[-1],
                "segmentation": segmentation_info,
            }
        )

    assignment_csv = args.experiment_dir / "david_bp_word_assignments.csv"
    summary_json = args.experiment_dir / "summary.json"
    write_rows(assignment_csv, assignment_rows)
    summary = {
        "speaker": args.speaker,
        "output_root": str(args.output_root),
        "total_assigned_word_segments": len(assignment_rows),
        "by_recording": summary_rows,
        "assignment_csv": str(assignment_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved assignment CSV: {assignment_csv}")
    print(f"Saved segmented WAVs under: {args.output_root / 'B'} and {args.output_root / 'P'}")


if __name__ == "__main__":
    main()
