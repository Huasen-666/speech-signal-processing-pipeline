import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MANIFEST = Path("recording_manifest.csv")


FIELDNAMES = [
    "recording_id",
    "source_audio",
    "converted_wav",
    "inspection_report",
    "speaker_id",
    "session_id",
    "protocol",
    "status",
    "notes",
    "sample_rate_hz",
    "channels",
    "sample_width_bytes",
    "duration_sec",
    "peak_dbfs",
    "rms_dbfs",
    "dc_offset",
    "clipping_samples",
    "clipping_ratio",
    "noise_floor_p10_dbfs",
    "energy_median_p50_dbfs",
    "speech_ratio",
    "detected_segment_count",
    "last_updated_utc",
]


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({field: row.get(field, "") for field in FIELDNAMES})
        return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def derive_recording_id(report: dict, explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id

    input_path = Path(report["input"])
    return input_path.stem


def report_to_manifest_row(
    report: dict,
    report_path: Path,
    recording_id: str,
    source_audio: str,
    converted_wav: str,
    speaker_id: str,
    session_id: str,
    protocol: str,
    status: str,
    notes: str,
) -> dict[str, str]:
    fmt = report["format"]
    level = report["level"]
    energy = report["short_time_energy"]
    vad = report["rough_energy_vad"]

    return {
        "recording_id": recording_id,
        "source_audio": source_audio,
        "converted_wav": converted_wav,
        "inspection_report": str(report_path),
        "speaker_id": speaker_id,
        "session_id": session_id,
        "protocol": protocol,
        "status": status,
        "notes": notes,
        "sample_rate_hz": str(fmt["sample_rate_hz"]),
        "channels": str(fmt["channels"]),
        "sample_width_bytes": str(fmt["sample_width_bytes"]),
        "duration_sec": str(fmt["duration_sec"]),
        "peak_dbfs": str(level["peak_dbfs"]),
        "rms_dbfs": str(level["rms_dbfs"]),
        "dc_offset": str(level["dc_offset"]),
        "clipping_samples": str(level["clipping_samples"]),
        "clipping_ratio": str(level["clipping_ratio"]),
        "noise_floor_p10_dbfs": str(energy["p10_dbfs"]),
        "energy_median_p50_dbfs": str(energy["p50_dbfs"]),
        "speech_ratio": str(vad["voiced_ratio"]),
        "detected_segment_count": str(vad["segment_count"]),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def upsert_row(rows: list[dict[str, str]], new_row: dict[str, str]) -> list[dict[str, str]]:
    for index, row in enumerate(rows):
        if row["recording_id"] == new_row["recording_id"]:
            rows[index] = new_row
            return rows

    rows.append(new_row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update recording_manifest.csv from an inspection_report.json file."
    )
    parser.add_argument("--report", type=Path, required=True, help="Path to inspection_report.json.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recording-id", help="Stable ID for this recording. Defaults to input file stem.")
    parser.add_argument("--source-audio", default="", help="Original source file path, such as an M4A.")
    parser.add_argument("--converted-wav", default="", help="Converted WAV path used for inspection.")
    parser.add_argument("--speaker-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--status", default="inspected")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.report.exists():
        raise FileNotFoundError(args.report)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    recording_id = derive_recording_id(report, args.recording_id)
    converted_wav = args.converted_wav or report["input"]

    new_row = report_to_manifest_row(
        report=report,
        report_path=args.report,
        recording_id=recording_id,
        source_audio=args.source_audio,
        converted_wav=converted_wav,
        speaker_id=args.speaker_id,
        session_id=args.session_id,
        protocol=args.protocol,
        status=args.status,
        notes=args.notes,
    )

    rows = load_manifest(args.manifest)
    rows = upsert_row(rows, new_row)
    write_manifest(args.manifest, rows)

    print(f"Updated {args.manifest} with recording_id={recording_id}")


if __name__ == "__main__":
    main()
