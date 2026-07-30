import argparse
import csv
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path


B_WORDS = [
    "back",
    "bad",
    "bag",
    "bait",
    "ball",
    "ban",
    "bar",
    "barn",
    "base",
    "bash",
    "bat",
    "bath",
    "bay",
    "beach",
    "bead",
    "beat",
    "bed",
    "bell",
    "bend",
    "best",
    "bid",
    "big",
    "bill",
    "bin",
    "bit",
    "bite",
    "black",
    "blade",
    "blank",
    "blast",
    "blot",
    "blue",
    "boat",
    "bob",
    "bond",
    "bone",
    "book",
    "boot",
    "boss",
    "both",
    "bound",
    "box",
    "boy",
    "bud",
    "bug",
    "bulk",
    "bull",
    "burn",
    "bus",
    "buy",
]

P_WORDS = [
    "pace",
    "pack",
    "pad",
    "pain",
    "pan",
    "park",
    "part",
    "pass",
    "past",
    "pat",
    "path",
    "pay",
    "peak",
    "peel",
    "peg",
    "pen",
    "pet",
    "pick",
    "pig",
    "pill",
    "pine",
    "pink",
    "pit",
    "place",
    "plain",
    "plan",
    "plant",
    "play",
    "plot",
    "plug",
    "plus",
    "pod",
    "pole",
    "pool",
    "pop",
    "port",
    "pot",
    "pour",
    "press",
    "print",
    "probe",
    "prop",
    "puff",
    "pull",
    "pump",
    "push",
    "put",
    "page",
    "pale",
    "palm",
]

EXPECTED_WORDS = {
    "B": B_WORDS,
    "P": P_WORDS,
}

FIELDNAMES = [
    "audio_path",
    "speaker",
    "label",
    "folder_label",
    "file_speaker",
    "file_label",
    "file_index",
    "protocol_index",
    "word",
    "usable",
    "exclude_reason",
    "sample_rate_hz",
    "channels",
    "sample_width_bits",
    "duration_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training dataset manifest from speaker/label WAV folders."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed"),
        help="Root directory containing speaker folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/metadata/dataset_manifest.csv"),
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/metadata/dataset_manifest_summary.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["B", "P"],
        help="Target labels to include, for example: B P T D K G S Z.",
    )
    return parser.parse_args()


def normalise_bool(value: bool) -> str:
    return "true" if value else "false"


def parse_label_folder(folder_name: str, allowed_labels: set[str]) -> tuple[str | None, bool, str]:
    normalized = folder_name.upper()
    if normalized in allowed_labels:
        return normalized, True, ""

    parts = normalized.split("_")
    if len(parts) >= 2 and parts[0] in allowed_labels:
        if any(part in {"WRONG", "BAD", "EXCLUDE", "EXCLUDED"} for part in parts[1:]):
            return parts[0], False, "folder_marked_wrong"

    return None, False, "not_target_label_folder"


def parse_filename_metadata(path: Path) -> dict[str, str]:
    stem = path.stem

    new_style = re.match(
        r"^(?P<speaker>[^_]+)_(?P<label>[A-Za-z]+)_(?P<index>\d{1,5})_(?P<word>.+)$",
        stem,
    )
    if new_style:
        return {key: value.lower() for key, value in new_style.groupdict().items()}

    old_style = re.match(
        r"^(?P<speaker>[^_]+)_(?P<label>[A-Za-z]+)_(?P<word>.+)_(?P<index>\d{1,5})$",
        stem,
    )
    if old_style:
        return {key: value.lower() for key, value in old_style.groupdict().items()}

    return {
        "speaker": "",
        "label": "",
        "index": "",
        "word": "",
    }


def wav_header(path: Path) -> tuple[int | str, int | str, int | str, float | str, str]:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width_bits = handle.getsampwidth() * 8
            duration_sec = handle.getnframes() / sample_rate
        return sample_rate, channels, sample_width_bits, round(duration_sec, 3), ""
    except (wave.Error, EOFError, OSError) as exc:
        return "", "", "", "", f"wav_header_error:{exc}"


def expected_index(label: str, word: str) -> str:
    words = EXPECTED_WORDS.get(label, [])
    if word in words:
        return f"{words.index(word) + 1:04d}"
    return ""


def build_rows(data_root: Path, labels: list[str]) -> list[dict[str, str]]:
    allowed_labels = {label.upper() for label in labels}
    rows = []

    for speaker_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        speaker = speaker_dir.name.lower()
        for label_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            label, folder_usable, folder_reason = parse_label_folder(label_dir.name, allowed_labels)
            if label is None:
                continue

            for audio_path in sorted(label_dir.glob("*.wav")):
                meta = parse_filename_metadata(audio_path)
                file_speaker = meta["speaker"]
                file_label = meta["label"].upper()
                file_index = meta["index"]
                word = meta["word"].lower()
                protocol_index = expected_index(label, word)
                usable = folder_usable
                reasons = []
                if folder_reason:
                    reasons.append(folder_reason)
                if file_label and file_label != label:
                    usable = False
                    reasons.append("filename_label_mismatch")
                if file_speaker and file_speaker != speaker:
                    reasons.append("filename_speaker_mismatch")
                if not word:
                    reasons.append("filename_word_missing")
                elif label in EXPECTED_WORDS and not protocol_index:
                    reasons.append("unexpected_word_for_label")

                sample_rate, channels, sample_width_bits, duration_sec, header_error = wav_header(audio_path)
                if header_error:
                    usable = False
                    reasons.append(header_error)

                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "speaker": speaker,
                        "label": label,
                        "folder_label": label_dir.name,
                        "file_speaker": file_speaker,
                        "file_label": file_label,
                        "file_index": file_index,
                        "protocol_index": protocol_index,
                        "word": word,
                        "usable": normalise_bool(usable),
                        "exclude_reason": ";".join(reasons),
                        "sample_rate_hz": str(sample_rate),
                        "channels": str(channels),
                        "sample_width_bits": str(sample_width_bits),
                        "duration_sec": str(duration_sec),
                    }
                )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], labels: list[str]) -> dict:
    by_speaker_label = defaultdict(Counter)
    missing_words = defaultdict(dict)
    duplicate_words = defaultdict(dict)
    reasons = Counter()

    for row in rows:
        key = f"{row['speaker']}/{row['label']}"
        by_speaker_label[key]["total"] += 1
        if row["usable"] == "true":
            by_speaker_label[key]["usable"] += 1
        else:
            by_speaker_label[key]["excluded"] += 1
        for reason in filter(None, row["exclude_reason"].split(";")):
            reasons[reason] += 1

    for label in [label.upper() for label in labels]:
        expected = EXPECTED_WORDS.get(label)
        if not expected:
            continue
        speakers = sorted({row["speaker"] for row in rows if row["label"] == label})
        for speaker in speakers:
            usable_words = [
                row["word"]
                for row in rows
                if row["speaker"] == speaker and row["label"] == label and row["usable"] == "true"
            ]
            word_counts = Counter(usable_words)
            missing = [word for word in expected if word_counts[word] == 0]
            duplicated = {word: count for word, count in sorted(word_counts.items()) if count > 1}
            if missing:
                missing_words[f"{speaker}/{label}"] = missing
            if duplicated:
                duplicate_words[f"{speaker}/{label}"] = duplicated

    return {
        "total_rows": len(rows),
        "usable_rows": sum(1 for row in rows if row["usable"] == "true"),
        "excluded_rows": sum(1 for row in rows if row["usable"] != "true"),
        "by_speaker_label": {
            key: dict(counter)
            for key, counter in sorted(by_speaker_label.items())
        },
        "exclude_reasons": dict(sorted(reasons.items())),
        "missing_expected_words": dict(sorted(missing_words.items())),
        "duplicate_usable_words": dict(sorted(duplicate_words.items())),
    }


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)

    rows = build_rows(args.data_root, args.labels)
    write_csv(args.out, rows)

    summary = summarize(rows, args.labels)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved dataset manifest: {args.out}")
    print(f"Saved dataset summary: {args.summary}")


if __name__ == "__main__":
    main()
