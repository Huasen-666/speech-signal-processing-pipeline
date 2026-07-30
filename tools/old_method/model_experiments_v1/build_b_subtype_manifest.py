import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VOWEL_TO_SUBTYPE = {
    "AE": "B_AE",
    "AH": "B_AH",
    "EH": "B_EH",
    "EY": "B_EY",
    "IH": "B_IH",
    "IY": "B_IY",
    "OY": "B_OY",
    "ER": "B_ER",
    "AA_R": "B_ER",
    "OW": "B_OW",
    "UW": "B_UW",
    "UH": "B_UH",
    "AA": "B_AA",
    "AO": "B_AO",
    "AY": "B_AY",
    "AW": "B_AW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a B subtype manifest from B word metadata.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-context", type=Path, default=Path("data/metadata/b_context_groups.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--out", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/metadata/b_subtype_manifest_summary.json"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def available_stub_groups(stub_root: Path) -> set[str]:
    if not stub_root.exists():
        return set()
    return {path.name for path in stub_root.iterdir() if path.is_dir() and any(path.glob("*.wav"))}


def subtype_for_context(row: dict[str, str]) -> str:
    if row["onset_type"] == "bl_cluster":
        return "B_L"
    return VOWEL_TO_SUBTYPE.get(row["first_vowel_arpabet"], "")


def main() -> None:
    args = parse_args()
    context_by_word = {row["word"].lower(): row for row in read_csv(args.b_context)}
    stubs = available_stub_groups(args.stub_root)
    rows = []

    for row in read_csv(args.manifest):
        if row["label"].upper() != "B":
            continue

        word = row["word"].lower()
        context = context_by_word.get(word)
        subtype = subtype_for_context(context) if context else ""
        stub_available = subtype in stubs
        original_usable = row["usable"].lower() == "true"
        usable = original_usable and stub_available

        exclude_reason = row.get("exclude_reason", "")
        if not context:
            exclude_reason = "missing_b_context"
        elif not subtype:
            exclude_reason = "unmapped_b_subtype"
        elif not stub_available:
            exclude_reason = f"missing_stub_{subtype}"
        elif not original_usable:
            exclude_reason = exclude_reason or "source_marked_unusable"

        rows.append(
            {
                **row,
                "original_label": row["label"],
                "label": subtype,
                "b_subtype": subtype,
                "b_onset_type": context["onset_type"] if context else "",
                "b_first_vowel": context["first_vowel_arpabet"] if context else "",
                "stub_available": str(stub_available).lower(),
                "usable": str(usable).lower(),
                "exclude_reason": exclude_reason if not usable else "",
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.out, rows, fieldnames)

    usable_rows = [row for row in rows if row["usable"].lower() == "true"]
    summary = {
        "source_manifest": str(args.manifest),
        "b_context": str(args.b_context),
        "stub_root": str(args.stub_root),
        "available_stub_groups": sorted(stubs),
        "total_b_rows": len(rows),
        "usable_rows": len(usable_rows),
        "excluded_rows": len(rows) - len(usable_rows),
        "usable_by_subtype": dict(sorted(Counter(row["b_subtype"] for row in usable_rows).items())),
        "excluded_by_reason": dict(
            sorted(Counter(row["exclude_reason"] for row in rows if row["usable"].lower() != "true").items())
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved B subtype manifest: {args.out}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
