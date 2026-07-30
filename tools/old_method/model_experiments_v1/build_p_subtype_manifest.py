import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VOWEL_TO_SUBTYPE = {
    "AE": "P_AE",
    "AH": "P_AH",
    "EH": "P_EH",
    "EY": "P_EY",
    "IH": "P_IH",
    "IY": "P_IY",
    "OW": "P_OW",
    "UW": "P_UW",
    "UH": "P_UH",
    "AA": "P_AA",
    "AY": "P_AY",
    "ER": "P_ER",
    "AA_R": "P_ER",
    "AO_R": "P_ER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a P subtype manifest from P word metadata.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--p-context", type=Path, default=Path("data/metadata/p_context_groups.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--out", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/metadata/p_subtype_manifest_summary.json"))
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
    if row["onset_type"] == "pl_cluster":
        return "P_L"
    if row["onset_type"] == "pr_cluster":
        return "P_R"
    return VOWEL_TO_SUBTYPE.get(row["first_vowel_arpabet"], "")


def main() -> None:
    args = parse_args()
    context_by_word = {row["word"].lower(): row for row in read_csv(args.p_context)}
    stubs = available_stub_groups(args.stub_root)
    rows = []

    for row in read_csv(args.manifest):
        if row["label"].upper() != "P":
            continue

        word = row["word"].lower()
        context = context_by_word.get(word)
        subtype = subtype_for_context(context) if context else ""
        stub_available = subtype in stubs
        original_usable = row["usable"].lower() == "true"
        usable = original_usable and bool(context) and bool(subtype)

        exclude_reason = row.get("exclude_reason", "")
        if not context:
            exclude_reason = "missing_p_context"
        elif not subtype:
            exclude_reason = "unmapped_p_subtype"
        elif not original_usable:
            exclude_reason = exclude_reason or "source_marked_unusable"
        else:
            exclude_reason = ""

        rows.append(
            {
                **row,
                "original_label": row["label"],
                "label": subtype,
                "p_subtype": subtype,
                "p_onset_type": context["onset_type"] if context else "",
                "p_first_vowel": context["first_vowel_arpabet"] if context else "",
                "stub_available": str(stub_available).lower(),
                "usable": str(usable).lower(),
                "exclude_reason": exclude_reason,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.out, rows, fieldnames)

    usable_rows = [row for row in rows if row["usable"].lower() == "true"]
    summary = {
        "source_manifest": str(args.manifest),
        "p_context": str(args.p_context),
        "stub_root": str(args.stub_root),
        "available_stub_groups": sorted(stubs),
        "total_p_rows": len(rows),
        "usable_rows": len(usable_rows),
        "excluded_rows": len(rows) - len(usable_rows),
        "usable_by_subtype": dict(sorted(Counter(row["p_subtype"] for row in usable_rows).items())),
        "usable_by_onset_type": dict(sorted(Counter(row["p_onset_type"] for row in usable_rows).items())),
        "excluded_by_reason": dict(
            sorted(Counter(row["exclude_reason"] for row in rows if row["usable"].lower() != "true").items())
        ),
        "stub_available_by_subtype": dict(
            sorted(Counter(row["p_subtype"] for row in usable_rows if row["stub_available"].lower() == "true").items())
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved P subtype manifest: {args.out}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
