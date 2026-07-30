from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


B_WORD_TO_SUBTYPE = {
    "back": "B_AE",
    "bad": "B_AE",
    "bag": "B_AE",
    "bait": "B_EY",
    "ball": "B_AA",
    "ban": "B_AE",
    "bar": "B_R",
    "barn": "B_R",
    "base": "B_EY",
    "bash": "B_AE",
    "bat": "B_AE",
    "bath": "B_AE",
    "bay": "B_EY",
    "beach": "B_IY",
    "bead": "B_IY",
    "beat": "B_IY",
    "bed": "B_EH",
    "bell": "B_EH",
    "bend": "B_EH",
    "best": "B_EH",
    "bid": "B_IH",
    "big": "B_IH",
    "bill": "B_IH",
    "bin": "B_IH",
    "bit": "B_IH",
    "bite": "B_AE",
    "black": "B_L",
    "blade": "B_L",
    "blank": "B_L",
    "blast": "B_L",
    "blot": "B_L",
    "blue": "B_L",
    "boat": "B_OW",
    "bob": "B_AA",
    "bond": "B_AA",
    "bone": "B_OW",
    "book": "B_AH",
    "boot": "B_OW",
    "boss": "B_AA",
    "both": "B_OW",
    "bound": "B_AA",
    "box": "B_AA",
    "boy": "B_OY",
    "bud": "B_AH",
    "bug": "B_AH",
    "bulk": "B_AH",
    "bull": "B_AH",
    "burn": "B_R",
    "bus": "B_AH",
    "buy": "B_AE",
}


P_WORD_TO_SUBTYPE = {
    "pace": "P_EY",
    "pack": "P_AE",
    "pad": "P_AE",
    "pain": "P_EY",
    "pan": "P_AE",
    "park": "P_AH",
    "part": "P_AH",
    "pass": "P_AE",
    "past": "P_AE",
    "pat": "P_AE",
    "path": "P_AE",
    "pay": "P_EY",
    "peak": "P_IH",
    "peel": "P_IH",
    "peg": "P_EH",
    "pen": "P_EH",
    "pet": "P_EH",
    "pick": "P_IH",
    "pig": "P_IH",
    "pill": "P_IH",
    "pine": "P_AH",
    "pink": "P_IH",
    "pit": "P_IH",
    "place": "P_AE",
    "plain": "P_AE",
    "plan": "P_AE",
    "plant": "P_AE",
    "play": "P_EY",
    "plot": "P_AH",
    "plug": "P_AH",
    "plus": "P_AH",
    "pod": "P_AH",
    "pole": "P_UH",
    "pool": "P_UH",
    "pop": "P_AH",
    "port": "P_AH",
    "pot": "P_AH",
    "pour": "P_AH",
    "press": "P_EH",
    "print": "P_IH",
    "probe": "P_UH",
    "prop": "P_AH",
    "puff": "P_UH",
    "pull": "P_UH",
    "pump": "P_UH",
    "push": "P_UH",
    "put": "P_UH",
    "page": "P_EY",
    "pale": "P_EY",
    "palm": "P_AH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build David-specific B/P subtype manifests for stub replacement.")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/david_bp_dataset_manifest.csv"))
    parser.add_argument("--b-stub-root", type=Path, default=Path("../consonant/B"))
    parser.add_argument("--p-stub-root", type=Path, default=Path("../consonant/P"))
    parser.add_argument("--b-out", type=Path, default=Path("data/metadata/david_b_subtype_manifest.csv"))
    parser.add_argument("--p-out", type=Path, default=Path("data/metadata/david_p_subtype_manifest.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/metadata/david_bp_subtype_manifest_summary.json"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def available_stub_groups(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir() and any(path.glob("*.wav"))}


def build_rows(
    source_rows: list[dict[str, str]],
    label: str,
    mapping: dict[str, str],
    available_stubs: set[str],
) -> list[dict[str, str]]:
    rows = []
    subtype_col = f"{label.lower()}_subtype"
    for row in source_rows:
        if row.get("label", "").upper() != label:
            continue
        word = row.get("word", "").lower()
        subtype = mapping.get(word, "")
        original_usable = row.get("usable", "").lower() == "true"
        stub_available = subtype in available_stubs
        usable = original_usable and bool(subtype) and stub_available
        exclude_reason = ""
        if not original_usable:
            exclude_reason = row.get("exclude_reason") or "source_marked_unusable"
        elif not subtype:
            exclude_reason = f"unmapped_{label.lower()}_subtype"
        elif not stub_available:
            exclude_reason = f"missing_stub_{subtype}"

        rows.append(
            {
                **row,
                "original_label": label,
                "label": subtype,
                subtype_col: subtype,
                f"{label.lower()}_onset_type": "simple_or_cluster",
                f"{label.lower()}_first_vowel": subtype.split("_", 1)[1] if "_" in subtype else "",
                "stub_available": str(stub_available).lower(),
                "usable": str(usable).lower(),
                "exclude_reason": exclude_reason,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    source_rows = read_csv(args.manifest)
    b_stubs = available_stub_groups(args.b_stub_root)
    p_stubs = available_stub_groups(args.p_stub_root)
    b_rows = build_rows(source_rows, "B", B_WORD_TO_SUBTYPE, b_stubs)
    p_rows = build_rows(source_rows, "P", P_WORD_TO_SUBTYPE, p_stubs)

    write_csv(args.b_out, b_rows)
    write_csv(args.p_out, p_rows)

    b_usable = [row for row in b_rows if row["usable"] == "true"]
    p_usable = [row for row in p_rows if row["usable"] == "true"]
    summary = {
        "source_manifest": str(args.manifest),
        "b_out": str(args.b_out),
        "p_out": str(args.p_out),
        "b_available_stub_groups": sorted(b_stubs),
        "p_available_stub_groups": sorted(p_stubs),
        "b_total_rows": len(b_rows),
        "b_usable_rows": len(b_usable),
        "b_excluded_rows": len(b_rows) - len(b_usable),
        "b_usable_by_subtype": dict(sorted(Counter(row["b_subtype"] for row in b_usable).items())),
        "b_excluded_by_reason": dict(sorted(Counter(row["exclude_reason"] for row in b_rows if row["usable"] != "true").items())),
        "p_total_rows": len(p_rows),
        "p_usable_rows": len(p_usable),
        "p_excluded_rows": len(p_rows) - len(p_usable),
        "p_usable_by_subtype": dict(sorted(Counter(row["p_subtype"] for row in p_usable).items())),
        "p_excluded_by_reason": dict(sorted(Counter(row["exclude_reason"] for row in p_rows if row["usable"] != "true").items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
