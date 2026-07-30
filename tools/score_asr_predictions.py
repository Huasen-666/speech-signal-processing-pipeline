"""Honest ASR scoring for patient (electrolaryngeal) speech.

Why this exists
---------------
Raw WER on our fixed-text sentences is dominated by *annotation-style noise*:
the reference transcripts sometimes drop function words (the / and / it / in / on),
while Whisper naturally emits full grammatical English. WER then punishes every
function-word mismatch as if the model misheard a consonant -- which it did not.

This script scores a predictions CSV (columns: reference, prediction) three ways
so you can see how much of your error is real vs. annotation style:

  1. RAW WER / CER            -- lowercased + punctuation-stripped only (headline).
  2. NORMALIZED WER          -- same, plus contraction/number normalization.
  3. CONTENT-WORD WER        -- drops a function-word stoplist; isolates the
                                acoustic / lexical accuracy (incl. B/P consonants)
                                from function-word annotation style.

The gap between RAW and CONTENT-WORD WER is the part of your error that is fixable
by standardizing transcripts, NOT by more modeling.

Usage
-----
  conda run --no-capture-output -n win_ai python tools\\score_asr_predictions.py \
      --pred-csv experiments\\asr_whisper_lora\\david_overfit_sanity\\adapted_predictions.csv

  # score several experiments at once:
  python tools\\score_asr_predictions.py --pred-csv a\\adapted_predictions.csv b\\adapted_predictions.csv

Outputs a per-row table to stdout and writes <pred-csv-dir>\\honest_scores.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Function words that (a) carry little lexical content and (b) our reference
# transcripts are inconsistent about. Dropping them yields the "content WER"
# diagnostic. This is NOT the headline metric -- it is a floor that shows how
# well the acoustics/consonants were recognized once annotation style is removed.
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "so", "if",
    "it", "its", "this", "that", "these", "those",
    "in", "on", "of", "to", "up", "for", "by", "at", "as", "with", "from",
    "is", "was", "are", "were", "be", "been", "am",
    "i", "he", "she", "we", "they", "you", "him", "her", "them", "his", "their",
    "do", "did", "does", "will", "would", "then",
}

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "i'm": "i am", "it's": "it is",
    "he's": "he is", "she's": "she is", "they're": "they are", "we're": "we are",
    "i'll": "i will", "you're": "you are", "there's": "there is",
}


def basic_normalize(text: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes), collapse whitespace."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def full_normalize(text: str) -> str:
    """basic_normalize + expand common contractions + strip stray apostrophes."""
    t = text.lower()
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = t.replace("'", " ")
    return re.sub(r"\s+", " ", t).strip()


def drop_function_words(text: str) -> str:
    return " ".join(w for w in text.split() if w not in FUNCTION_WORDS)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref)][len(hyp)]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    if not r:
        return (len(h), 0)
    return edit_distance(r, h), len(r)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    if not r:
        return (len(h), 0)
    return edit_distance(r, h), len(r)


def score_csv(path: Path, ref_col: str, hyp_col: str) -> dict:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"No rows in {path}")
    if ref_col not in rows[0] or hyp_col not in rows[0]:
        raise KeyError(f"{path} needs columns '{ref_col}' and '{hyp_col}'; got {list(rows[0])}")

    agg = {k: [0, 0] for k in ("raw_wer", "raw_cer", "norm_wer", "content_wer")}
    per_row = []
    for x in rows:
        ref_raw, hyp_raw = basic_normalize(x[ref_col]), basic_normalize(x[hyp_col])
        ref_n, hyp_n = full_normalize(x[ref_col]), full_normalize(x[hyp_col])
        ref_c, hyp_c = drop_function_words(ref_n), drop_function_words(hyp_n)

        e_rw, n_rw = wer(ref_raw, hyp_raw)
        e_rc, n_rc = cer(ref_raw, hyp_raw)
        e_nw, n_nw = wer(ref_n, hyp_n)
        e_cw, n_cw = wer(ref_c, hyp_c)
        for k, (e, n) in (
            ("raw_wer", (e_rw, n_rw)), ("raw_cer", (e_rc, n_rc)),
            ("norm_wer", (e_nw, n_nw)), ("content_wer", (e_cw, n_cw)),
        ):
            agg[k][0] += e
            agg[k][1] += n
        per_row.append({
            "reference": x[ref_col],
            "prediction": x[hyp_col],
            "raw_wer": round(e_rw / n_rw, 3) if n_rw else None,
            "content_wer": round(e_cw / n_cw, 3) if n_cw else None,
        })

    def rate(k: str) -> float:
        e, n = agg[k]
        return round(100.0 * e / n, 2) if n else 0.0

    return {
        "file": str(path),
        "n_rows": len(rows),
        "raw_wer_pct": rate("raw_wer"),
        "raw_cer_pct": rate("raw_cer"),
        "normalized_wer_pct": rate("norm_wer"),
        "content_word_wer_pct": rate("content_wer"),
        "annotation_style_gap_pct": round(rate("raw_wer") - rate("content_wer"), 2),
        "per_row": per_row,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-csv", nargs="+", required=True, type=Path,
                    help="one or more predictions CSVs (columns: reference, prediction)")
    ap.add_argument("--ref-col", default="reference")
    ap.add_argument("--hyp-col", default="prediction")
    args = ap.parse_args()

    for path in args.pred_csv:
        result = score_csv(path, args.ref_col, args.hyp_col)
        print("\n" + "=" * 70)
        print(f"FILE: {result['file']}  ({result['n_rows']} clips)")
        print("-" * 70)
        print(f"  RAW WER            : {result['raw_wer_pct']:5.2f}%   (headline, punctuation/case removed)")
        print(f"  NORMALIZED WER     : {result['normalized_wer_pct']:5.2f}%   (+ contractions/number normalization)")
        print(f"  CONTENT-WORD WER   : {result['content_word_wer_pct']:5.2f}%   (drops function words -> real acoustic/consonant accuracy)")
        print(f"  RAW CER            : {result['raw_cer_pct']:5.2f}%")
        print(f"  --> annotation-style gap = {result['annotation_style_gap_pct']:.2f}%  (fixable by standardizing transcripts, NOT modeling)")
        print("-" * 70)
        for r in result["per_row"]:
            flag = "" if (r["content_wer"] or 0) == 0 else "  <-- real error"
            print(f"    raw={r['raw_wer']}  content={r['content_wer']}{flag}")
            print(f"      REF: {r['reference']}")
            print(f"      HYP: {r['prediction']}")
        out = path.parent / "honest_scores.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
