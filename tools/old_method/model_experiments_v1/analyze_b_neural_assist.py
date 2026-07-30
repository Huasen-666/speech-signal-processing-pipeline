import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_pipeline.audio_io import read_wav_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use wav2vec2-phoneme as an assistive layer for B-initial boundary/subtype analysis."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/neural_assist/b_initial_wav2vec2"))
    parser.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face network downloads/checks.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--include-speakers", nargs="*", default=["bascom", "corrick", "mickey"])
    parser.add_argument("--include-words", nargs="*", default=[])
    parser.add_argument("--include-subtypes", nargs="*", default=[])
    parser.add_argument("--max-rows", type=int, default=24, help="Set <=0 to process every selected row.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-extra-ms", type=float, default=45.0)
    parser.add_argument("--mask-min-ms", type=float, default=90.0)
    parser.add_argument("--mask-max-ms", type=float, default=320.0)
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


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    allowed_speakers = {speaker.lower() for speaker in args.include_speakers}
    allowed_words = {word.lower() for word in args.include_words}
    allowed_subtypes = {subtype.upper() for subtype in args.include_subtypes}
    candidates = [
        row
        for row in rows
        if row["usable"].lower() == "true"
        and row["speaker"].lower() in allowed_speakers
        and (not allowed_words or row["word"].lower() in allowed_words)
        and (not allowed_subtypes or row["b_subtype"].upper() in allowed_subtypes)
    ]
    candidates = sorted(candidates, key=lambda row: (row["speaker"], row["protocol_index"], row["word"]))
    if args.max_rows > 0 and len(candidates) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(len(candidates), size=args.max_rows, replace=False).tolist())
        candidates = [candidates[idx] for idx in indices]
    return candidates


def list_stubs(stub_root: Path) -> dict[str, Path]:
    stubs = {}
    for group_dir in sorted(stub_root.iterdir() if stub_root.exists() else []):
        if not group_dir.is_dir():
            continue
        files = sorted(group_dir.glob("*.wav"))
        if files:
            stubs[group_dir.name.upper()] = files[0]
    return stubs


def load_audio(path: Path, target_sample_rate: int) -> tuple[int, np.ndarray]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32)
        sample_rate = target_sample_rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return sample_rate, audio


def energy_region(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return {"start_sec": 0.0, "end_sec": len(audio) / sample_rate, "threshold_db": -55.0}

    starts = []
    rms_values = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frame = audio[start : start + frame_len]
        starts.append(start)
        rms_values.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))

    if not rms_values or max(rms_values) <= 1e-10:
        return {"start_sec": 0.0, "end_sec": len(audio) / sample_rate, "threshold_db": -55.0}

    frame_db = 20.0 * np.log10(np.maximum(rms_values, 1e-12))
    noise_floor = float(np.percentile(frame_db, 10))
    high_energy = float(np.percentile(frame_db, 95))
    threshold = max(noise_floor + 15.0, high_energy - 30.0, -55.0)
    active = np.flatnonzero(frame_db >= threshold)
    if not len(active):
        return {"start_sec": 0.0, "end_sec": len(audio) / sample_rate, "threshold_db": threshold}
    start_sec = starts[int(active[0])] / sample_rate
    end_sec = (starts[int(active[-1])] + frame_len) / sample_rate
    return {"start_sec": start_sec, "end_sec": end_sec, "threshold_db": threshold}


def frame_segments(pred_ids: np.ndarray, frame_conf: np.ndarray, tokenizer, time_step: float) -> list[dict[str, float | str]]:
    blank_ids = {idx for idx in [tokenizer.pad_token_id, tokenizer.word_delimiter_token_id] if idx is not None}
    segments = []
    start = 0
    while start < len(pred_ids):
        token_id = int(pred_ids[start])
        end = start + 1
        while end < len(pred_ids) and int(pred_ids[end]) == token_id:
            end += 1
        if token_id not in blank_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            segments.append(
                {
                    "phoneme": str(token),
                    "start_sec": start * time_step,
                    "end_sec": end * time_step,
                    "duration_sec": (end - start) * time_step,
                    "mean_confidence": float(np.mean(frame_conf[start:end])),
                }
            )
        start = end
    return segments


def collapse_segments(segments: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    collapsed = []
    for segment in segments:
        if collapsed and collapsed[-1]["phoneme"] == segment["phoneme"]:
            previous = collapsed[-1]
            previous["end_sec"] = segment["end_sec"]
            previous["duration_sec"] = float(previous["end_sec"]) - float(previous["start_sec"])
            previous["mean_confidence"] = (
                float(previous["mean_confidence"]) + float(segment["mean_confidence"])
            ) / 2.0
        else:
            collapsed.append(dict(segment))
    return collapsed


def phoneme_report(audio: np.ndarray, sample_rate: int, model_bundle: dict) -> dict[str, object]:
    feature_extractor = model_bundle["feature_extractor"]
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits[0]
        probs = torch.softmax(logits, dim=-1)
        frame_conf, pred_ids = torch.max(probs, dim=-1)

    pred_ids_np = pred_ids.detach().cpu().numpy()
    frame_conf_np = frame_conf.detach().cpu().numpy()
    decoded = tokenizer.decode(pred_ids_np.tolist(), skip_special_tokens=True)
    time_step = float(model.config.inputs_to_logits_ratio) / sample_rate
    segments = collapse_segments(frame_segments(pred_ids_np, frame_conf_np, tokenizer, time_step))
    first = segments[0] if segments else {}
    b_like = bool(first) and str(first["phoneme"]).lower().startswith("b")
    return {
        "decoded": decoded,
        "segments": segments,
        "first_phoneme": first.get("phoneme", ""),
        "first_start_sec": first.get("start_sec", ""),
        "first_end_sec": first.get("end_sec", ""),
        "first_confidence": first.get("mean_confidence", ""),
        "first_is_b_like": b_like,
    }


def rule_mask_window(row: dict[str, str], stubs: dict[str, Path], audio_len: int, sample_rate: int, args: argparse.Namespace) -> dict[str, object]:
    del audio_len, sample_rate
    subtype = row["b_subtype"].upper()
    stub_path = stubs.get(subtype)
    stub_duration_sec = ""
    if stub_path:
        stub_rate, stub_audio = read_wav_float(stub_path)
        stub_duration_sec = len(stub_audio) / stub_rate if stub_rate else 0.0
        stub_ms = float(stub_duration_sec) * 1000.0
    else:
        stub_ms = args.mask_min_ms
    mask_ms = min(max(stub_ms + args.mask_extra_ms, args.mask_min_ms), args.mask_max_ms)
    return {
        "stub_available": bool(stub_path),
        "stub_path": str(stub_path or ""),
        "stub_duration_sec": stub_duration_sec,
        "rule_mask_duration_sec": mask_ms / 1000.0,
    }


def format_segments(segments: list[dict[str, float | str]], count: int = 8) -> str:
    parts = []
    for seg in segments[:count]:
        parts.append(
            "{}@{:.3f}-{:.3f}".format(
                seg["phoneme"],
                float(seg["start_sec"]),
                float(seg["end_sec"]),
            )
        )
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    rows = select_rows(read_csv(args.manifest), args)
    stubs = list_stubs(args.stub_root)
    device = choose_device(args.device)

    args.outdir.mkdir(parents=True, exist_ok=True)
    local_only = not args.allow_download
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, local_files_only=local_only)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model, local_files_only=local_only)
    model = AutoModelForCTC.from_pretrained(args.model, local_files_only=local_only).to(device)
    model.eval()
    model_bundle = {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }

    output_rows = []
    for row in rows:
        sample_rate, audio = load_audio(Path(row["audio_path"]), args.target_sample_rate)
        energy = energy_region(audio, sample_rate)
        rule = rule_mask_window(row, stubs, len(audio), sample_rate, args)
        neural = phoneme_report(audio, sample_rate, model_bundle)

        rule_mask_start_sec = float(energy["start_sec"])
        rule_mask_end_sec = min(len(audio) / sample_rate, rule_mask_start_sec + float(rule["rule_mask_duration_sec"]))
        neural_boundary = neural["first_end_sec"] if neural["first_is_b_like"] else ""
        warning = ""
        if not neural["first_is_b_like"]:
            warning = "neural_first_not_b_like"
        elif neural_boundary != "":
            delta_ms = (float(neural_boundary) - rule_mask_end_sec) * 1000.0
            if abs(delta_ms) > 60.0:
                warning = "neural_rule_boundary_disagree"

        output_rows.append(
            {
                "audio_path": row["audio_path"],
                "speaker": row["speaker"],
                "word": row["word"],
                "oracle_subtype": row["b_subtype"],
                "first_vowel": row.get("b_first_vowel", ""),
                "duration_sec": f"{len(audio) / sample_rate:.4f}",
                "energy_start_sec": f"{float(energy['start_sec']):.4f}",
                "energy_end_sec": f"{float(energy['end_sec']):.4f}",
                "energy_threshold_db": f"{float(energy['threshold_db']):.2f}",
                "rule_mask_start_sec": f"{rule_mask_start_sec:.4f}",
                "rule_mask_end_sec": f"{rule_mask_end_sec:.4f}",
                "rule_mask_duration_sec": f"{float(rule['rule_mask_duration_sec']):.4f}",
                "stub_available": str(rule["stub_available"]).lower(),
                "stub_path": rule["stub_path"],
                "stub_duration_sec": f"{float(rule['stub_duration_sec']):.4f}" if rule["stub_duration_sec"] != "" else "",
                "wav2vec_decoded": str(neural["decoded"]),
                "wav2vec_first_segments": format_segments(neural["segments"]),
                "wav2vec_first_phoneme": str(neural["first_phoneme"]),
                "wav2vec_first_start_sec": f"{float(neural['first_start_sec']):.4f}" if neural["first_start_sec"] != "" else "",
                "wav2vec_first_end_sec": f"{float(neural['first_end_sec']):.4f}" if neural["first_end_sec"] != "" else "",
                "wav2vec_first_confidence": f"{float(neural['first_confidence']):.4f}" if neural["first_confidence"] != "" else "",
                "wav2vec_first_is_b_like": str(neural["first_is_b_like"]).lower(),
                "neural_suggested_boundary_sec": f"{float(neural_boundary):.4f}" if neural_boundary != "" else "",
                "warning": warning,
            }
        )

    csv_path = args.outdir / "b_initial_neural_assist.csv"
    write_csv(csv_path, output_rows)
    b_like_count = sum(row["wav2vec_first_is_b_like"] == "true" for row in output_rows)
    warning_count = sum(bool(row["warning"]) for row in output_rows)
    summary = {
        "manifest": str(args.manifest),
        "model": args.model,
        "device": str(device),
        "selected_count": len(output_rows),
        "wav2vec_first_b_like_count": b_like_count,
        "wav2vec_first_b_like_rate": round(b_like_count / len(output_rows), 4) if output_rows else None,
        "warning_count": warning_count,
        "output_csv": str(csv_path),
        "note": "Neural outputs are assistive only. Keep energy/rule fallback until validation is strong.",
    }
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in output_rows[:20]:
        print(
            "{speaker} {word} {subtype}: energy={energy_start}->{rule_end}, "
            "wav2vec={phoneme} {start}->{end} warn={warning}".format(
                speaker=row["speaker"],
                word=row["word"],
                subtype=row["oracle_subtype"],
                energy_start=row["energy_start_sec"],
                rule_end=row["rule_mask_end_sec"],
                phoneme=row["wav2vec_first_phoneme"],
                start=row["wav2vec_first_start_sec"],
                end=row["wav2vec_first_end_sec"],
                warning=row["warning"],
            )
        )


if __name__ == "__main__":
    main()
