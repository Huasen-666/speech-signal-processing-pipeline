import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate phoneme pseudo-labels with wav2vec2-phoneme.")
    parser.add_argument("--input", type=Path, required=True, help="Input WAV/audio file.")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phoneme_labels/wav2vec2"))
    parser.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--trim-silence", action="store_true", help="Trim leading/trailing silence before inference.")
    parser.add_argument("--trim-top-db", type=float, default=35.0)
    return parser.parse_args()


def read_audio(path: Path, target_sample_rate: int, trim_silence: bool, trim_top_db: float) -> tuple[int, np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32)
        sample_rate = target_sample_rate
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    offset_samples = 0
    if trim_silence and len(audio):
        trimmed, index = librosa.effects.trim(audio, top_db=trim_top_db)
        offset_samples = int(index[0])
        audio = trimmed.astype(np.float32)
    return sample_rate, audio, offset_samples


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def frame_segments(
    pred_ids: np.ndarray,
    frame_conf: np.ndarray,
    tokenizer: Wav2Vec2CTCTokenizer,
    time_step: float,
) -> list[dict[str, str]]:
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
                    "start_sec": f"{start * time_step:.4f}",
                    "end_sec": f"{end * time_step:.4f}",
                    "duration_sec": f"{(end - start) * time_step:.4f}",
                    "mean_confidence": f"{float(np.mean(frame_conf[start:end])):.4f}",
                    "start_frame": str(start),
                    "end_frame": str(end),
                    "token_id": str(token_id),
                }
            )
        start = end
    return segments


def collapse_segments(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    collapsed = []
    for seg in segments:
        if collapsed and collapsed[-1]["phoneme"] == seg["phoneme"]:
            prev = collapsed[-1]
            prev["end_sec"] = seg["end_sec"]
            prev["duration_sec"] = f"{float(prev['end_sec']) - float(prev['start_sec']):.4f}"
            prev["end_frame"] = seg["end_frame"]
            prev["mean_confidence"] = f"{(float(prev['mean_confidence']) + float(seg['mean_confidence'])) / 2.0:.4f}"
        else:
            collapsed.append(dict(seg))
    return collapsed


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phoneme",
        "start_sec",
        "end_sec",
        "duration_sec",
        "mean_confidence",
        "start_frame",
        "end_frame",
        "token_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sample_rate, audio, offset_samples = read_audio(
        args.input,
        args.target_sample_rate,
        args.trim_silence,
        args.trim_top_db,
    )
    device = choose_device(args.device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device)
    model.eval()

    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits[0]
        probs = torch.softmax(logits, dim=-1)
        frame_conf, pred_ids = torch.max(probs, dim=-1)

    pred_ids_np = pred_ids.detach().cpu().numpy()
    frame_conf_np = frame_conf.detach().cpu().numpy()
    decoded = tokenizer.decode(pred_ids_np.tolist(), skip_special_tokens=True)

    time_step = float(model.config.inputs_to_logits_ratio) / sample_rate
    offset_sec = offset_samples / sample_rate
    raw_segments = frame_segments(pred_ids_np, frame_conf_np, tokenizer, time_step)
    collapsed = collapse_segments(raw_segments)
    if offset_sec:
        for rows in (raw_segments, collapsed):
            for row in rows:
                row["start_sec"] = f"{float(row['start_sec']) + offset_sec:.4f}"
                row["end_sec"] = f"{float(row['end_sec']) + offset_sec:.4f}"

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    raw_csv = args.outdir / f"{stem}_wav2vec2_phoneme_frames.csv"
    collapsed_csv = args.outdir / f"{stem}_wav2vec2_phoneme_segments.csv"
    report_path = args.outdir / f"{stem}_wav2vec2_phoneme_report.json"
    write_csv(raw_csv, raw_segments)
    write_csv(collapsed_csv, collapsed)

    report = {
        "input": str(args.input),
        "model": args.model,
        "device": str(device),
        "sample_rate": sample_rate,
        "duration_sec": len(audio) / sample_rate if sample_rate else 0.0,
        "trim_silence": args.trim_silence,
        "trim_offset_sec": offset_sec,
        "time_step_sec": time_step,
        "decoded_phonemes": decoded,
        "segment_count": len(collapsed),
        "raw_frame_segments_csv": str(raw_csv),
        "collapsed_segments_csv": str(collapsed_csv),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    for row in collapsed[:30]:
        print(
            f"{row['phoneme']} {row['start_sec']}..{row['end_sec']} "
            f"conf={row['mean_confidence']}"
        )


if __name__ == "__main__":
    main()
