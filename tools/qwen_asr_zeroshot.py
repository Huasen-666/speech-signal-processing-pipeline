"""Zero-shot Qwen3-ASR evaluation on electrolaryngeal speech.

This script runs Qwen3-ASR without patient adaptation on the fixed Dave
holdout set and writes a predictions CSV compatible with
tools/score_asr_predictions.py.

The test set remains 100% real patient audio. This is only a zero-shot
robustness probe, not a fine-tuning script.

Example:
  conda run --no-capture-output -n qwen_asr python tools\\qwen_asr_zeroshot.py ^
    --eval-csv data\\asr\\dave_holdout_s6\\test_s6.csv ^
    --outdir experiments\\qwen_asr_zeroshot\\dave_holdout_s6

Then score with the same scorer used for the Whisper experiments:
  conda run --no-capture-output -n qwen_asr python tools\\score_asr_predictions.py ^
    --pred-csv experiments\\qwen_asr_zeroshot\\dave_holdout_s6\\qwen3_asr_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CSV = PROJECT_ROOT / "data" / "asr" / "dave_holdout_s6" / "test_s6.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "experiments" / "qwen_asr_zeroshot" / "dave_holdout_s6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot Qwen3-ASR on a CSV of patient speech clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-csv-name", default="qwen3_asr_predictions.csv")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--fallback-model-id", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="do not fall back to the smaller Qwen3-ASR model if loading fails",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="passed to from_pretrained; use auto, cuda, cpu, or a custom device map",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="model/input dtype; auto uses bf16/fp16 on CUDA and fp32 on CPU",
    )
    parser.add_argument("--language", default="English", help="force output language")
    parser.add_argument(
        "--prompt",
        default="Transcribe the electrolaryngeal speech in clear English. Return only the spoken words.",
        help="optional Qwen3-ASR context/hotword prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="debug: only process first N clips")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="write partial predictions every N clips",
    )
    return parser.parse_args()


def resolve_project_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_rows(eval_csv: Path, limit: int | None) -> list[dict[str, str]]:
    rows = list(csv.DictReader(eval_csv.open("r", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"No rows found in {eval_csv}")
    required = {"clip_id", "audio_path", "text"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{eval_csv} is missing required columns: {sorted(missing)}")
    return rows[:limit] if limit else rows


def choose_dtype(torch_module, dtype_arg: str):
    if dtype_arg == "float32":
        return torch_module.float32
    if dtype_arg == "float16":
        return torch_module.float16
    if dtype_arg == "bfloat16":
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def first_model_device_and_dtype(model) -> tuple[Any, Any]:
    try:
        device = model.device
        dtype = model.dtype
        return device, dtype
    except Exception:
        first_param = next(model.parameters())
        return first_param.device, first_param.dtype


def load_qwen_model(model_id: str, args: argparse.Namespace):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    dtype = choose_dtype(torch, args.dtype)
    kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype != "float32" or torch.cuda.is_available():
        kwargs["dtype"] = dtype

    print(f"[load] processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[load] model: {model_id}", flush=True)
    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)

    model.eval()
    device, model_dtype = first_model_device_and_dtype(model)
    print(f"[load] ready: device={device}, dtype={model_dtype}", flush=True)
    return processor, model, device, model_dtype


def load_with_fallback(args: argparse.Namespace):
    try:
        return args.model_id, *load_qwen_model(args.model_id, args)
    except Exception as exc:
        if args.no_fallback or not args.fallback_model_id:
            raise
        print(
            f"[warn] failed to load {args.model_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(f"[warn] falling back to {args.fallback_model_id}", file=sys.stderr, flush=True)
        return args.fallback_model_id, *load_qwen_model(args.fallback_model_id, args)


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def transcribe_one(processor, model, audio_path: Path, args: argparse.Namespace) -> str:
    import torch

    device, dtype = first_model_device_and_dtype(model)
    if args.prompt:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": args.prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "path": str(audio_path)}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"language {args.language}<asr_text>"}],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            continue_final_message=True,
        )
    else:
        inputs = processor.apply_transcription_request(
            audio=str(audio_path),
            language=args.language,
        )
    inputs = inputs.to(device, dtype)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    try:
        prediction = processor.decode(
            generated_ids,
            return_format="transcription_only",
        )[0]
    except TypeError:
        raw = processor.decode(generated_ids)[0]
        prediction = raw.split("<asr_text>", 1)[-1]
    return clean_prediction(prediction)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "clip_id",
        "reference",
        "prediction",
        "audio_path",
        "model_id",
        "elapsed_sec",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_csv = resolve_project_path(args.eval_csv)
    outdir = resolve_project_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_csv = outdir / args.output_csv_name

    rows = load_rows(eval_csv, args.limit)
    print(f"[data] eval_csv={eval_csv}", flush=True)
    print(f"[data] clips={len(rows)}", flush=True)

    actual_model_id, processor, model, _, _ = load_with_fallback(args)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        audio_path = resolve_project_path(row["audio_path"])
        reference = row["text"].strip()
        print(f"[{index:03d}/{len(rows):03d}] {clip_id}", flush=True)

        t0 = time.perf_counter()
        status = "ok"
        error = ""
        prediction = ""
        try:
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            prediction = transcribe_one(processor, model, audio_path, args)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            print(f"[error] {clip_id}: {error}", file=sys.stderr, flush=True)

        elapsed = time.perf_counter() - t0
        predictions.append(
            {
                "clip_id": clip_id,
                "reference": reference,
                "prediction": prediction,
                "audio_path": str(audio_path),
                "model_id": actual_model_id,
                "elapsed_sec": f"{elapsed:.3f}",
                "status": status,
                "error": error,
            }
        )
        print(f"       ref: {reference}", flush=True)
        print(f"      pred: {prediction}", flush=True)

        if args.save_every > 0 and index % args.save_every == 0:
            write_predictions(pred_csv, predictions)

    write_predictions(pred_csv, predictions)
    summary = {
        "eval_csv": str(eval_csv),
        "prediction_csv": str(pred_csv),
        "requested_model_id": args.model_id,
        "actual_model_id": actual_model_id,
        "fallback_model_id": args.fallback_model_id,
        "language": args.language,
        "prompt": args.prompt,
        "n_rows": len(rows),
        "n_ok": sum(1 for r in predictions if r["status"] == "ok"),
        "n_error": sum(1 for r in predictions if r["status"] != "ok"),
        "total_elapsed_sec": round(time.perf_counter() - started, 3),
    }
    summary_path = outdir / "qwen3_asr_zeroshot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {pred_csv}", flush=True)
    print(f"[done] wrote {summary_path}", flush=True)
    print(
        "[next] score with: "
        f"python tools\\score_asr_predictions.py --pred-csv \"{pred_csv}\"",
        flush=True,
    )


if __name__ == "__main__":
    main()
