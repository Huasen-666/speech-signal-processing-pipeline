import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT))

from speech_pipeline.audio_io import read_wav_float, write_wav_float
from speech_pipeline.filters import bandpass, remove_dc_offset
from speech_pipeline.ml_features import resample_linear
from speech_pipeline.normalization import peak_protect, rms_normalize_with_mask
from speech_pipeline.quality import estimate_speech_threshold_dbfs, frame_rms_db, quality_warnings, summarize_audio
from speech_pipeline.segmentation import SegmentConfig, detect_energy_segments

from enhance_scripted_long_recording import (
    allocate_word_intervals,
    dataset_lookup,
    function_word_lookup,
    group_segments_to_lines,
    read_csv,
    replacement_args,
    replace_word_onset_in_place,
    subtype_lookup,
    word_duration_weight,
    write_csv,
)
from render_b_initial_stub_library_demo import list_stubs, load_stub_for_group, prepare_stub_for_insert
from render_bascom_content_heavy_reading_demo import LONG_STORY_LINES


DEFAULT_INPUT = PROJECT_ROOT.parent / "Cleaned Sound" / "David" / "Word details David_16k_mono.wav"
DEFAULT_MODEL = "facebook/hubert-base-ls960"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DSP-clean David's long reading, classify B/P vowel context with frozen HuBERT features, "
            "and replace weak initial consonant cues with the local stub library."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=Path("experiments/phone_prototype/david_long_story_hubert_v2"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/metadata/dataset_manifest.csv"))
    parser.add_argument("--b-manifest", type=Path, default=Path("data/metadata/b_subtype_manifest.csv"))
    parser.add_argument("--p-manifest", type=Path, default=Path("data/metadata/p_subtype_manifest.csv"))
    parser.add_argument("--function-root", type=Path, default=PROJECT_ROOT.parent / "word_library" / "function_words")
    parser.add_argument("--b-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "B")
    parser.add_argument("--p-stub-root", type=Path, default=PROJECT_ROOT.parent / "consonant" / "P")
    parser.add_argument("--speaker-for-word-durations", default="bascom")
    parser.add_argument("--hubert-model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hubert-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hubert-window-ms", type=float, default=900.0)
    parser.add_argument("--hubert-onset-ms", type=float, default=260.0)
    parser.add_argument("--classification", choices=["hubert", "hybrid", "guarded_hybrid", "script"], default="guarded_hybrid")
    parser.add_argument("--min-hubert-confidence", type=float, default=0.42)
    parser.add_argument("--min-hubert-margin", type=float, default=0.015)
    parser.add_argument("--training-speakers", nargs="*", default=[], help="Empty means all usable speakers.")
    parser.add_argument("--skip-labels", nargs="*", default=["B_L"])
    parser.add_argument("--low-hz", type=float, default=80.0)
    parser.add_argument("--high-hz", type=float, default=7600.0)
    parser.add_argument("--target-rms-dbfs", type=float, default=-24.0)
    parser.add_argument("--max-peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.45)
    parser.add_argument("--min-duration-sec", type=float, default=0.25)
    parser.add_argument("--padding-sec", type=float, default=0.03)
    parser.add_argument("--word-merge-gap-sec", type=float, default=0.12)
    parser.add_argument("--word-min-duration-sec", type=float, default=0.08)
    parser.add_argument("--word-padding-sec", type=float, default=0.015)
    parser.add_argument("--line-padding-sec", type=float, default=0.08)
    parser.add_argument("--line-boundary-gap-weight", type=float, default=0.22)
    parser.add_argument("--crossfade-ms", type=float, default=18.0)
    parser.add_argument("--stub-time-scale", type=float, default=1.0)
    parser.add_argument("--stub-time-mode", choices=["speed", "tempo"], default="speed")
    parser.add_argument("--stub-fit-mode", choices=["crop", "stretch"], default="crop")
    parser.add_argument("--level-mode", choices=["match_rms", "dave_peak"], default="match_rms")
    parser.add_argument("--stub-rms-ratio", type=float, default=1.18)
    parser.add_argument("--mask-extra-ms", type=float, default=35.0)
    parser.add_argument("--mask-min-ms", type=float, default=75.0)
    parser.add_argument("--mask-max-ms", type=float, default=260.0)
    parser.add_argument("--min-target-room-ms", type=float, default=45.0)
    parser.add_argument("--ab-gap-sec", type=float, default=1.5)
    parser.add_argument("--preview-sec", type=float, default=90.0)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def str_true(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def speech_mask(audio: np.ndarray, sample_rate: int, threshold_dbfs: float, frame_ms: float, hop_ms: float) -> np.ndarray:
    frame_times, frame_db = frame_rms_db(audio, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    mask = np.zeros(len(audio), dtype=bool)
    for start_sec in frame_times[frame_db >= threshold_dbfs]:
        start = round(float(start_sec) * sample_rate)
        mask[start : min(len(mask), start + frame_len)] = True
    return mask


def conservative_dsp_cleanup(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    _, raw_frame_db = frame_rms_db(audio, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    threshold_dbfs, threshold_info = estimate_speech_threshold_dbfs(raw_frame_db)
    before = summarize_audio(audio, sample_rate, speech_threshold_dbfs=threshold_dbfs, threshold_info=threshold_info)

    cleaned = remove_dc_offset(audio)
    cleaned = bandpass(cleaned, sample_rate, low_hz=args.low_hz, high_hz=args.high_hz)
    mask = speech_mask(cleaned, sample_rate, threshold_dbfs, args.frame_ms, args.hop_ms)
    cleaned, rms_info = rms_normalize_with_mask(cleaned, mask, target_dbfs=args.target_rms_dbfs)
    cleaned, peak_info = peak_protect(cleaned, max_peak_dbfs=args.max_peak_dbfs)

    after = summarize_audio(cleaned, sample_rate, speech_threshold_dbfs=threshold_dbfs, threshold_info=threshold_info)
    report = {
        "pipeline": "remove_dc_offset -> bandpass -> speech_only_rms_normalize -> peak_protect",
        "speech_threshold_dbfs": threshold_dbfs,
        "speech_threshold_info": threshold_info,
        "stages": {
            "bandpass": {"low_hz": args.low_hz, "high_hz": min(args.high_hz, sample_rate / 2.0 * 0.95)},
            "rms_normalization": rms_info,
            "peak_protection": peak_info,
        },
        "before": before,
        "after": after,
        "warnings": {"before": quality_warnings(before), "after": quality_warnings(after)},
    }
    return cleaned.astype(np.float32), report


def detect_segments_for_script(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> tuple[list[dict], dict]:
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


def detect_fine_segments(
    audio: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    args: argparse.Namespace,
) -> list[dict]:
    start_sample = max(0, min(len(audio), round(start_sec * sample_rate)))
    end_sample = max(start_sample + 1, min(len(audio), round(end_sec * sample_rate)))
    chunk = audio[start_sample:end_sample]
    _, frame_db = frame_rms_db(chunk, sample_rate, frame_ms=args.frame_ms, hop_ms=args.hop_ms)
    threshold_dbfs, _ = estimate_speech_threshold_dbfs(frame_db)
    config = SegmentConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        merge_gap_sec=args.word_merge_gap_sec,
        min_duration_sec=args.word_min_duration_sec,
        padding_sec=args.word_padding_sec,
    )
    local_segments, _ = detect_energy_segments(chunk, sample_rate, threshold_dbfs, config=config)
    absolute = []
    for segment in local_segments:
        row = dict(segment)
        row["start_sec"] = round(float(segment["start_sec"]) + start_sample / sample_rate, 3)
        row["end_sec"] = round(float(segment["end_sec"]) + start_sample / sample_rate, 3)
        row["start_sample"] = int(round(row["start_sec"] * sample_rate))
        row["end_sample"] = int(round(row["end_sec"] * sample_rate))
        row["duration_sec"] = round(float(row["end_sec"]) - float(row["start_sec"]), 3)
        absolute.append(row)
    return absolute


def fine_word_intervals(
    audio: np.ndarray,
    sample_rate: int,
    line_start: float,
    line_end: float,
    words: list[str],
    data_by_word: dict[str, dict[str, str]],
    function_by_word: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[list[tuple[str, float, float, float]], str, int]:
    fine_segments = detect_fine_segments(audio, sample_rate, line_start, line_end, args)
    weights = [word_duration_weight(word, data_by_word, function_by_word) for word in words]
    if len(fine_segments) >= len(words):
        groups = group_segments_to_lines(fine_segments, weights, args)
        intervals = []
        for word, weight, (start_idx, end_idx) in zip(words, weights, groups):
            intervals.append(
                (
                    word,
                    float(fine_segments[start_idx]["start_sec"]),
                    float(fine_segments[end_idx]["end_sec"]),
                    weight,
                )
            )
        return intervals, "fine_energy_dp", len(fine_segments)

    return (
        allocate_word_intervals(line_start, line_end, words, data_by_word, function_by_word),
        "duration_weight_fallback",
        len(fine_segments),
    )


def cache_key(parts: dict[str, object]) -> str:
    text = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def trim_leading_silence(audio: np.ndarray, sample_rate: int, preroll_ms: float = 25.0) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame_len = max(1, round(sample_rate * 0.02))
    hop_len = max(1, round(sample_rate * 0.005))
    if len(audio) < frame_len:
        return audio
    starts = np.arange(0, len(audio) - frame_len + 1, hop_len)
    power = audio.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    frame_power = (cumulative[starts + frame_len] - cumulative[starts]) / frame_len
    frame_db = 20.0 * np.log10(np.maximum(np.sqrt(frame_power), 1e-12))
    threshold = max(float(np.percentile(frame_db, 10)) + 12.0, float(np.percentile(frame_db, 95)) - 35.0, -60.0)
    active = np.flatnonzero(frame_db >= threshold)
    if len(active) == 0:
        return audio
    start = max(0, int(starts[int(active[0])]) - round(sample_rate * preroll_ms / 1000.0))
    return audio[start:]


@dataclass(frozen=True)
class TrainingItem:
    audio_path: Path
    label: str
    speaker: str
    word: str


class HubertCentroidClassifier:
    def __init__(self, args: argparse.Namespace, cache_dir: Path):
        try:
            from transformers import AutoFeatureExtractor, HubertModel
        except ImportError as exc:
            raise RuntimeError("Missing transformers. Run this script with the win_ai environment.") from exc

        self.args = args
        self.sample_rate = 16000
        self.cache_dir = cache_dir
        self.device = choose_device(args.hubert_device)
        local_only = not args.allow_download
        log(f"Loading HuBERT on {self.device}: {args.hubert_model}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(args.hubert_model, local_files_only=local_only)
        self.model = HubertModel.from_pretrained(args.hubert_model, local_files_only=local_only).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.centroids: dict[str, dict[str, np.ndarray]] = {}
        self.mean_std: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.counts: dict[str, dict[str, int]] = {}

    def prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            mono = resample_linear(mono, sample_rate, self.sample_rate).astype(np.float32)
        mono = trim_leading_silence(mono, self.sample_rate)
        max_len = max(1, round(self.sample_rate * self.args.hubert_window_ms / 1000.0))
        if len(mono) < max_len:
            mono = np.pad(mono, (0, max_len - len(mono)))
        else:
            mono = mono[:max_len]
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        if peak > 1.0:
            mono = mono / peak
        return mono.astype(np.float32)

    def pool(self, hidden: torch.Tensor) -> np.ndarray:
        frames = hidden[0].detach().cpu().numpy().astype(np.float32)
        full_mean = frames.mean(axis=0)
        full_std = frames.std(axis=0)
        stride = float(getattr(self.model.config, "inputs_to_logits_ratio", 320))
        frame_sec = stride / self.sample_rate
        onset_frames = max(1, min(len(frames), round((self.args.hubert_onset_ms / 1000.0) / frame_sec)))
        onset = frames[:onset_frames]
        onset_mean = onset.mean(axis=0)
        onset_std = onset.std(axis=0)
        return np.concatenate([full_mean, full_std, onset_mean, onset_std]).astype(np.float32)

    def extract_feature(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        prepared = self.prepare_audio(audio, sample_rate)
        inputs = self.feature_extractor(prepared, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs)
        return self.pool(output.last_hidden_state)

    def cached_feature_for_item(self, item: TrainingItem) -> np.ndarray:
        stat = item.audio_path.stat()
        key = cache_key(
            {
                "audio_path": str(item.audio_path.resolve()).lower(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "label": item.label,
                "model": self.args.hubert_model,
                "window_ms": self.args.hubert_window_ms,
                "onset_ms": self.args.hubert_onset_ms,
            }
        )
        cache_path = self.cache_dir / "training" / f"{key}.npz"
        if cache_path.exists():
            return np.load(cache_path)["feature"].astype(np.float32)
        sample_rate, audio = read_wav_float(item.audio_path)
        feature = self.extract_feature(audio, sample_rate)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            feature=feature,
            label=item.label,
            speaker=item.speaker,
            word=item.word,
            audio_path=str(item.audio_path),
        )
        return feature

    def fit(self, items: list[TrainingItem]) -> None:
        by_prefix: dict[str, list[tuple[str, np.ndarray]]] = {"B": [], "P": []}
        for idx, item in enumerate(items, start=1):
            feature = self.cached_feature_for_item(item)
            by_prefix[item.label[0]].append((item.label, feature))
            if idx == 1 or idx % 20 == 0 or idx == len(items):
                log(f"HuBERT reference feature {idx}/{len(items)} ready")

        for prefix, values in by_prefix.items():
            if not values:
                continue
            labels = [label for label, _ in values]
            matrix = np.stack([feature for _, feature in values]).astype(np.float32)
            mean = matrix.mean(axis=0, keepdims=True)
            std = matrix.std(axis=0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            normalized = (matrix - mean) / std
            self.mean_std[prefix] = (mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32))
            self.centroids[prefix] = {}
            self.counts[prefix] = {}
            for label in sorted(set(labels)):
                label_matrix = normalized[[idx for idx, name in enumerate(labels) if name == label]]
                centroid = label_matrix.mean(axis=0)
                norm = np.linalg.norm(centroid)
                self.centroids[prefix][label] = (centroid / max(norm, 1e-12)).astype(np.float32)
                self.counts[prefix][label] = int(label_matrix.shape[0])
            log(f"HuBERT centroid labels for {prefix}: {self.counts[prefix]}")

    def predict(self, audio: np.ndarray, sample_rate: int, prefix: str) -> dict[str, object]:
        if prefix not in self.centroids:
            return {"predicted": "", "confidence": 0.0, "margin": 0.0, "scores": {}}
        feature = self.extract_feature(audio, sample_rate)
        mean, std = self.mean_std[prefix]
        normalized = (feature - mean) / std
        normalized = normalized / max(float(np.linalg.norm(normalized)), 1e-12)
        scores = {
            label: float(np.dot(normalized, centroid))
            for label, centroid in self.centroids[prefix].items()
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return {"predicted": "", "confidence": 0.0, "margin": 0.0, "scores": {}}
        score_values = np.array([score for _, score in ordered], dtype=np.float64)
        scaled = (score_values - score_values.max()) / 0.05
        probs = np.exp(scaled)
        probs = probs / max(float(probs.sum()), 1e-12)
        margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else ordered[0][1])
        return {
            "predicted": ordered[0][0],
            "confidence": float(probs[0]),
            "margin": float(margin),
            "scores": scores,
        }


def select_training_items(args: argparse.Namespace) -> list[TrainingItem]:
    allowed_speakers = {speaker.lower() for speaker in args.training_speakers}
    items = []
    for manifest_path, label_col in [(args.b_manifest, "b_subtype"), (args.p_manifest, "p_subtype")]:
        for row in read_csv(manifest_path):
            if not str_true(row.get("usable", "")) or not str_true(row.get("stub_available", "")):
                continue
            if allowed_speakers and row.get("speaker", "").lower() not in allowed_speakers:
                continue
            label = row.get(label_col, "").upper()
            path = Path(row.get("audio_path", ""))
            if not label or not path.exists():
                continue
            items.append(
                TrainingItem(
                    audio_path=path,
                    label=label,
                    speaker=row.get("speaker", ""),
                    word=row.get("word", ""),
                )
            )
    return sorted(items, key=lambda item: (item.label, item.speaker, item.word))


def resolve_label(script_subtype: str, prediction: dict[str, object], args: argparse.Namespace) -> tuple[str, str]:
    if args.classification == "script":
        return script_subtype, "script"
    predicted = str(prediction.get("predicted", ""))
    confidence = float(prediction.get("confidence", 0.0))
    margin = float(prediction.get("margin", 0.0))
    if args.classification == "hubert":
        return predicted, "hubert"
    if args.classification == "guarded_hybrid":
        if predicted == script_subtype and confidence >= args.min_hubert_confidence and margin >= args.min_hubert_margin:
            return predicted, "hubert_confirmed"
        return script_subtype, "script_guarded_fallback"
    if predicted and confidence >= args.min_hubert_confidence and margin >= args.min_hubert_margin:
        return predicted, "hubert"
    return script_subtype, "script_fallback"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample_rate, raw_audio = read_wav_float(args.input)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz WAV, got {sample_rate} Hz")
    log(f"Loaded input: {args.input} ({len(raw_audio) / sample_rate:.2f}s)")

    clean_audio, dsp_report = conservative_dsp_cleanup(raw_audio, sample_rate, args)
    log(
        "DSP cleanup done: "
        f"peak {dsp_report['before']['peak_dbfs']} -> {dsp_report['after']['peak_dbfs']} dBFS, "
        f"RMS {dsp_report['before']['rms_dbfs']} -> {dsp_report['after']['rms_dbfs']} dBFS"
    )

    data_by_word = dataset_lookup(read_csv(args.dataset_manifest), args.speaker_for_word_durations)
    b_subtypes = subtype_lookup(read_csv(args.b_manifest), args.speaker_for_word_durations, "b_subtype")
    p_subtypes = subtype_lookup(read_csv(args.p_manifest), args.speaker_for_word_durations, "p_subtype")
    function_by_word = function_word_lookup(args.function_root)
    b_stubs = list_stubs(args.b_stub_root)
    p_stubs = list_stubs(args.p_stub_root)
    skip_labels = {label.upper() for label in args.skip_labels}
    rargs = replacement_args(args)

    classifier = None
    if args.classification != "script":
        training_items = select_training_items(args)
        if len(training_items) < 10:
            raise ValueError("Not enough HuBERT reference items selected.")
        classifier = HubertCentroidClassifier(args, args.outdir / "hubert_feature_cache")
        classifier.fit(training_items)

    segments, segmentation_info = detect_segments_for_script(clean_audio, sample_rate, args)
    if not segments:
        raise ValueError("No speech segments detected after DSP cleanup.")
    log(f"Detected {len(segments)} speech segments after DSP cleanup")

    line_weights = [
        sum(word_duration_weight(word, data_by_word, function_by_word) for word in line)
        for line in LONG_STORY_LINES
    ]
    line_groups = group_segments_to_lines(segments, line_weights, args)

    enhanced = clean_audio.copy()
    line_rows = []
    word_rows = []
    replacement_count = 0
    skipped_count = 0
    hubert_used_count = 0
    fallback_count = 0

    for line_index, ((start_seg_idx, end_seg_idx), words) in enumerate(zip(line_groups, LONG_STORY_LINES), start=1):
        line_start = max(0.0, float(segments[start_seg_idx]["start_sec"]) - args.line_padding_sec)
        line_end = min(len(clean_audio) / sample_rate, float(segments[end_seg_idx]["end_sec"]) + args.line_padding_sec)
        line_rows.append(
            {
                "line_index": str(line_index),
                "segment_start_index": str(start_seg_idx + 1),
                "segment_end_index": str(end_seg_idx + 1),
                "start_sec": f"{line_start:.3f}",
                "end_sec": f"{line_end:.3f}",
                "duration_sec": f"{line_end - line_start:.3f}",
                "script": " ".join(words),
            }
        )

        intervals, word_alignment_method, fine_segment_count = fine_word_intervals(
            clean_audio,
            sample_rate,
            line_start,
            line_end,
            words,
            data_by_word,
            function_by_word,
            args,
        )
        for word_index, (word, word_start, word_end, weight) in enumerate(intervals, start=1):
            role = "function"
            label = "FUNCTION"
            script_subtype = ""
            resolved_subtype = ""
            resolve_source = ""
            hubert_predicted = ""
            hubert_confidence = 0.0
            hubert_margin = 0.0
            stub_path = ""
            applied = False
            replace_start = replace_end = fitted_len = 0
            skip_reason = ""
            stubs = {}

            if word in b_subtypes:
                role = "content"
                label = "B"
                script_subtype = b_subtypes[word]
                stubs = b_stubs
            elif word in p_subtypes:
                role = "content"
                label = "P"
                script_subtype = p_subtypes[word]
                stubs = p_stubs

            prediction = {"predicted": "", "confidence": 0.0, "margin": 0.0, "scores": {}}
            if role == "content" and classifier is not None:
                start_sample = max(0, round(word_start * sample_rate))
                end_sample = min(len(clean_audio), round(word_end * sample_rate))
                prediction = classifier.predict(clean_audio[start_sample:end_sample], sample_rate, label)
                hubert_predicted = str(prediction["predicted"])
                hubert_confidence = float(prediction["confidence"])
                hubert_margin = float(prediction["margin"])

            if role == "content":
                resolved_subtype, resolve_source = resolve_label(script_subtype, prediction, args)
                hubert_used_count += int(resolve_source.startswith("hubert"))
                fallback_count += int("fallback" in resolve_source)
                if not resolved_subtype:
                    skip_reason = "no_resolved_subtype"
                elif resolved_subtype in skip_labels:
                    skip_reason = f"skip_label_{resolved_subtype}"
                elif resolved_subtype not in stubs:
                    skip_reason = f"missing_stub_{resolved_subtype}"
                else:
                    raw_stub, raw_stub_path = load_stub_for_group(resolved_subtype, stubs, sample_rate)
                    prepared_stub = prepare_stub_for_insert(raw_stub, rargs)
                    applied, replace_start, replace_end, fitted_len = replace_word_onset_in_place(
                        enhanced,
                        sample_rate,
                        word_start,
                        word_end,
                        prepared_stub,
                        rargs,
                        args,
                    )
                    stub_path = str(raw_stub_path)
                    if not applied:
                        skip_reason = "replacement_window_too_short"
                replacement_count += int(applied)
                skipped_count += int(not applied)

            word_rows.append(
                {
                    "line_index": str(line_index),
                    "word_index": str(word_index),
                    "word": word,
                    "role": role,
                    "label": label,
                    "script_subtype": script_subtype,
                    "hubert_predicted_subtype": hubert_predicted,
                    "hubert_confidence": f"{hubert_confidence:.4f}" if role == "content" else "",
                    "hubert_margin": f"{hubert_margin:.4f}" if role == "content" else "",
                    "resolved_subtype": resolved_subtype,
                    "resolve_source": resolve_source,
                    "estimated_word_start_sec": f"{word_start:.3f}",
                    "estimated_word_end_sec": f"{word_end:.3f}",
                    "estimated_word_duration_sec": f"{word_end - word_start:.3f}",
                    "word_alignment_method": word_alignment_method,
                    "line_fine_segment_count": str(fine_segment_count),
                    "replacement_applied": str(applied).lower(),
                    "skip_reason": skip_reason,
                    "replace_start_sec": f"{replace_start / sample_rate:.3f}" if replace_start else "",
                    "replace_end_sec": f"{replace_end / sample_rate:.3f}" if replace_end else "",
                    "replace_duration_sec": f"{(replace_end - replace_start) / sample_rate:.3f}" if replace_end else "",
                    "fitted_stub_duration_sec": f"{fitted_len / sample_rate:.3f}" if fitted_len else "",
                    "stub_path": stub_path,
                    "duration_weight": f"{weight:.4f}",
                }
            )

    enhanced, final_peak_info = peak_protect(enhanced, max_peak_dbfs=args.max_peak_dbfs)
    preview_len = min(len(clean_audio), round(sample_rate * args.preview_sec))
    ab_gap = np.zeros(round(sample_rate * args.ab_gap_sec), dtype=np.float32)

    original_path = args.outdir / "david_word_details_original_amplified.wav"
    clean_path = args.outdir / "david_word_details_dsp_clean.wav"
    enhanced_path = args.outdir / "david_word_details_enhanced_hubert_v2.wav"
    ab_path = args.outdir / "david_word_details_A_clean_B_enhanced_hubert_v2.wav"
    preview_clean_path = args.outdir / "preview_first90_dsp_clean.wav"
    preview_enhanced_path = args.outdir / "preview_first90_enhanced_hubert_v2.wav"
    preview_ab_path = args.outdir / "preview_first90_A_clean_B_enhanced_hubert_v2.wav"

    write_wav_float(original_path, sample_rate, raw_audio)
    write_wav_float(clean_path, sample_rate, clean_audio)
    write_wav_float(enhanced_path, sample_rate, enhanced)
    write_wav_float(ab_path, sample_rate, np.concatenate([clean_audio, ab_gap, enhanced]).astype(np.float32))
    write_wav_float(preview_clean_path, sample_rate, clean_audio[:preview_len])
    write_wav_float(preview_enhanced_path, sample_rate, enhanced[:preview_len])
    write_wav_float(
        preview_ab_path,
        sample_rate,
        np.concatenate([clean_audio[:preview_len], ab_gap, enhanced[:preview_len]]).astype(np.float32),
    )

    line_csv = args.outdir / "alignment_lines.csv"
    word_csv = args.outdir / "alignment_words_hubert.csv"
    write_csv(line_csv, line_rows)
    write_csv(word_csv, word_rows)

    summary = {
        "input": str(args.input),
        "sample_rate_hz": sample_rate,
        "duration_sec": round(len(raw_audio) / sample_rate, 3),
        "script_line_count": len(LONG_STORY_LINES),
        "detected_segment_count": len(segments),
        "classification": args.classification,
        "hubert_model": args.hubert_model if args.classification != "script" else "",
        "hubert_used_count": hubert_used_count,
        "script_fallback_count": fallback_count,
        "replacement_count": replacement_count,
        "skipped_content_count": skipped_count,
        "final_peak_protection": final_peak_info,
        "original_audio": str(original_path),
        "dsp_clean_audio": str(clean_path),
        "enhanced_audio": str(enhanced_path),
        "ab_audio": str(ab_path),
        "preview_clean_audio": str(preview_clean_path),
        "preview_enhanced_audio": str(preview_enhanced_path),
        "preview_ab_audio": str(preview_ab_path),
        "alignment_lines_csv": str(line_csv),
        "alignment_words_csv": str(word_csv),
        "dsp_report": dsp_report,
        "segmentation": segmentation_info,
        "method_note": (
            "V2 uses conservative DSP cleanup before segmentation. B/P content words are still script-aligned, "
            "but the replacement subtype is selected by frozen HuBERT nearest-centroid classification when confidence "
            "and margin pass thresholds; otherwise it falls back to the scripted word subtype."
        ),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
