import math

import numpy as np


def amplitude_to_dbfs(value: float, eps: float = 1e-12) -> float:
    return 20.0 * math.log10(max(float(value), eps))


def frame_rms_db(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, round(sample_rate * hop_ms / 1000.0))
    mono = np.asarray(audio, dtype=np.float32)

    if len(mono) < frame_len:
        mono = np.pad(mono, (0, frame_len - len(mono)))

    starts = np.arange(0, len(mono) - frame_len + 1, hop_len)
    power = mono.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    frame_power = (cumulative[starts + frame_len] - cumulative[starts]) / frame_len
    frame_rms = np.sqrt(np.maximum(frame_power, 0.0))
    frame_db = np.array([amplitude_to_dbfs(value) for value in frame_rms], dtype=np.float32)
    return starts / sample_rate, frame_db


def estimate_speech_threshold_dbfs(
    frame_db: np.ndarray,
    noise_percentile: float = 10.0,
    speech_percentile: float = 95.0,
    noise_margin_db: float = 15.0,
    speech_margin_db: float = 25.0,
    min_threshold_dbfs: float = -60.0,
    max_threshold_dbfs: float = -35.0,
) -> tuple[float, dict[str, float]]:
    """Estimate an energy VAD threshold from the recording's own level distribution."""
    if len(frame_db) == 0:
        return -45.0, {
            "method": "adaptive_energy",
            "fallback": 1.0,
            "threshold_dbfs": -45.0,
        }

    noise_floor = float(np.percentile(frame_db, noise_percentile))
    high_energy = float(np.percentile(frame_db, speech_percentile))
    threshold = max(noise_floor + noise_margin_db, high_energy - speech_margin_db)
    threshold = min(max(threshold, min_threshold_dbfs), max_threshold_dbfs)

    return round(float(threshold), 3), {
        "method": "adaptive_energy",
        "noise_percentile": float(noise_percentile),
        "speech_percentile": float(speech_percentile),
        "noise_floor_dbfs": round(noise_floor, 3),
        "high_energy_dbfs": round(high_energy, 3),
        "noise_margin_db": float(noise_margin_db),
        "speech_margin_db": float(speech_margin_db),
        "min_threshold_dbfs": float(min_threshold_dbfs),
        "max_threshold_dbfs": float(max_threshold_dbfs),
        "threshold_dbfs": round(float(threshold), 3),
    }


def energy_segments(
    frame_time: np.ndarray,
    frame_db: np.ndarray,
    threshold_dbfs: float = -45.0,
    frame_ms: float = 25.0,
    merge_gap_sec: float = 0.2,
    min_duration_sec: float = 0.15,
) -> list[tuple[float, float]]:
    voiced = frame_db >= threshold_dbfs
    segments = []
    start = None
    frame_dur = frame_ms / 1000.0

    for idx, is_voiced in enumerate(voiced):
        if is_voiced and start is None:
            start = idx
        elif not is_voiced and start is not None:
            segments.append([float(frame_time[start]), float(frame_time[idx - 1] + frame_dur)])
            start = None

    if start is not None:
        segments.append([float(frame_time[start]), float(frame_time[-1] + frame_dur)])

    merged = []
    for seg_start, seg_end in segments:
        if not merged or seg_start - merged[-1][1] > merge_gap_sec:
            merged.append([seg_start, seg_end])
        else:
            merged[-1][1] = seg_end

    return [
        (seg_start, seg_end)
        for seg_start, seg_end in merged
        if seg_end - seg_start >= min_duration_sec
    ]


def summarize_audio(
    audio: np.ndarray,
    sample_rate: int,
    speech_threshold_dbfs: float = -45.0,
    threshold_info: dict[str, float] | None = None,
) -> dict[str, float | int | list[float] | dict[str, float]]:
    mono = np.asarray(audio, dtype=np.float32)
    duration = len(mono) / sample_rate
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if len(mono) else 0.0
    frame_time, frame_db = frame_rms_db(mono, sample_rate)
    segments = energy_segments(frame_time, frame_db, threshold_dbfs=speech_threshold_dbfs)
    segment_durations = [end - start for start, end in segments]
    voiced_duration = float(sum(segment_durations))
    clipped = int(np.count_nonzero(np.abs(mono) >= 0.999969))

    return {
        "duration_sec": round(duration, 3),
        "sample_rate_hz": int(sample_rate),
        "peak_dbfs": round(amplitude_to_dbfs(peak), 3),
        "rms_dbfs": round(amplitude_to_dbfs(rms), 3),
        "dc_offset": round(float(np.mean(mono)) if len(mono) else 0.0, 8),
        "clipping_samples": clipped,
        "clipping_ratio": round(clipped / max(1, len(mono)), 8),
        "noise_floor_p10_dbfs": round(float(np.percentile(frame_db, 10)), 3),
        "energy_median_p50_dbfs": round(float(np.percentile(frame_db, 50)), 3),
        "energy_p90_dbfs": round(float(np.percentile(frame_db, 90)), 3),
        "speech_threshold_dbfs": float(speech_threshold_dbfs),
        "speech_threshold_info": threshold_info or {"method": "fixed", "threshold_dbfs": float(speech_threshold_dbfs)},
        "speech_ratio": round(voiced_duration / max(duration, 1e-12), 4),
        "detected_segment_count": len(segments),
        "segment_duration_histogram_sec": [
            int(np.count_nonzero(np.array(segment_durations) < 0.5)),
            int(np.count_nonzero((np.array(segment_durations) >= 0.5) & (np.array(segment_durations) < 1.0))),
            int(np.count_nonzero((np.array(segment_durations) >= 1.0) & (np.array(segment_durations) < 2.0))),
            int(np.count_nonzero(np.array(segment_durations) >= 2.0)),
        ],
    }


def quality_warnings(summary: dict[str, float | int | list[float]]) -> list[str]:
    warnings = []
    if float(summary["peak_dbfs"]) > -0.1:
        warnings.append("Peak is very close to 0 dBFS; check clipping risk.")
    if float(summary["clipping_ratio"]) > 0.0001:
        warnings.append("Clipping ratio is non-trivial; inspect clipped regions.")
    if abs(float(summary["dc_offset"])) > 0.005:
        warnings.append("DC offset is high; remove DC before filtering or feature extraction.")
    if float(summary["speech_ratio"]) < 0.05:
        warnings.append("Detected speech ratio is very low; verify VAD threshold or recording protocol.")
    return warnings
