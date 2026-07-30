from dataclasses import asdict, dataclass

import numpy as np

from speech_pipeline.normalization import amplitude_to_dbfs, peak, rms
from speech_pipeline.quality import energy_segments, frame_rms_db


@dataclass
class SegmentConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    merge_gap_sec: float = 0.2
    min_duration_sec: float = 0.15
    padding_sec: float = 0.05


def _merge_overlapping_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start_sec, end_sec in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def add_padding(
    segments: list[tuple[float, float]],
    duration_sec: float,
    padding_sec: float,
) -> list[tuple[float, float]]:
    padded = [
        (max(0.0, start_sec - padding_sec), min(duration_sec, end_sec + padding_sec))
        for start_sec, end_sec in segments
    ]
    return _merge_overlapping_segments(padded)


def detect_energy_segments(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    config: SegmentConfig | None = None,
) -> tuple[list[dict], dict]:
    """Detect candidate speech segments from short-time RMS energy."""
    config = config or SegmentConfig()
    mono = np.asarray(audio, dtype=np.float32)
    duration_sec = len(mono) / sample_rate

    frame_time, frame_db = frame_rms_db(
        mono,
        sample_rate,
        frame_ms=config.frame_ms,
        hop_ms=config.hop_ms,
    )
    raw_segments = energy_segments(
        frame_time,
        frame_db,
        threshold_dbfs=threshold_dbfs,
        frame_ms=config.frame_ms,
        merge_gap_sec=config.merge_gap_sec,
        min_duration_sec=config.min_duration_sec,
    )
    padded_segments = add_padding(raw_segments, duration_sec, config.padding_sec)

    segment_rows = []
    for index, (start_sec, end_sec) in enumerate(padded_segments, start=1):
        start_sample = max(0, round(start_sec * sample_rate))
        end_sample = min(len(mono), round(end_sec * sample_rate))
        chunk = mono[start_sample:end_sample]
        segment_rows.append(
            {
                "segment_index": index,
                "start_sec": round(start_sample / sample_rate, 3),
                "end_sec": round(end_sample / sample_rate, 3),
                "duration_sec": round((end_sample - start_sample) / sample_rate, 3),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "peak_dbfs": round(amplitude_to_dbfs(peak(chunk)), 3) if len(chunk) else -240.0,
                "rms_dbfs": round(amplitude_to_dbfs(rms(chunk)), 3) if len(chunk) else -240.0,
            }
        )

    total_segment_duration = sum(row["duration_sec"] for row in segment_rows)
    summary = {
        "threshold_dbfs": float(threshold_dbfs),
        "config": asdict(config),
        "raw_segment_count": len(raw_segments),
        "padded_segment_count": len(segment_rows),
        "total_segment_duration_sec": round(total_segment_duration, 3),
        "segment_ratio": round(total_segment_duration / max(duration_sec, 1e-12), 4),
    }
    return segment_rows, summary
