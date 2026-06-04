# Speech Signal Processing Pipeline

This repository documents a practical speech signal processing workflow for turning raw recordings into cleaner speech signals and segmentation-ready metadata. The project is built as an engineering learning track: each script is small enough to study, but structured so the code can grow into a full preprocessing and machine learning pipeline.

## Goals

- Inspect raw speech recordings before modifying them.
- Visualize waveform, short-time energy, and spectrogram structure.
- Implement STFT concepts from first principles.
- Prepare for later stages: denoising, VAD, segmentation, word-level labels, and machine learning baselines.

## Current Pipeline

1. `tools/inspect_audio_signal.py`
   - Reads a WAV file.
   - Converts PCM samples to floating-point waveform values.
   - Reports sample rate, duration, peak level, RMS level, DC offset, clipping count, and rough energy-based VAD statistics.
   - Saves a JSON report, rough VAD CSV, waveform overview, and spectrogram preview.

2. `tools/plot_audio_windows.py`
   - Selects representative local windows from a long recording.
   - Exports waveform and spectrogram plots for first speech, typical speech, loudest window, and quietest window.

3. `tools/learn_stft_spectrogram.py`
   - Builds a spectrogram manually using frame slicing, Hann windowing, FFT, magnitude, and dB conversion.
   - Creates teaching plots for waveform, window shape, single-frame spectrum, and manual STFT spectrogram.

## Example Usage

Place a local WAV file under `data/raw/`, or pass an explicit input path:

```powershell
python tools\inspect_audio_signal.py --input "data\raw\example.wav" --outdir "experiments\step1_audio_inspection"
python tools\plot_audio_windows.py --input "data\raw\example.wav" --outdir "experiments\step2_audio_windows"
python tools\learn_stft_spectrogram.py --input "data\raw\example.wav" --outdir "experiments\step3_stft"
```

## Privacy Note

Raw recordings are intentionally excluded from version control. Speech data can contain personal, clinical, or identifying information. This repository should contain code, notes, anonymized plots, and reproducible experiment descriptions, not private audio.

## Roadmap

The project is organized as a layered speech pipeline:

```text
Raw recording
-> conservative DSP cleanup
-> QA report
-> segmentation
-> protocol-based labels
-> optional neural assist
-> ML baseline
-> DSP-exportable classifier
```

The full roadmap is documented in `docs/roadmap.md`. A key design rule is that modern neural tools are used as optional assistants or cross-checks, not as the only source of truth.
