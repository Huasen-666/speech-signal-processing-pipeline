# Roadmap

This project follows a layered speech signal processing roadmap. The goal is to build from interpretable DSP foundations toward modern neural tools while preserving raw data integrity and measurable quality control.

## Guiding Principle

Modern neural tools should assist, cross-check, or accelerate the pipeline. They should not replace the raw recording, conservative DSP baseline, or quality reports as the only source of truth.

For this project, this matters because consonant classes such as `P/T/K/S/Z` often contain burst or fricative energy that can resemble noise. Aggressive neural denoising may improve listening quality while damaging the exact acoustic cues needed for classification.

## Level 0: Raw Data Governance

Purpose: protect the data before any algorithm touches it.

Core work:

- Preserve original recordings without overwrite.
- Use consistent file naming.
- Record sample rate, channel count, bit depth, source device, speaker/session ID, and recording protocol.
- Keep a processing history for every derived file.
- Exclude private audio from Git.

Expected outputs:

```text
data/raw/                # ignored by git
data/processed/          # ignored by git
recording_manifest.csv
processing_log.json
```

## Level 1: Traditional DSP Foundation

Purpose: understand the signal with explainable methods.

Core topics:

- PCM representation.
- RMS and dBFS.
- DC offset removal.
- High-pass and band-pass filtering.
- STFT and spectrograms.
- Band energy.
- Notch filtering for narrowband interference.
- Energy-based VAD.

Current scripts:

```text
tools/inspect_audio_signal.py
tools/plot_audio_windows.py
tools/learn_stft_spectrogram.py
```

## Level 2: Speech Cleaning and QA Pipeline

Purpose: clean conservatively and prove what changed.

Cleaning is not the final goal. QA is the final goal. Every cleaning step should produce before/after evidence.

Required report fields:

```text
duration
sample rate
peak dBFS
RMS dBFS
clipping %
noise floor
speech/silence ratio
detected segment count
segment duration histogram
warnings
```

Planned modules:

```text
src/speech_pipeline/filters.py
src/speech_pipeline/normalization.py
src/speech_pipeline/quality.py
tools/clean_audio_baseline.py
```

Initial implementation status:

- `filters.py`: DC removal, high-pass, low-pass, band-pass, and notch filtering.
- `normalization.py`: RMS normalization and peak protection.
- `quality.py`: before/after audio summaries, adaptive energy thresholding, VAD-style speech ratio, segment count, and warnings.
- `clean_audio_baseline.py`: command-line baseline cleaner with QA outputs.

## Level 3: Segmentation and Labeling

Purpose: convert clean recordings into training-ready examples.

Core work:

- VAD-based speech segmentation.
- Segment merging and padding.
- Patient/speaker section classification when needed.
- Word list or protocol-based alignment.
- Metadata export.
- Manual spot checks.

Expected outputs:

```text
segments/
word_audio/
manifest_segments.csv
manifest_words.csv
label_review_sheet.csv
```

Initial implementation status:

- `src/speech_pipeline/segmentation.py`: converts frame-level energy decisions into padded segment metadata.
- `tools/segment_audio_energy.py`: exports candidate segment WAVs and manifests for review.

## Level 4: Modern Neural Assist

Purpose: use modern neural tools as optional support, not as the only truth.

Potential tools:

- Neural denoising.
- Neural VAD.
- ASR transcription.
- Forced alignment.
- Pretrained embeddings such as wav2vec2, HuBERT, or WavLM.

Rules:

- Keep a switch to disable neural denoising.
- Always compare raw, conservative DSP, and neural-assisted versions.
- Check whether bursts and fricatives are weakened after enhancement.
- Treat ASR and forced alignment outputs as candidates that may need QA.

## Level 5: ML Baseline

Purpose: build measurable classification baselines for speech units.

Initial target:

```text
B/P/T/D/K/G/S/Z classification
```

Baseline path:

- Extract interpretable features such as log-mel, MFCC, short-time energy, zero-crossing rate, and band energy.
- Train a small classifier.
- Evaluate on held-out speakers or sessions.
- Report confusion matrix and per-class precision/recall.
- Compare with pretrained embedding features.
- Explore DSP-exportable classifiers for deployment constraints.

Expected outputs:

```text
experiments/ml_baseline/
confusion_matrix.png
classification_report.json
selected_features.json
```
