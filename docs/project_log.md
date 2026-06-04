# Project Log

## 2026-06-04

Started the speech signal processing pipeline project.

Completed initial learning scripts:

- Audio inspection: format, duration, peak, RMS, clipping, DC offset, and rough energy VAD.
- Local window visualization: waveform and spectrogram views for representative audio regions.
- STFT learning: manual frame slicing, Hann windowing, FFT, magnitude conversion, and spectrogram plotting.

Current technical focus:

- Understanding PCM audio representation.
- Measuring audio level with peak and RMS.
- Reading waveform and spectrogram plots.
- Building intuition for short-time analysis.

Next planned stage:

- Build conservative speech cleaning with high-pass filtering, level normalization, and before/after quality checks.

Roadmap decision:

- Added a formal layered roadmap from raw data governance through DSP foundations, cleaning QA, segmentation and labeling, optional neural assist, and ML baseline development.
- Defined Level 4 neural tools as assistive rather than authoritative because neural denoising can suppress consonant bursts and fricatives that matter for `P/T/K/S/Z` classification.
