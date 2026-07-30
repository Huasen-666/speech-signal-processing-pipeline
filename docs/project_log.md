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

Implementation update:

- Added the first Level 2 conservative DSP cleaning baseline.
- Added reusable modules for filtering, normalization, audio IO, and quality summaries.
- The baseline intentionally avoids aggressive denoising and writes before/after QA outputs for review.
- Updated the cleaner to use speech-only RMS normalization by default after QA showed that full-audio RMS is distorted by long silences and isolated peaks.
- Added adaptive short-time energy thresholding so speech-frame selection can follow each recording's own noise floor and high-energy distribution.

Level 3 implementation update:

- Added energy-based segmentation that exports candidate speech clips and CSV/JSON manifests.
- Added merge-gap, minimum-duration, and padding controls to make segment boundaries reviewable rather than final truth.

## 2026-06-29

HuBERT v2 fairness decision for B/P initial-consonant detection:

- The early HuBERT replacement experiments used `last_hidden_state` plus mean/std pooling. That is a weak configuration for initial consonants, so those results should not be used to dismiss HuBERT.
- A stricter later test used HuBERT hidden states with learnable layer weighting on onset windows, but it still did not clearly outperform the MFCC/log-mel onset baseline on Dave train -> DRB validation.
- The remaining fair test is a temporal onset version: HuBERT middle layers, onset-only windows, preserved frame sequence, and small attention pooling. This should be judged only on cross-session validation.

Experimental constraints before running HuBERT v2:

- HuBERT has about 20 ms frame stride, so a 150-220 ms onset window contains only about 7-11 frames. It can model coarse onset trajectory, but it may still smooth out 5-30 ms burst/VOT cues.
- Explicit DSP onset features remain important because they measure short-time physical cues such as early RMS, ZCR, high-band ratio, low-band ratio, peak timing, and spectral centroid.
- The decisive comparison should include four arms: MFCC/log-mel onset baseline, HuBERT middle-layer mean/std, HuBERT middle-layer temporal attention, and a hybrid HuBERT-plus-explicit-onset model.
- Attention pooling is preferred before a temporal CNN because the dataset is small and the HuBERT onset sequence has very few frames.
- Burst/onset alignment must be checked, because inconsistent alignment between sessions can cause a false model failure.
- With about 100 validation clips, small accuracy gaps are not decisive. Report multiple seeds, mean +/- standard deviation, and B/P precision/recall.

Decision rule:

- If HuBERT temporal or hybrid clearly beats the MFCC/log-mel baseline across seeds and sessions, it becomes the phone/PC teacher candidate.
- If all four arms remain near 0.65-0.78 cross-session accuracy, the bottleneck is probably session variability, segmentation/alignment, and data coverage rather than the model family.

Session-shift diagnosis:

- Dave train -> DRB validation showed a large recording/session mismatch: validation RMS was about 5 dB lower, onset was delayed from about 0.050 s to 0.117 s, and validation clips were about 190 ms longer.
- The session-level RMS shift is larger than the B/P level difference inside the training session, so absolute level and boundary timing can dominate the real consonant cue.
- A quick fix experiment tested split-level RMS normalization, per-session CMVN, relative onset features, and target-session threshold calibration.
- Best quick-fix result was `onset_logmel_mfcc`, 220 ms, split CMVN, and 5 labeled B plus 5 labeled P calibration clips: 0.633 accuracy on the remaining 90 validation clips.
- Without target labels, the best quick fix was about 0.602 accuracy using MFCC/log-mel onset at 220 ms with split CMVN.
- Split-level RMS normalization did not change onset timing, suggesting the delayed validation onset is not only a loudness-gate problem; it may reflect different trimming, weak onset production, or recording protocol.

Updated conclusion:

- The model can fit same-session B/P data, but the current cross-session detector is not product-ready.
- The next technical priority is recording/session normalization and collection of multiple sessions per patient, not simply a larger HuBERT model.

Segmentation v2 implementation:

- Added a new recommended dataset preparation script, now named `tools/audio_auto_process.py`.
- The v2 path converts recordings to 16 kHz mono, removes DC offset, applies a light 60 Hz high-pass filter, performs speech-mask session RMS normalization, then segments words with adaptive energy VAD.
- After segmentation, each exported word clip is normalized again using active speech RMS so quiet words are brought into the same dBFS range without relying on full-clip silence.
- The script writes warning flags for very quiet clips, clips that hit the maximum gain cap, post-normalization RMS outliers, and peak-protection scaling.
- On Dave's first B/P recording, v2 detected 100/100 words and normalized clip active RMS to about -20.02 dBFS with 0.14 dB standard deviation.
- On Dave's DRB validation recording, v2 detected 100/100 words but flagged several quiet/outlier clips (`bead`, `beat`, `peak`, `pink`, `plain`, `page`) for manual review or rerecording.
- Older segmentation scripts are now kept only for reproducibility and learning history; the v2 script should be the default path for new B/P consecutive recordings.
- Archived old preprocessing/model scripts and old generated outputs under `tools/old_method`, `experiments/old_method`, and `data/old_method` so the active project root points to the current v2 workflow.
