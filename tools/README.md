# Tools Index

Use these as the current default entry points.

## Current Recommended

- `audio_auto_process.py`
  - Best current B/P word-list preparation script.
  - Converts to 16 kHz mono, removes DC, applies light high-pass filtering, normalizes session speech RMS, segments words, then normalizes each word clip to a consistent active speech RMS range.

- `evaluate_bp_session_shift_fixes.py`
  - Current diagnostic script for checking whether cross-session failure is caused by loudness, timing, and feature distribution shift.
  - Defaults to the v2 Dave train/validation manifests.

- `compare_bp_cross_session_four_arm.py`
  - Current fair HuBERT/MFCC comparison script.
  - Defaults to the v2 Dave train/validation manifests.

## Supporting Utilities

- `build_dataset_manifest.py`
  - Provides B/P word lists and general dataset manifest helpers.

- `inspect_audio_signal.py`, `plot_audio_windows.py`, `learn_stft_spectrogram.py`
  - Learning and inspection tools. Useful for understanding audio, but not the recommended B/P segmentation path.

## Legacy

Old preprocessing, segmentation, and model experiment scripts are archived under:

- `tools/old_method/preprocessing_segmentation_v1`
- `tools/old_method/model_experiments_v1`

Do not start new B/P data preparation from those scripts unless you are intentionally reproducing an old result.
