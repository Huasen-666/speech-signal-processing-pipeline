# Old Method Archive

This folder stores scripts that are useful for history, learning, or reproducing old experiments, but they are not the recommended path for new B/P data preparation.

## Folders

- `preprocessing_segmentation_v1`
  - Older cleaning and energy segmentation scripts.
  - These did not apply the current session-level and per-word dB normalization strategy.

- `model_experiments_v1`
  - Older model training, HuBERT, TinyCNN, and listening-demo scripts.
  - These were useful experiments, but some use older feature pooling, same-session validation, or old manifests.

## Current Replacement

Use this script for new B/P word-list recordings:

```powershell
python tools\audio_auto_process.py --input "data\raw\input.wav" --speaker speaker_id --overwrite
```

The v2 path is designed to keep word clips in a consistent active speech RMS dBFS range and to flag weak words before they pollute training.
