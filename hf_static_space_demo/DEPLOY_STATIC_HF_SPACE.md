# Deploy Static Demo To Hugging Face Spaces

This version is free because it uses the Static SDK. It does not run live ASR or
TTS, but it publicly shows authorized preset samples, cached ASR results, and
cached Ryan/Piper voice output.

Run from the repo root:

```powershell
conda activate win_ai
cd "<repo-root>"
```

Create the free static Space:

```powershell
hf repos create <hf-username>/<neutral-space-name> `
  --type space `
  --space-sdk static `
  --public `
  --exist-ok
```

Upload only the static demo folder:

```powershell
hf upload <hf-username>/<neutral-space-name> hf_static_space_demo . `
  --repo-type space `
  --exclude "__pycache__/*" `
  --exclude "*.pyc" `
  --exclude "static_*.log" `
  --exclude "static_*.err" `
  --commit-message "Deploy static electrolaryngeal ASR demo"
```

The public URL will be:

```text
https://huggingface.co/spaces/<hf-username>/<neutral-space-name>
```

Do not upload the project-level `data/`, `experiments/`, or `Cleaned Sound/`
folders.
