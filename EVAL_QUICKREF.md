# CPU Evaluation Quick Reference

Quick commands for evaluating models on CPU with your test dataset.

## TL;DR - Complete Workflow

```bash
# Set your checkpoint path
CKPT="./logs/train/flow_multi/dataset_20k_40k-2025-10-24_12-59-16/checkpoints/last.ckpt"

# 1. Generate predictions (CPU)
uv run python src/eval.py \
    experiment=surge/flow_full \
    ckpt_path=${CKPT} \
    mode=predict \
    trainer=cpu \
    callbacks=prediction_writer \
    data.predict_file=test.h5 \
    data.dataset_root=./datasets/surge-20k \
    paths.output_dir=audio_samples

# 2. Render to audio
PRED_DIR=$(find audio_samples -type d -name "predictions" | head -n 1)
uv run python scripts/predict_vst_audio.py \
    ${PRED_DIR} \
    audio_samples/audio \
    --plugin_path "plugins/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so" \
    --preset_path "presets/surge-base.vstpreset" \
    --skip-spectrogram

# 3. Compute metrics
uv run python scripts/compute_audio_metrics_no_pesto.py \
    audio_samples/audio \
    audio_samples/metrics \
    --num_workers 4

# 4. View results
cat audio_samples/metrics/aggregated_metrics.csv
```

## Step-by-Step Commands

### 1. Predictions Only
```bash
uv run python src/eval.py \
    experiment=surge/flow_full \
    ckpt_path=./logs/train/flow_multi/dataset_20k_40k-2025-10-24_12-59-16/checkpoints/last.ckpt \
    mode=predict \
    trainer=cpu \
    callbacks=prediction_writer \
    data.predict_file=test.h5 \
    data.dataset_root=./datasets/surge-20k \
    paths.output_dir=audio_samples
```

### 2. Render Audio Only
```bash
uv run python scripts/predict_vst_audio.py \
    audio_samples/predictions \
    audio_samples/audio \
    --plugin_path "plugins/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so" \
    --preset_path "presets/surge-base.vstpreset" \
    --skip-spectrogram
```

### 3. Compute Metrics Only
```bash
uv run python scripts/compute_audio_metrics_no_pesto.py \
    audio_samples/audio \
    audio_samples/metrics \
    --num_workers 4
```

## Common Variations

### Use Different Dataset
```bash
data.dataset_root=./datasets/surge-100k
```

### Use Validation Set
```bash
data.predict_file=val.h5
```

### Use Different Experiment Config
```bash
experiment=surge/flow_simple  # or ffn_simple, etc.
```

### Adjust CPU Workers
```bash
--num_workers 8  # or 2, 4, etc. based on CPU cores
```

## Output Structure

```
audio_samples/
├── predictions/
│   ├── predictions.h5
│   └── targets.h5
├── audio/
│   ├── sample_0/
│   │   ├── target.wav
│   │   └── pred.wav
│   └── sample_1/...
└── metrics/
    ├── metrics.csv
    └── aggregated_metrics.csv
```

## Find Your Checkpoint

```bash
# List all checkpoints
find logs/train -name "*.ckpt"

# Find best checkpoint (if available)
find logs/train -name "epoch_*.ckpt"

# Find last checkpoint
find logs/train -name "last.ckpt"
```

## Verify Files Exist

```bash
# Check dataset
ls -lh ./datasets/surge-20k/test.h5

# Check VST plugin
ls -lh "plugins/Surge XT.vst3/Contents/x86_64-linux/Surge XT.so"

# Check preset
ls -lh presets/surge-base.vstpreset

# Check checkpoint
ls -lh ./logs/train/flow_multi/dataset_20k_40k-2025-10-24_12-59-16/checkpoints/last.ckpt
```

## Troubleshooting One-Liners

```bash
# Force CPU mode
trainer=cpu trainer.accelerator=cpu

# Find predictions directory
find audio_samples -type d -name "predictions"

# Check metric results
tail -n +1 audio_samples/metrics/*.csv

# Count rendered samples
ls -d audio_samples/audio/sample_* | wc -l
```

---

See [TUTORIAL_CPU_EVALUATION.md](TUTORIAL_CPU_EVALUATION.md) for detailed explanation.
