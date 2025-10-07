# Surge Parameter Estimation

Simple PyTorch implementation for predicting synthesizer parameters from audio.

## What's Here

- **model.py** - Flow matching & feedforward models
- **train.py** - Training loop
- **evaluate.py** - Run inference on test set  
- **predict.py** - Predict params from audio file
- **config.yaml** - All settings in one place

## Usage

```bash
# 1. Compute normalization stats (optional but helps)
python compute_stats.py ../data/dataset/

# 2. Train
python train.py

# 3. Evaluate
python evaluate.py

# 4. Predict on new audio
python predict.py audio.wav -o params.npy
```

## Data Setup

Dump your `.h5` files in a folder:
```
data/dataset/
├── dataset_0000.h5
├── dataset_0001.h5
├── ...
└── stats.npz  (computed by compute_stats.py)
```

Each HDF5 needs: `mel_spec` (N, 2, 128, 401) and `param_array` (N, 92)

The dataloader auto-splits into train/val/test (80/10/10 by default). Change in config if needed.

## Config

Edit `config.yaml` for your paths and settings:

## Models

**Flow Matching** (recommended) - Better quality, uses diffusion-like process with classifier-free guidance. Slower inference.

**Feedforward** - Direct prediction. Fast but lower quality.

## Large Datasets

Handles 100s of GB fine. Lazy loads from HDF5, only keeps one batch in memory at a time.