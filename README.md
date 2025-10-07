## Audio to DAW

Synthesizer parameter estimation from audio using deep learning.

Project inspired by the work of [Ben Hayes](https://github.com/ben-hayes/synth-permutations/tree/main)

## Project Structure

```
.
├── src/
│   ├── data/                 # Dataset generation
│   │   ├── generate_vst_dataset.py  # Generate Surge VST datasets
│   │   ├── vst/              # VST parameter specifications
│   │   └── config_example.yaml
│   └── models/               # Simplified model implementation ⭐
│       ├── config.yaml       # Model configuration
│       ├── model.py          # Flow matching & feedforward models
│       ├── dataloader.py     # HDF5 data loading
│       ├── train.py          # Training script
│       ├── evaluate.py       # Evaluation script
│       ├── predict.py        # Inference on audio files
│       └── README.md         # Detailed model documentation
│
└── src_hydra/                # Reference implementation (complex)
    └── ...                   # Original PyTorch Lightning code
```

## Quick Start

### 1. Generate Dataset

```bash
# Create a config file (see src/data/config_example.yaml)
python src/data/generate_vst_dataset.py --config my_config.yaml
```

This will generate HDF5 files with:
- Audio waveforms
- Mel spectrograms  
- Synthesizer parameters

### 2. Train a Model

```bash
cd src/models

# Edit config.yaml to point to your dataset
# Then train:
python train.py
```

The model will train and save checkpoints to `checkpoints/`

### 3. Evaluate

```bash
python evaluate.py
```

This runs inference on the test set and saves predictions.

### 4. Predict on New Audio

```bash
python predict.py path/to/audio.wav --output params.npy
```

## Model Implementation

Two versions are available:

### Simplified Version (Recommended for Experimentation)
- **Location**: `src/models/`
- **Style**: Simple PyTorch, single config file, minimal abstractions
- **Best for**: Quick experiments, understanding the algorithm, modifications
- See `src/models/README.md` for details

### Reference Version (Production-Ready)
- **Location**: `src_hydra/`  
- **Style**: PyTorch Lightning, Hydra configs, extensive logging
- **Best for**: Reproducing published results, large-scale training
- Based on [Ben Hayes' work](https://github.com/ben-hayes/synth-permutations)

## Models Available

### Flow Matching (Recommended)
- Uses a diffusion-like process to generate parameters
- Classifier-free guidance for better quality
- More robust to multi-modal distributions

### Feedforward Baseline
- Direct encoder-decoder prediction
- Faster inference but lower quality
- Good for quick prototyping

## Dataset Generation Notes

**Performance metrics** (from testing):
- ~2.6 s/sample/core with Surge XT
- With 40 cores: ~36k samples/hour
- Note: min_loudness filtering causes ~33% regeneration

**Important settings**:
- Set `min_loudness: -55` to match original experiment
- Use `param_set: "surge_simple"` for 92 parameters (recommended)
- Use `param_set: "surge_full"` for 189 parameters (advanced)

## Development Log

### Recent Updates
- ✅ Simplified model implementation in `src/models/`
- ✅ Flow matching and feedforward models
- ✅ Simple config-based training
- ✅ Inference scripts for audio files
- ✅ Multi-file dataset support with automatic splitting
- ✅ Efficient handling of large datasets (100s of GB)

### Earlier Progress
- Working dataset generation with parallelization
- Surge parameter specifications (simple & full)
- Chunked HDF5 dataset format
- Mel spectrogram computation
- Matched original experiment parameters

## Requirements

```bash
# For dataset generation
pip install h5py librosa mido pedalboard pyloudnorm pyyaml tqdm

# For model training
pip install torch numpy h5py pyyaml tqdm

# For reference implementation
pip install pytorch-lightning hydra-core wandb
```

## References

- [Ben Hayes - Synth Permutations](https://github.com/ben-hayes/synth-permutations)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)