# Setup Guide

## Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- VST3 plugins (required for Surge XT dataset generation)

## Installation

### Using uv (faster alternative)

```bash
uv pip install -r requirements.txt
```

## GPU Setup

For GPU training, ensure PyTorch is installed with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (e.g., `cu121` for CUDA 12.1).

## VST Setup

Required for generating Surge XT datasets.

1. Download Surge XT from https://surge-synthesizer.github.io/
2. Install the VST3 plugin
3. Place preset files in the `presets/` directory
4. Update VST path in config file

## Verify Installation

Run tests to verify installation:

```bash
pytest tests/
```

## Directory Setup

Create required directories for datasets and outputs:

```bash
mkdir -p datasets/experiment_1
mkdir -p logs
mkdir -p checkpoints
```

## Configuration

The project uses Hydra for configuration management. Configs are stored in `configs/`:

- `configs/data/` - Dataset configurations
- `configs/model/` - Model architectures
- `configs/trainer/` - Training parameters
- `configs/experiment/` - Complete experiment setups

## Environment Variables

Optional environment variables:

- `PROJECT_ROOT` - Project root directory (auto-detected)
- `LOGURU_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `CUDA_VISIBLE_DEVICES` - GPU device selection

## Troubleshooting

**Import errors:**
- Ensure you're in the project root directory
- The project uses rootutils to manage paths automatically

**CUDA out of memory:**
- Reduce batch size: `data.batch_size=32`
- Use gradient accumulation: `trainer.accumulate_grad_batches=2`
- Enable mixed precision: `trainer.precision=16`

**VST plugin not found:**
- Verify plugin path with `--plugin_path` argument
- Check plugin compatibility (VST3 format required)
- Ensure plugin is properly installed on your system


