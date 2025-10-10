# Configuration Guide

## Overview

This project uses [Hydra](https://hydra.cc/) for configuration management. Configurations are composable YAML files in the `configs/` directory.

## Configuration Structure

```
configs/
├── train.yaml           # Main training config
├── eval.yaml            # Main evaluation config
├── data/               # Dataset configurations
├── model/              # Model architectures
├── trainer/            # PyTorch Lightning trainer settings
├── callbacks/          # Training callbacks
├── logger/             # Experiment loggers
├── experiment/         # Full experiment configs
├── paths/              # Path configurations
└── hydra/              # Hydra framework settings
```

## Basic Usage

### Override Single Parameter

```bash
python src/train.py data.batch_size=64
```

### Override Config Group

```bash
python src/train.py data=surge model=surge_ffn
```

### Use Experiment Config

```bash
python src/train.py experiment=surge/baseline
```

Experiment configs override all other settings.

## Configuration Groups

### Data Configs (`configs/data/`)

Specify dataset and data loading parameters:

- `surge.yaml` - Surge XT synthesizer dataset
- `ksin.yaml` - K-oscillator sine wave dataset
- `kosc.yaml` - K-oscillator complex waveform dataset
- `nsynth.yaml` - NSynth dataset
- `fsd.yaml` - FreeSound Dataset

**Key parameters:**
- `dataset_root` - Path to dataset directory
- `batch_size` - Batch size for training
- `num_workers` - Number of data loading workers
- `sample_rate` - Audio sample rate
- `signal_duration_seconds` - Length of audio clips

### Model Configs (`configs/model/`)

Define model architecture and hyperparameters:

- `surge_ffn.yaml` - Feed-forward network for Surge
- `surge_flow.yaml` - Flow matching model for Surge
- `surge_flowvae.yaml` - VAE with flow decoder for Surge
- `ksin_flow_matching.yaml` - Flow matching for KSIN

**Key parameters:**
- `net` - Network architecture configuration
- `optimizer` - Optimizer settings
- `scheduler` - Learning rate scheduler
- `compile` - Whether to use torch.compile

### Trainer Configs (`configs/trainer/`)

PyTorch Lightning trainer settings:

- `default.yaml` - Default trainer configuration
- `gpu.yaml` - Single GPU training
- `ddp.yaml` - Distributed training
- `cpu.yaml` - CPU-only training

**Key parameters:**
- `accelerator` - Hardware accelerator (gpu, cpu, mps)
- `devices` - Device indices to use
- `max_epochs` - Maximum training epochs
- `precision` - Training precision (32, 16, bf16)
- `strategy` - Training strategy (ddp, deepspeed)

### Callbacks (`configs/callbacks/`)

Training callbacks for monitoring and control:

- `model_checkpoint.yaml` - Save best models
- `early_stopping.yaml` - Stop training early
- `lr_monitor.yaml` - Log learning rate
- `rich_progress_bar.yaml` - Enhanced progress bar

### Loggers (`configs/logger/`)

Experiment tracking:

- `wandb.yaml` - Weights & Biases
- `tensorboard.yaml` - TensorBoard
- `mlflow.yaml` - MLflow
- `csv.yaml` - CSV logger

### Experiment Configs (`configs/experiment/`)

Complete experiment configurations that override all other settings.

Structure:
```yaml
# @package _global_

defaults:
  - override /data: surge
  - override /model: surge_ffn
  - override /trainer: gpu

data:
  batch_size: 128
  
model:
  net:
    hidden_dim: 512
    
trainer:
  max_epochs: 100
```

## Common Patterns

### Training Configuration

Minimal training command:
```bash
python src/train.py data=surge model=surge_ffn
```

Full configuration:
```bash
python src/train.py \
    data=surge \
    model=surge_ffn \
    trainer=gpu \
    logger=wandb \
    callbacks=default \
    data.batch_size=128 \
    trainer.max_epochs=100 \
    seed=42
```

CUDA_VISIBLE_DEVICES=2 python src/train.py     data=surge     model=surge_flow     data.dataset_root=datasets/experiment_1     data.batch_size=128     data.num_workers=11     trainer.devices=[0]     trainer.accelerator=gpu

### Evaluation Configuration

```bash
python src/eval.py \
    ckpt_path=path/to/checkpoint.ckpt \
    data=surge \
    model=surge_ffn \
    mode=test
```

## Creating Custom Configs

### Custom Data Config

Create `configs/data/my_dataset.yaml`:

```yaml
_target_: src.data.surge_datamodule.SurgeDataModule

dataset_root: datasets/my_dataset
batch_size: 64
num_workers: 4
sample_rate: 44100
signal_duration_seconds: 4.0
```

### Custom Experiment

Create `configs/experiment/my_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - override /data: surge
  - override /model: surge_ffn
  - override /trainer: gpu
  - override /logger: wandb

task_name: my_experiment
tags: ["surge", "baseline"]

data:
  batch_size: 128
  num_workers: 12

model:
  optimizer:
    lr: 0.001

trainer:
  max_epochs: 100
  devices: [0]
```

Use with:
```bash
python src/train.py experiment=my_experiment
```

## Configuration Priority

Priority from highest to lowest:

1. Command-line overrides
2. Experiment config
3. Specific group configs (data, model, trainer)
4. Main config (train.yaml, eval.yaml)

## Special Parameters

### Global Parameters

- `seed` - Random seed for reproducibility
- `train` - Whether to run training (default: true)
- `test` - Whether to run testing (default: false)
- `ckpt_path` - Path to checkpoint for resuming/evaluation
- `task_name` - Name of the task for logging
- `tags` - List of tags for experiment tracking

### Path Configuration

Configured in `configs/paths/default.yaml`:

- `root_dir` - Project root directory
- `data_dir` - Data directory
- `log_dir` - Logging directory
- `work_dir` - Working directory for outputs

## Advanced Usage

### Multi-run

Run multiple experiments with different parameters:

```bash
python src/train.py -m data.batch_size=32,64,128
```

### Hyperparameter Search

Using Optuna:

```bash
python src/train.py -m hparams_search=ksin_optuna
```

### Debug Modes

- `debug=limit` - Run with limited data
- `debug=overfit` - Test overfitting capability
- `debug=profiler` - Profile performance

## Environment Variables

- `PROJECT_ROOT` - Project root (auto-detected)
- `HYDRA_FULL_ERROR` - Show full Hydra errors (set to 1)
- `CUDA_VISIBLE_DEVICES` - GPU device visibility

## Tips

- Use experiment configs for reproducible research
- Keep individual config files small and focused
- Override specific parameters via command line
- Use defaults lists to compose configurations
- Organize experiments in subdirectories (e.g., `experiment/surge/`)
- Document non-obvious parameter choices in config files

