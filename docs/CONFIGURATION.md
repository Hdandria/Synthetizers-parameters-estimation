# Configuration Guide

This document provides a comprehensive guide to the configuration system used in the synth-permutations project. The project uses Hydra for configuration management, allowing for flexible and composable experiment setups.

## Table of Contents

- [Configuration Structure](#configuration-structure)
- [Main Configuration Files](#main-configuration-files)
- [Data Configurations](#data-configurations)
- [Model Configurations](#model-configurations)
- [Experiment Configurations](#experiment-configurations)
- [Training Configurations](#training-configurations)
- [Advanced Usage](#advanced-usage)

## Configuration Structure

The configuration system is organized hierarchically:

```
configs/
├── train.yaml                 # Main training configuration
├── eval.yaml                  # Main evaluation configuration
├── data/                      # Dataset configurations
├── model/                     # Model configurations
├── experiment/                # Pre-configured experiments
├── trainer/                   # Training configurations
├── callbacks/                 # Callback configurations
├── logger/                    # Logging configurations
└── paths/                     # Path configurations
```

## Main Configuration Files

### `configs/train.yaml`

The main training configuration file that defines the default training setup:

```yaml
# @package _global_

defaults:
  - _self_
  - data: ksin                    # Default dataset
  - model: ksin_ff               # Default model
  - callbacks: default           # Default callbacks
  - logger: many_loggers         # Default loggers
  - trainer: gpu                 # Default trainer
  - paths: default              # Default paths
  - extras: default             # Default extras
  - hydra: default              # Hydra settings

  # Optional configurations
  - experiment: null            # Experiment-specific configs
  - hparams_search: null        # Hyperparameter search
  - local: default              # Local machine settings
  - debug: null                 # Debug configurations

# Task settings
task_name: "train"
tags: ["dev"]
train: True
test: True
ckpt_path: null
seed: null
```

**Key Parameters:**
- `task_name`: Name for output directory
- `tags`: Tags for experiment identification
- `train`: Whether to train the model
- `test`: Whether to run testing after training
- `ckpt_path`: Path to checkpoint for resuming training
- `seed`: Random seed for reproducibility

### `configs/eval.yaml`

The main evaluation configuration file:

```yaml
# @package _global_

defaults:
  - _self_
  - data: ksin
  - model: ksin_ff
  - callbacks: default
  - logger: many_loggers
  - trainer: gpu
  - paths: default
  - extras: default
  - hydra: default

# Evaluation settings
task_name: "eval"
mode: "test"  # "test", "val", or "predict"
ckpt_path: null  # Required for evaluation
```

## Data Configurations

### `configs/data/ksin.yaml`

Configuration for the k-sin synthetic dataset:

```yaml
# @package _global_

_target_: src.data.ksin_datamodule.KSinDataModule

# Dataset parameters
k: 4                           # Number of sinusoids
signal_length: 1024            # Signal length in samples
sort_frequencies: true         # Sort frequencies to break symmetry
break_symmetry: false          # Add frequency shifts
shift_test_distribution: false # Different test distribution

# Dataset sizes
train_val_test_sizes: [100000, 10000, 10000]
train_val_test_seeds: [123, 456, 789]

# Data loading
batch_size: 1024
ot: true                       # Use optimal transport
num_workers: 0
```

### `configs/data/surge.yaml`

Configuration for the Surge XT synthesizer dataset:

```yaml
# @package _global_

_target_: src.data.surge_datamodule.SurgeDataModule

# Dataset parameters
dataset_root: ${paths.data_dir}/surge
use_saved_mean_and_variance: true
batch_size: 1024
ot: true
num_workers: 0
fake: false
repeat_first_batch: false
predict_file: null
conditioning: "mel"  # "mel" or "m2l"
```

## Model Configurations

### `configs/model/surge_flow_matching.yaml`

Configuration for Surge flow matching model:

```yaml
# @package _global_

_target_: src.models.surge_flow_matching_module.SurgeFlowMatchingModule

# Model components
encoder:
  _target_: src.models.components.cnn.ASTWithProjectionHead
  d_model: 768
  d_out: 128
  n_heads: 8
  n_layers: 16
  patch_size: 16
  patch_stride: 10
  input_channels: 2
  spec_shape: [128, 401]

vector_field:
  _target_: src.models.components.transformer.ApproxEquivTransformer
  projection:
    _target_: src.models.components.transformer.LearntProjection
    d_model: 512
    d_token: 128
    num_params: 90
    num_tokens: 32
    initial_ffn: true
    final_ffn: true
  d_model: 512
  conditioning_dim: 128
  num_layers: 6
  num_heads: 8
  d_ff: 1024
  num_tokens: 32
  learn_pe: true
  learn_projection: true
  pe_type: "initial"
  pe_penalty: 0.01
  projection_penalty: 0.01
  time_encoding: "scalar"
  norm: "layer"
  adaln_mode: "basic"
  zero_init: true
  outer_residual: false

# Training parameters
conditioning: "mel"
warmup_steps: 5000
cfg_dropout_rate: 0.1
rectified_sigma_min: 0.0
validation_sample_steps: 50
validation_cfg_strength: 4.0
test_sample_steps: 100
test_cfg_strength: 4.0
compile: false
num_params: 90

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 100000
  eta_min: 0.00001
```

### `configs/model/ksin_flow_matching.yaml`

Configuration for k-sin flow matching model:

```yaml
# @package _global_

_target_: src.models.ksin_flow_matching_module.KSinFlowMatchingModule

# Model components
encoder:
  _target_: src.models.components.residual_mlp.ResidualMLP
  input_dim: 1024
  hidden_dim: 512
  output_dim: 128
  num_layers: 4
  dropout: 0.1

vector_field:
  _target_: src.models.components.transformer.ApproxEquivTransformer
  projection:
    _target_: src.models.components.transformer.KSinParamToTokenProjection
    d_model: 256
    params_per_token: 2
  d_model: 256
  conditioning_dim: 128
  num_layers: 4
  num_heads: 8
  d_ff: 512
  num_tokens: 16
  learn_pe: true
  learn_projection: true
  pe_type: "initial"
  pe_penalty: 0.01
  projection_penalty: 0.01
  time_encoding: "scalar"
  norm: "layer"
  adaln_mode: "basic"
  zero_init: true

# Training parameters
conditioning: "signal"
warmup_steps: 1000
cfg_dropout_rate: 0.1
rectified_sigma_min: 0.0
validation_sample_steps: 20
validation_cfg_strength: 2.0
test_sample_steps: 50
test_cfg_strength: 2.0
compile: false
num_params: 8  # 4 frequencies + 4 amplitudes

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.01

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 50000
  eta_min: 0.0001
```

## Experiment Configurations

### `configs/experiment/surge/baseline.yaml`

Pre-configured experiment for Surge baseline:

```yaml
# @package _global_

# Override defaults
defaults:
  - override /data: surge
  - override /model: surge_flow_matching

# Experiment-specific settings
tags: ["surge", "baseline", "flow_matching"]

# Model overrides
model:
  vector_field:
    d_model: 512
    num_layers: 6
    num_heads: 8
    pe_penalty: 0.01
    projection_penalty: 0.01

# Training overrides
trainer:
  max_epochs: 100
  gradient_clip_val: 1.0

# Data overrides
data:
  batch_size: 512
  ot: true
```

### `configs/experiment/ksin/baseline.yaml`

Pre-configured experiment for k-sin baseline:

```yaml
# @package _global_

# Override defaults
defaults:
  - override /data: ksin
  - override /model: ksin_flow_matching

# Experiment-specific settings
tags: ["ksin", "baseline", "flow_matching"]

# Data overrides
data:
  k: 4
  sort_frequencies: true
  break_symmetry: false
  batch_size: 1024
  ot: true

# Model overrides
model:
  vector_field:
    d_model: 256
    num_layers: 4
    num_heads: 8
    pe_penalty: 0.01
    projection_penalty: 0.01

# Training overrides
trainer:
  max_epochs: 50
  gradient_clip_val: 1.0
```

## Training Configurations

### `configs/trainer/gpu.yaml`

GPU training configuration:

```yaml
# @package _global_

_target_: lightning.Trainer

# Hardware settings
accelerator: "gpu"
devices: 1
precision: "16-mixed"

# Training settings
max_epochs: 100
gradient_clip_val: 1.0
gradient_clip_algorithm: "norm"

# Logging and checkpointing
log_every_n_steps: 50
enable_checkpointing: true
enable_progress_bar: true
enable_model_summary: true

# Validation
val_check_interval: 1.0
check_val_every_n_epoch: 1

# Reproducibility
deterministic: false
```

### `configs/trainer/cpu.yaml`

CPU training configuration:

```yaml
# @package _global_

_target_: lightning.Trainer

# Hardware settings
accelerator: "cpu"
devices: 1
precision: "32"

# Training settings
max_epochs: 100
gradient_clip_val: 1.0
gradient_clip_algorithm: "norm"

# Logging and checkpointing
log_every_n_steps: 50
enable_checkpointing: true
enable_progress_bar: true
enable_model_summary: true

# Validation
val_check_interval: 1.0
check_val_every_n_epoch: 1

# Reproducibility
deterministic: true
```

## Callback Configurations

### `configs/callbacks/default.yaml`

Default callback configuration:

```yaml
# @package _global_

defaults:
  - model_checkpoint
  - early_stopping
  - lr_monitor
  - rich_progress_bar

# Callback settings
model_checkpoint:
  monitor: "val/param_mse"
  mode: "min"
  save_top_k: 3
  save_last: true
  filename: "epoch={epoch:02d}-val_loss={val/param_mse:.2f}"

early_stopping:
  monitor: "val/param_mse"
  mode: "min"
  patience: 10
  min_delta: 0.001
```

### `configs/callbacks/model_checkpoint.yaml`

Model checkpoint callback:

```yaml
# @package _global_

_target_: lightning.pytorch.callbacks.ModelCheckpoint

# Checkpoint settings
monitor: "val/param_mse"
mode: "min"
save_top_k: 3
save_last: true
filename: "epoch={epoch:02d}-val_loss={val/param_mse:.2f}"
auto_insert_metric_name: false
```

## Logger Configurations

### `configs/logger/many_loggers.yaml`

Multiple logger configuration:

```yaml
# @package _global_

defaults:
  - tensorboard
  - wandb
  - csv

# Logger settings
tensorboard:
  save_dir: "${paths.log_dir}/tensorboard"

wandb:
  project: "synth-permutations"
  name: "${task_name}"
  tags: "${tags}"

csv:
  save_dir: "${paths.log_dir}/csv"
```

### `configs/logger/wandb.yaml`

Weights & Biases logger:

```yaml
# @package _global_

_target_: lightning.pytorch.loggers.WandbLogger

# W&B settings
project: "synth-permutations"
name: "${task_name}"
tags: "${tags}"
save_dir: "${paths.log_dir}/wandb"
log_model: true
```

## Advanced Usage

### Command Line Overrides

You can override any configuration parameter from the command line:

```bash
# Override model parameters
python src/train.py model.vector_field.d_model=1024 model.vector_field.num_layers=8

# Override data parameters
python src/train.py data.k=8 data.batch_size=512

# Override training parameters
python src/train.py trainer.max_epochs=200 trainer.gradient_clip_val=0.5

# Override multiple parameters
python src/train.py \
  data.k=8 \
  model.vector_field.d_model=1024 \
  trainer.max_epochs=200 \
  tags="[experiment, large_model]"
```

### Experiment Configurations

Use pre-configured experiments:

```bash
# Use baseline experiment
python src/train.py experiment=ksin/baseline

# Override experiment parameters
python src/train.py experiment=ksin/baseline data.k=8 trainer.max_epochs=100

# Use Surge experiment
python src/train.py experiment=surge/baseline
```

### Hyperparameter Search

Use Optuna for hyperparameter optimization:

```bash
# Run hyperparameter search
python src/train.py hparams_search=ksin_optuna

# Override search parameters
python src/train.py hparams_search=ksin_optuna hparams_search.n_trials=100
```

### Multi-run Experiments

Run multiple experiments with different configurations:

```bash
# Run with different seeds
python src/train.py --multirun seed=1,2,3,4,5

# Run with different model sizes
python src/train.py --multirun model.vector_field.d_model=256,512,1024

# Run with different datasets
python src/train.py --multirun data.k=2,4,8,16
```

### Local Configuration

Create local configuration for machine-specific settings:

```yaml
# configs/local/local.yaml
# @package _global_

# Override paths for local machine
paths:
  data_dir: "/local/path/to/data"
  log_dir: "/local/path/to/logs"
  checkpoint_dir: "/local/path/to/checkpoints"

# Override hardware settings
trainer:
  devices: 2
  accelerator: "gpu"
```

### Debug Configuration

Use debug configuration for development:

```bash
# Enable debug mode
python src/train.py debug=default

# Override debug settings
python src/train.py debug=default debug.limit_train_batches=10 debug.limit_val_batches=5
```

## Best Practices

1. **Use experiment configurations** for reproducible experiments
2. **Override parameters** from command line for quick experiments
3. **Use local configuration** for machine-specific settings
4. **Enable debug mode** during development
5. **Use hyperparameter search** for optimization
6. **Tag experiments** for easy identification
7. **Use multi-run** for systematic experiments

## Configuration Validation

The configuration system includes validation to ensure consistency:

- Required parameters are checked
- Type validation for parameters
- Cross-configuration consistency checks
- Path existence validation

For more information, see the [Hydra documentation](https://hydra.cc/).
