# Usage Guide

This guide provides comprehensive examples and tutorials for using the synth-permutations project. It covers everything from basic usage to advanced techniques.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Training](#basic-training)
- [Evaluation](#evaluation)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ben-hayes/synth-permutations.git
   cd synth-permutations
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"
   ```

### First Training Run

Train a simple model on the k-sin dataset:

```bash
# Basic training
python src/train.py data=ksin model=ksin_flow_matching

# With specific parameters
python src/train.py data=ksin model=ksin_flow_matching data.k=4 trainer.max_epochs=10
```

## Basic Training

### k-Sin Dataset Training

The k-sin dataset is a synthetic dataset designed to test permutation-invariant parameter estimation:

```bash
# Train with default settings
python src/train.py data=ksin model=ksin_flow_matching

# Train with different number of sinusoids
python src/train.py data=ksin model=ksin_flow_matching data.k=8

# Train with symmetry breaking
python src/train.py data=ksin model=ksin_flow_matching data.break_symmetry=true

# Train with sorted frequencies
python src/train.py data=ksin model=ksin_flow_matching data.sort_frequencies=true
```

### Surge XT Dataset Training

Train on the real-world Surge XT synthesizer dataset:

```bash
# Train with default settings
python src/train.py data=surge model=surge_flow_matching

# Train with different conditioning
python src/train.py data=surge model=surge_flow_matching data.conditioning=m2l

# Train with smaller batch size
python src/train.py data=surge model=surge_flow_matching data.batch_size=256
```

### Custom Training Parameters

Override training parameters:

```bash
# Change learning rate
python src/train.py model.optimizer.lr=0.001

# Change model architecture
python src/train.py model.vector_field.d_model=1024 model.vector_field.num_layers=8

# Change training duration
python src/train.py trainer.max_epochs=200

# Enable gradient clipping
python src/train.py trainer.gradient_clip_val=1.0
```

## Evaluation

### Basic Evaluation

Evaluate a trained model:

```bash
# Evaluate on test set
python src/eval.py ckpt_path=path/to/checkpoint.ckpt

# Evaluate on validation set
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=val

# Generate predictions
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=predict
```

### Evaluation with Different Settings

```bash
# Evaluate with different sampling steps
python src/eval.py ckpt_path=path/to/checkpoint.ckpt model.test_sample_steps=200

# Evaluate with different CFG strength
python src/eval.py ckpt_path=path/to/checkpoint.ckpt model.test_cfg_strength=8.0

# Evaluate on different dataset
python src/eval.py ckpt_path=path/to/checkpoint.ckpt data=ksin data.k=8
```

## Data Preparation

### k-Sin Dataset

The k-sin dataset is generated on-the-fly, so no preparation is needed. However, you can customize the generation:

```python
from src.data.ksin_datamodule import KSinDataModule

# Create custom dataset
dm = KSinDataModule(
    k=4,                    # Number of sinusoids
    signal_length=1024,     # Signal length
    sort_frequencies=True,  # Sort frequencies
    break_symmetry=False,   # Don't break symmetry
    batch_size=64,          # Batch size
    ot=True                 # Use optimal transport
)

# Setup and use
dm.setup()
train_loader = dm.train_dataloader()
```

### Surge XT Dataset

The Surge XT dataset requires pre-generated HDF5 files:

1. **Generate dataset using VST plugin:**
   ```bash
   python scripts/generate_surge_xt_data.py \
     --output_dir /path/to/dataset \
     --num_samples 100000 \
     --param_spec surge_xt
   ```

2. **Compute dataset statistics:**
   ```bash
   python scripts/get_dataset_stats.py \
     --dataset_dir /path/to/dataset \
     --output_file /path/to/dataset/stats.npz
   ```

3. **Use in training:**
   ```bash
   python src/train.py data=surge data.dataset_root=/path/to/dataset
   ```

### Custom Dataset

Create a custom dataset by extending the base classes:

```python
from src.data.ksin_datamodule import KSinDataModule

class CustomKSinDataModule(KSinDataModule):
    def __init__(self, custom_param, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def setup(self, stage=None):
        # Custom setup logic
        super().setup(stage)
```

## Model Architectures

### Flow Matching Models

The project implements flow matching for parameter estimation:

```python
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule
from src.models.components.transformer import ApproxEquivTransformer, LearntProjection
from src.models.components.cnn import ASTWithProjectionHead

# Create encoder
encoder = ASTWithProjectionHead(
    d_model=768,
    d_out=128,
    n_heads=8,
    n_layers=16
)

# Create vector field
projection = LearntProjection(
    d_model=512,
    d_token=128,
    num_params=90,
    num_tokens=32
)

vector_field = ApproxEquivTransformer(
    projection=projection,
    d_model=512,
    conditioning_dim=128,
    num_layers=6,
    num_heads=8
)

# Create flow matching model
model = SurgeFlowMatchingModule(
    encoder=encoder,
    vector_field=vector_field,
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    conditioning="mel",
    num_params=90
)
```

### Custom Model Components

Create custom model components:

```python
from src.models.components.transformer import ApproxEquivTransformer

class CustomTransformer(ApproxEquivTransformer):
    def __init__(self, custom_param, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def forward(self, x, t, conditioning=None):
        # Custom forward logic
        return super().forward(x, t, conditioning)
```

## Advanced Techniques

### Hyperparameter Optimization

Use Optuna for hyperparameter optimization:

```bash
# Run hyperparameter search
python src/train.py hparams_search=ksin_optuna

# Custom search parameters
python src/train.py hparams_search=ksin_optuna hparams_search.n_trials=100
```

### Multi-run Experiments

Run multiple experiments with different configurations:

```bash
# Different seeds
python src/train.py --multirun seed=1,2,3,4,5

# Different model sizes
python src/train.py --multirun model.vector_field.d_model=256,512,1024

# Different datasets
python src/train.py --multirun data.k=2,4,8,16
```

### Distributed Training

Train on multiple GPUs:

```bash
# Multi-GPU training
python src/train.py trainer.accelerator=gpu trainer.devices=4

# Distributed training
python src/train.py trainer.accelerator=gpu trainer.devices=4 trainer.strategy=ddp
```

### Mixed Precision Training

Use mixed precision for faster training:

```bash
# Enable mixed precision
python src/train.py trainer.precision=16-mixed

# Full precision
python src/train.py trainer.precision=32
```

### Gradient Accumulation

Use gradient accumulation for larger effective batch sizes:

```bash
# Accumulate gradients
python src/train.py trainer.accumulate_grad_batches=4
```

## Experiment Management

### Using Experiment Configurations

Use pre-configured experiments:

```bash
# Use baseline experiment
python src/train.py experiment=ksin/baseline

# Override experiment parameters
python src/train.py experiment=ksin/baseline data.k=8 trainer.max_epochs=100
```

### Logging and Monitoring

Use different loggers:

```bash
# Use TensorBoard
python src/train.py logger=tensorboard

# Use Weights & Biases
python src/train.py logger=wandb

# Use multiple loggers
python src/train.py logger=many_loggers
```

### Checkpointing

Configure checkpointing:

```bash
# Custom checkpoint settings
python src/train.py callbacks.model_checkpoint.monitor=val/param_mse \
  callbacks.model_checkpoint.save_top_k=5 \
  callbacks.model_checkpoint.save_last=true
```

## Debugging and Development

### Debug Mode

Use debug configuration for development:

```bash
# Enable debug mode
python src/train.py debug=default

# Limit batches for quick testing
python src/train.py debug=default debug.limit_train_batches=10 debug.limit_val_batches=5
```

### Profiling

Profile training performance:

```bash
# Enable profiling
python src/train.py debug=profiler

# Profile specific functions
python src/train.py debug=profiler debug.profiler.profile_functions=["training_step"]
```

### Overfitting Test

Test if model can overfit on small dataset:

```bash
# Overfit on small dataset
python src/train.py debug=overfit data.train_val_test_sizes=[100, 10, 10] trainer.max_epochs=1000
```

## Custom Metrics

### Adding Custom Metrics

Create custom metrics:

```python
from torchmetrics import Metric
import torch

class CustomMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, pred, target):
        # Custom metric computation
        self.metric += (pred - target).abs().mean()
        self.count += 1
    
    def compute(self):
        return self.metric / self.count
```

### Using Custom Metrics

```python
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule

class CustomSurgeFlowMatchingModule(SurgeFlowMatchingModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_metric = CustomMetric()
    
    def validation_step(self, batch, batch_idx):
        # Custom validation logic
        result = super().validation_step(batch, batch_idx)
        self.custom_metric.update(pred, target)
        self.log("val/custom_metric", self.custom_metric)
        return result
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   ```bash
   # Reduce batch size
   python src/train.py data.batch_size=256
   
   # Use gradient accumulation
   python src/train.py trainer.accumulate_grad_batches=4
   
   # Use mixed precision
   python src/train.py trainer.precision=16-mixed
   ```

2. **Slow training:**
   ```bash
   # Increase batch size
   python src/train.py data.batch_size=2048
   
   # Use multiple workers
   python src/train.py data.num_workers=4
   
   # Use mixed precision
   python src/train.py trainer.precision=16-mixed
   ```

3. **Poor convergence:**
   ```bash
   # Adjust learning rate
   python src/train.py model.optimizer.lr=0.0001
   
   # Add gradient clipping
   python src/train.py trainer.gradient_clip_val=1.0
   
   # Use different optimizer
   python src/train.py model.optimizer._target_=torch.optim.Adam
   ```

### Debugging Tips

1. **Check data loading:**
   ```python
   from src.data.ksin_datamodule import KSinDataModule
   
   dm = KSinDataModule(k=4, batch_size=32)
   dm.setup()
   train_loader = dm.train_dataloader()
   
   # Check first batch
   batch = next(iter(train_loader))
   print(f"Batch shape: {batch[0].shape}")
   print(f"Params shape: {batch[1].shape}")
   ```

2. **Check model output:**
   ```python
   import torch
   from src.models.ksin_flow_matching_module import KSinFlowMatchingModule
   
   # Create model
   model = KSinFlowMatchingModule(...)
   
   # Test forward pass
   x = torch.randn(32, 8)  # 4 freqs + 4 amps
   t = torch.rand(32, 1)
   conditioning = torch.randn(32, 128)
   
   output = model.vector_field(x, t, conditioning)
   print(f"Output shape: {output.shape}")
   ```

3. **Check gradients:**
   ```bash
   # Enable gradient logging
   python src/train.py trainer.log_gradients=true
   ```

### Performance Optimization

1. **Data loading optimization:**
   ```bash
   # Use multiple workers
   python src/train.py data.num_workers=4
   
   # Use pin memory
   python src/train.py data.pin_memory=true
   ```

2. **Model optimization:**
   ```bash
   # Compile model
   python src/train.py model.compile=true
   
   # Use mixed precision
   python src/train.py trainer.precision=16-mixed
   ```

3. **Training optimization:**
   ```bash
   # Use gradient accumulation
   python src/train.py trainer.accumulate_grad_batches=4
   
   # Use gradient clipping
   python src/train.py trainer.gradient_clip_val=1.0
   ```

## Best Practices

1. **Start with simple experiments** using the k-sin dataset
2. **Use experiment configurations** for reproducible results
3. **Monitor training** with appropriate loggers
4. **Use validation** to prevent overfitting
5. **Save checkpoints** regularly
6. **Use hyperparameter search** for optimization
7. **Test on different datasets** to ensure generalization
8. **Use debug mode** during development
9. **Profile performance** to identify bottlenecks
10. **Document experiments** with proper tags and descriptions

## Getting Help

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Look at the [API documentation](API.md)
3. Check the [configuration guide](CONFIGURATION.md)
4. Review the [README](README.md)
5. Open an issue on GitHub with detailed information about your problem

For more advanced usage, see the [API documentation](API.md) and [configuration guide](CONFIGURATION.md).
