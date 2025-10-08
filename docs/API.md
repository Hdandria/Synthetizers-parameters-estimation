# API Documentation

This document provides comprehensive API documentation for the synth-permutations project.

## Table of Contents

- [Training and Evaluation](#training-and-evaluation)
- [Data Modules](#data-modules)
- [Models](#models)
- [Metrics](#metrics)
- [Utilities](#utilities)

## Training and Evaluation

### `src.train.train`

```python
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Train a model using PyTorch Lightning with Hydra configuration.

**Parameters:**
- `cfg` (DictConfig): Hydra configuration dictionary containing all training parameters

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`: Tuple containing metrics and instantiated objects

**Example:**
```python
from omegaconf import DictConfig
from src.train import train

# Train with configuration
metrics, objects = train(cfg)
model = objects['model']
trainer = objects['trainer']
```

### `src.eval.evaluate`

```python
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Evaluate a trained model checkpoint on test/validation data.

**Parameters:**
- `cfg` (DictConfig): Hydra configuration dictionary containing ckpt_path and other settings

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`: Tuple containing metrics and instantiated objects

**Example:**
```python
from omegaconf import DictConfig
from src.eval import evaluate

# Evaluate model
metrics, objects = evaluate(cfg)
```

## Data Modules

### `src.data.ksin_datamodule.KSinDataModule`

```python
class KSinDataModule(LightningDataModule):
    def __init__(
        self,
        k: int,
        signal_length: int = 1024,
        sort_frequencies: bool = False,
        break_symmetry: bool = False,
        shift_test_distribution: bool = False,
        train_val_test_sizes: Tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: Tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 1024,
        ot: bool = False,
        num_workers: int = 0,
    )
```

PyTorch Lightning data module for the k-sin parameter estimation task.

**Parameters:**
- `k` (int): Number of sinusoidal components in each signal
- `signal_length` (int): Length of generated signals in samples
- `sort_frequencies` (bool): If True, sort frequencies to partially break permutation symmetry
- `break_symmetry` (bool): If True, add frequency shifts to break permutation symmetry
- `shift_test_distribution` (bool): If True, use different frequency distribution for test set
- `train_val_test_sizes` (Tuple[int, int, int]): Dataset sizes for train/val/test splits
- `train_val_test_seeds` (Tuple[int, int, int]): Random seeds for each split
- `batch_size` (int): Batch size for data loaders
- `ot` (bool): If True, use optimal transport for minibatch coupling
- `num_workers` (int): Number of worker processes for data loading

**Example:**
```python
from src.data.ksin_datamodule import KSinDataModule

# Create data module
dm = KSinDataModule(
    k=4,
    signal_length=1024,
    sort_frequencies=True,
    batch_size=64,
    ot=True
)

# Setup and use
dm.setup()
train_loader = dm.train_dataloader()
```

### `src.data.surge_datamodule.SurgeDataModule`

```python
class SurgeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
        fake: bool = False,
        repeat_first_batch: bool = False,
        predict_file: Optional[str] = None,
        conditioning: Literal["mel", "m2l"] = "mel",
    )
```

PyTorch Lightning data module for Surge XT synthesizer dataset.

**Parameters:**
- `dataset_root` (Union[str, Path]): Path to dataset directory containing train.h5, val.h5, test.h5
- `use_saved_mean_and_variance` (bool): If True, use precomputed dataset statistics
- `batch_size` (int): Batch size for data loaders
- `ot` (bool): If True, use optimal transport for minibatch coupling
- `num_workers` (int): Number of worker processes for data loading
- `fake` (bool): If True, generate fake data for testing
- `repeat_first_batch` (bool): If True, repeat first batch for debugging
- `predict_file` (Optional[str]): Path to prediction dataset file
- `conditioning` (Literal["mel", "m2l"]): Type of conditioning features to use

**Example:**
```python
from src.data.surge_datamodule import SurgeDataModule

# Create data module
dm = SurgeDataModule(
    dataset_root="/path/to/surge/dataset",
    batch_size=64,
    ot=True,
    conditioning="mel"
)

# Setup and use
dm.setup()
train_loader = dm.train_dataloader()
```

## Models

### `src.models.surge_flow_matching_module.SurgeFlowMatchingModule`

```python
class SurgeFlowMatchingModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        conditioning: Literal["mel", "m2l"] = "mel",
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        compile: bool = False,
        num_params: int = 90,
    )
```

Flow matching module for Surge XT synthesizer parameter estimation.

**Parameters:**
- `encoder` (torch.nn.Module): Audio encoder (e.g., AST, CNN)
- `vector_field` (torch.nn.Module): Vector field network (e.g., ApproxEquivTransformer)
- `optimizer` (torch.optim.Optimizer): Optimizer configuration
- `scheduler` (torch.optim.lr_scheduler): Learning rate scheduler configuration
- `conditioning` (Literal["mel", "m2l"]): Type of conditioning features
- `warmup_steps` (int): Number of warmup steps for learning rate
- `cfg_dropout_rate` (float): Classifier-free guidance dropout rate
- `rectified_sigma_min` (float): Minimum sigma for rectified flow
- `validation_sample_steps` (int): Number of sampling steps for validation
- `validation_cfg_strength` (float): CFG strength for validation
- `test_sample_steps` (int): Number of sampling steps for testing
- `test_cfg_strength` (float): CFG strength for testing
- `compile` (bool): If True, compile model with torch.compile
- `num_params` (int): Number of synthesizer parameters

**Example:**
```python
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule
from src.models.components.transformer import ApproxEquivTransformer
from src.models.components.cnn import ASTWithProjectionHead

# Create model components
encoder = ASTWithProjectionHead(d_model=512, d_out=128)
vector_field = ApproxEquivTransformer(
    projection=LearntProjection(d_model=512, d_token=128, num_params=90, num_tokens=32),
    d_model=512,
    conditioning_dim=128
)

# Create flow matching module
model = SurgeFlowMatchingModule(
    encoder=encoder,
    vector_field=vector_field,
    optimizer=torch.optim.AdamW,
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
    conditioning="mel",
    num_params=90
)
```

### `src.models.components.transformer.ApproxEquivTransformer`

```python
class ApproxEquivTransformer(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        num_layers: int = 5,
        d_model: int = 1024,
        conditioning_dim: int = 128,
        num_heads: int = 8,
        d_ff: int = 1024,
        num_tokens: int = 32,
        learn_pe: bool = False,
        learn_projection: bool = False,
        pe_type: Literal["initial", "layerwise"] = "initial",
        pe_penalty: float = 0.0,
        time_encoding: Literal["sinusoidal", "scalar"] = "scalar",
        d_enc: int = 256,
        projection_penalty: float = 0.0,
        norm: Literal["layer", "rms"] = "layer",
        skip_first_norm: bool = False,
        adaln_mode: Literal["basic", "zero"] = "basic",
        zero_init: bool = True,
        outer_residual: bool = False,
    )
```

Approximately equivariant transformer for flow matching.

**Parameters:**
- `projection` (nn.Module): Parameter-to-token projection module
- `num_layers` (int): Number of transformer layers
- `d_model` (int): Model dimension
- `conditioning_dim` (int): Conditioning vector dimension
- `num_heads` (int): Number of attention heads
- `d_ff` (int): Feed-forward dimension
- `num_tokens` (int): Number of tokens for positional encoding
- `learn_pe` (bool): If True, learn positional encodings
- `learn_projection` (bool): If True, learn projection parameters
- `pe_type` (Literal["initial", "layerwise"]): Type of positional encoding
- `pe_penalty` (float): Penalty weight for positional encoding regularization
- `time_encoding` (Literal["sinusoidal", "scalar"]): Type of time encoding
- `d_enc` (int): Time encoding dimension
- `projection_penalty` (float): Penalty weight for projection regularization
- `norm` (Literal["layer", "rms"]): Normalization type
- `skip_first_norm` (bool): If True, skip normalization in first layer
- `adaln_mode` (Literal["basic", "zero"]): Adaptive layer norm mode
- `zero_init` (bool): If True, use zero initialization for output layers
- `outer_residual` (bool): If True, add residual connection around entire network

**Example:**
```python
from src.models.components.transformer import ApproxEquivTransformer, LearntProjection

# Create projection
projection = LearntProjection(
    d_model=512,
    d_token=128,
    num_params=90,
    num_tokens=32
)

# Create transformer
transformer = ApproxEquivTransformer(
    projection=projection,
    d_model=512,
    conditioning_dim=128,
    num_layers=6,
    num_heads=8
)
```

## Metrics

### `src.metrics.LogSpectralDistance`

```python
class LogSpectralDistance(Metric):
    def __init__(self, eps: float = 1e-8, **kwargs)
```

Log Spectral Distance (LSD) metric for audio quality assessment.

**Parameters:**
- `eps` (float): Small epsilon value to prevent numerical issues
- `**kwargs`: Additional arguments passed to the base Metric class

**Methods:**
- `update(predicted_params, target_signal, synth_fn)`: Update metric with new batch
- `compute()`: Compute final metric value

**Example:**
```python
from src.metrics import LogSpectralDistance

# Create metric
lsd_metric = LogSpectralDistance()

# Update with predictions
lsd_metric.update(predicted_params, target_audio, synth_function)

# Compute final value
lsd_value = lsd_metric.compute()
```

### `src.metrics.ChamferDistance`

```python
class ChamferDistance(Metric):
    def __init__(self, params_per_token: int, **kwargs)
```

Chamfer Distance metric for permutation-invariant parameter comparison.

**Parameters:**
- `params_per_token` (int): Number of parameters per token/component
- `**kwargs`: Additional arguments passed to the base Metric class

**Methods:**
- `update(predicted, target)`: Update metric with new batch
- `compute()`: Compute final metric value

**Example:**
```python
from src.metrics import ChamferDistance

# Create metric
chamfer_metric = ChamferDistance(params_per_token=2)

# Update with predictions
chamfer_metric.update(predicted_params, target_params)

# Compute final value
chamfer_value = chamfer_metric.compute()
```

### `src.metrics.LinearAssignmentDistance`

```python
class LinearAssignmentDistance(Metric):
    def __init__(self, params_per_token: int, **kwargs)
```

Linear Assignment Distance metric using Hungarian algorithm.

**Parameters:**
- `params_per_token` (int): Number of parameters per token/component
- `**kwargs`: Additional arguments passed to the base Metric class

**Methods:**
- `update(predicted, target)`: Update metric with new batch
- `compute()`: Compute final metric value

**Example:**
```python
from src.metrics import LinearAssignmentDistance

# Create metric
lad_metric = LinearAssignmentDistance(params_per_token=2)

# Update with predictions
lad_metric.update(predicted_params, target_params)

# Compute final value
lad_value = lad_metric.compute()
```

## Utilities

### `src.data.ksin_datamodule.make_sin`

```python
def make_sin(params: torch.Tensor, length: int, break_symmetry: bool = False) -> torch.Tensor
```

Generate a sinusoidal signal from frequency and amplitude parameters.

**Parameters:**
- `params` (torch.Tensor): Parameter tensor of shape (..., 2*k)
- `length` (int): Length of the output signal in samples
- `break_symmetry` (bool): If True, adds frequency shifts to break permutation symmetry

**Returns:**
- `torch.Tensor`: Generated sinusoidal signal

**Example:**
```python
from src.data.ksin_datamodule import make_sin

# Generate 4-sinusoid mixture
params = torch.randn(1, 8)  # 4 freqs + 4 amps
signal = make_sin(params, length=1024)
```

### `src.metrics.complex_to_dbfs`

```python
def complex_to_dbfs(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor
```

Convert complex frequency domain signal to dBFS.

**Parameters:**
- `z` (torch.Tensor): Complex tensor representing frequency domain signal
- `eps` (float): Small epsilon value to prevent log(0)

**Returns:**
- `torch.Tensor`: Power spectral density in dBFS

**Example:**
```python
from src.metrics import complex_to_dbfs

# Convert FFT output to dBFS
fft_output = torch.fft.rfft(audio_signal)
dbfs = complex_to_dbfs(fft_output)
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/train.yaml`: Main training configuration
- `configs/eval.yaml`: Main evaluation configuration
- `configs/data/`: Dataset configurations
- `configs/model/`: Model configurations
- `configs/experiment/`: Pre-configured experiments

**Example:**
```bash
# Train with specific configuration
python src/train.py data=ksin model=ksin_flow_matching data.k=4

# Override multiple parameters
python src/train.py data=ksin model=ksin_flow_matching data.k=4 model.d_model=512 trainer.max_epochs=100

# Use experiment configuration
python src/train.py experiment=ksin/baseline
```
