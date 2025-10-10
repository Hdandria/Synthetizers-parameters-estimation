# Usage Guide

## Training Models

### Basic Training

Train with default configuration:

```bash
python src/train.py
```

### Training with Specific Dataset and Model

Train a feed-forward network on Surge dataset:

```bash
python src/train.py data=surge model=surge_ffn
```

Train a flow matching model on KSIN dataset:

```bash
python src/train.py data=ksin model=ksin_flow_matching
```

### Using Experiment Configurations

Run a pre-configured experiment:

```bash
python src/train.py experiment=surge/baseline
```

Available experiments are in `configs/experiment/`.

### Overriding Parameters

Override specific parameters via command line:

```bash
python src/train.py \
    data=surge \
    model=surge_ffn \
    data.batch_size=128 \
    trainer.max_epochs=100 \
    trainer.devices=[0,1]
```

### GPU Configuration

Train on specific GPU:

```bash
python src/train.py trainer.devices=[0] trainer.accelerator=gpu
```

Train on multiple GPUs:

```bash
python src/train.py trainer.devices=[0,1,2,3] trainer.accelerator=gpu trainer.strategy=ddp
```

### Resuming Training

Resume from checkpoint:

```bash
python src/train.py ckpt_path=path/to/checkpoint.ckpt
```

## Evaluation

### Test Mode

Evaluate on test set:

```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=test
```

### Validation Mode

Evaluate on validation set:

```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=val
```

### Prediction Mode

Generate predictions:

```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=predict
```

## Dataset Generation

### Surge XT Dataset

Generate training set:

```bash
python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/train.h5 \
    16000 \
    --plugin_path "vsts/Surge XT.vst3" \
    --preset_path "presets/surge-simple.vstpreset" \
    --sample_rate 44100 \
    --channels 2 \
    --signal_duration_seconds 4.0 \
    --min_loudness -55.0 \
    --param_spec surge_simple \
    --sample_batch_size 64 \
    --num_workers 20
```

Generate validation set (change count to 2000) and test set similarly.

### Compute Dataset Statistics

Required for normalization:

```bash
python scripts/get_dataset_stats.py datasets/experiment_1/train.h5
python scripts/get_dataset_stats.py datasets/experiment_1/val.h5
python scripts/get_dataset_stats.py datasets/experiment_1/test.h5
```

## Using Loggers

### Weights & Biases

```bash
python src/train.py logger=wandb
```

Configure W&B settings in `configs/logger/wandb.yaml`.

### TensorBoard

```bash
python src/train.py logger=tensorboard
```

View logs:

```bash
tensorboard --logdir logs/
```

### Multiple Loggers

```bash
python src/train.py logger=many_loggers
```

## Debugging

### Fast Development Run

Test training loop with limited batches:

```bash
python src/train.py debug=limit
```

### Overfit on Small Data

Test model capacity:

```bash
python src/train.py debug=overfit
```

### Profile Performance

```bash
python src/train.py debug=profiler
```

## Hyperparameter Optimization

Using Optuna:

```bash
python src/train.py -m hparams_search=ksin_optuna
```

## Scripts

Useful utility scripts in `scripts/`:

- `compute_audio_metrics.py` - Calculate audio quality metrics
- `predict_vst_audio.py` - Generate audio from predictions
- `plot_param2tok.py` - Visualize parameter embeddings
- `get_dataset_stats.py` - Compute dataset statistics

## Common Workflows

### Full Training Pipeline

1. Generate dataset:
```bash
python src/data/vst/generate_vst_dataset.py [args...]
```

2. Compute statistics:
```bash
python scripts/get_dataset_stats.py datasets/path/train.h5
```

3. Train model:
```bash
python src/train.py data=surge model=surge_ffn
```

4. Evaluate:
```bash
python src/eval.py ckpt_path=path/to/best.ckpt
```

### Parameter Inference from Audio

1. Train or load model
2. Run prediction mode:
```bash
python src/eval.py ckpt_path=checkpoint.ckpt mode=predict
```

3. Use predictions:
```bash
python scripts/predict_vst_audio.py [args...]
```

## Output Locations

- **Checkpoints** - `logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/`
- **Logs** - `logs/train/runs/YYYY-MM-DD_HH-MM-SS/`
- **Predictions** - Configured per experiment, typically in `predictions/`

## Tips

- Use `experiment` configs for reproducible experiments
- Start with smaller models and datasets for testing
- Monitor GPU memory usage with `nvidia-smi`
- Use `trainer.fast_dev_run=true` to test code without full training
- Set seed for reproducibility: `seed=42`

