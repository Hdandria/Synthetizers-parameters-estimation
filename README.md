# Synthesizer Parameters Estimation

Deep learning models to estimate synthesizer parameters from audio. Uses flow matching to predict VST plugin parameters (mainly Surge XT) from audio input.

> Forked from [synth-permutations](https://github.com/ben-hayes/synth-permutations) by **Benjamin Hayes**.

## What it does

Given an audio file, the model predicts the synthesizer parameters that could have generated it. Think of it as reverse-engineering sound design.

## Setup

Run the setup script to install dependencies (Docker, ovhai CLI, Terraform):

```bash
./scripts/ovh/setup.sh
```

Configure your credentials in `.env`:

```bash
nano .env
```

For local development, install Python dependencies:

```bash
pip install uv
uv pip install -r pyproject.toml
```

## Usage

**Train on OVH cloud:**

```bash
./launch.sh flow_multi/dataset_20k_40k
```

**Train locally with Docker:**

```bash
./launch.sh flow_multi/dataset_20k_40k --local
```

**Train locally without Docker:**

```bash
python src/train.py experiment=flow_multi/dataset_20k_40k
```

**Evaluate a checkpoint:**

```bash
python src/eval.py experiment=flow_multi/dataset_20k_40k ckpt_path=path/to/checkpoint.ckpt
```

**Monitor jobs:**

```bash
./scripts/ovh/status.sh <job-id>
./scripts/ovh/logs.sh <job-id>
ovhai job stop <job-id>
```

## Data

Datasets are HDF5 files containing audio features and VST parameters. Located in `datasets/` with train/val/test splits.

Generate new data:

```bash
python scripts/dataset/generate_surge_xt_data.py --surge-path vsts/Surge\ XT.vst3
```

## Model

Flow matching model that learns audio → parameters mapping:

- Audio encoder (mel spectrograms or Music2Latent embeddings)
- Vector field network (parameter trajectory prediction)
- Classifier-free guidance during inference

Configs in `configs/experiment/flow_multi/`.

## Project structure

- `src/` - source code. models, data loaders, training/eval
- `configs/` - Hydra experiment configs
- `scripts/` - utilities split into subfolders: `scripts/ovh/`, `scripts/dataset/`, `scripts/eval/`, `scripts/render/` and helpers
- `datasets/` - HDF5 training data
- `launch.sh` - main entry point for training
- `terraform/` - cloud infrastructure

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU)
- Lightning 2.0+
- Docker (for cloud/containerized runs)
