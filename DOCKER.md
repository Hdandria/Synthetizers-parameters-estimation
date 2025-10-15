# Docker Setup

GPU-optimized container for synthesizer parameter estimation with CUDA 12.1, PyTorch, and audio processing libraries.

## Prerequisites

- Docker Desktop with GPU support
- NVIDIA Container Toolkit

## Build

```bash
# Build image
./docker/build.bat

# Check size
docker images synth-param-estimation:latest
```

## What It Contains

- **Base**: Ubuntu 22.04 + CUDA 12.1 + cuDNN 8
- **Python**: 3.10 with uv package manager
- **ML Stack**: PyTorch, Lightning, audio libraries
- **Code**: Complete project with Hydra configs
- **User**: Non-root `trainer` user (UID 1000)

## Usage

### Automated Training

```bash
# Run experiment
./docker/run_local.bat flow_first_tests/docker_100k_fp16

# Or with custom experiment
./docker/run_local.bat surge/base
```

### Development Environment

```bash
# Interactive shell
./docker/shell.bat

# Inside container
python src/train.py experiment=surge/base paths=docker
```

### Monitoring

```bash
# TensorBoard
docker-compose up tensorboard-service
# Access: http://localhost:6006
```

## Volume Mounts

- `./data:/workspace/data:ro` - Datasets
- `./logs:/workspace/logs` - Training logs  
- `./outputs:/workspace/outputs` - Model outputs
- `./plugins:/workspace/plugins:ro` - VST plugins

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `PROJECT_ROOT` | Container working directory |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `WANDB_API_KEY` | Experiment tracking |

### Paths

Uses `configs/paths/docker.yaml` for container-specific paths.

## Troubleshooting

**GPU not detected:**
```bash
docker run --gpus all synth-param-estimation:latest nvidia-smi
```

**Out of memory:**
```bash
# Reduce batch size
python src/train.py experiment=surge/base data.batch_size=64
```

**Check container:**
```bash
docker-compose run --rm shell-service env | grep CUDA
```