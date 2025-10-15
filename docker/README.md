# Docker Training Environment

This directory contains Docker configuration for running the synthesizer parameter estimation training in a containerized environment with GPU support.

## Prerequisites

### System Requirements
- **Docker**: Version 20.10+ with Docker Compose
- **NVIDIA Container Toolkit**: For GPU support
- **CUDA 12.1+ drivers**: Compatible with your GPU
- **At least 8GB RAM**: Recommended for training
- **20GB+ free disk space**: For datasets and checkpoints

### Install NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows:**
- Install Docker Desktop with WSL2 backend
- Enable GPU support in Docker Desktop settings
- Install NVIDIA drivers for Windows

**macOS:**
- Docker Desktop with Apple Silicon support
- Note: No GPU acceleration on macOS

## Quick Start

### 1. Prepare VST Plugins

Create a `vsts/` directory and place your VST3 plugins:
```bash
mkdir -p vsts
# Download Surge XT and place it in vsts/
# vsts/Surge XT.vst3
```

### 2. Build the Docker Image

```bash
docker-compose build
```

### 3. Run Training

**Using helper script:**
```bash
# Basic training
./docker/run_training.sh

# With experiment config
./docker/run_training.sh surge/baseline

# With additional parameters
./docker/run_training.sh surge/baseline data.batch_size=64 trainer.max_epochs=100
```

**Using docker-compose directly:**
```bash
# Interactive shell
docker-compose run --rm train

# Run specific command
docker-compose run --rm train python src/train.py experiment=surge/baseline
```

## Volume Mounts

The Docker setup uses the following volume mounts:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/workspace/data` | HDF5 datasets |
| `./logs` | `/workspace/logs` | Training logs and checkpoints |
| `./vsts` | `/workspace/vsts` | VST3 plugins |
| `./presets` | `/workspace/presets` | VST presets |
| `./checkpoints` | `/workspace/checkpoints` | Model checkpoints |

## GPU Configuration

### Single GPU
```bash
export CUDA_VISIBLE_DEVICES=0
docker-compose run --rm train python src/train.py
```

### Multiple GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
docker-compose run --rm train python src/train.py trainer.devices=[0,1,2,3] trainer.strategy=ddp
```

### Check GPU Status
```bash
docker-compose run --rm train nvidia-smi
```

## Services

### Training Service
Main service for running training and evaluation:
```bash
docker-compose run --rm train [command]
```

### Interactive Shell
Quick access to container shell:
```bash
./docker/shell.sh
# or
docker-compose run --rm shell
```

### Jupyter Lab (Optional)
Start Jupyter Lab for interactive development:
```bash
docker-compose up jupyter
# Access at http://localhost:8888
```

## Common Workflows

### 1. Generate Dataset
```bash
# Interactive shell
./docker/shell.sh

# Inside container
python src/data/vst/generate_vst_dataset.py \
    /workspace/data/train.h5 \
    10000 \
    --plugin_path "/workspace/vsts/Surge XT.vst3" \
    --preset_path "/workspace/presets/surge-simple.vstpreset" \
    --sample_rate 44100 \
    --channels 2 \
    --signal_duration_seconds 4.0 \
    --min_loudness -55.0 \
    --param_spec surge_simple \
    --sample_batch_size 64 \
    --num_workers 4
```

### 2. Compute Dataset Statistics
```bash
docker-compose run --rm train python scripts/get_dataset_stats.py /workspace/data/train.h5
```

### 3. Train Model
```bash
./docker/run_training.sh surge/baseline
```

### 4. Evaluate Model
```bash
./docker/run_eval.sh logs/train/runs/2024-01-01_12-00-00/checkpoints/best.ckpt test
```

### 5. Generate Predictions
```bash
./docker/run_eval.sh logs/train/runs/2024-01-01_12-00-00/checkpoints/best.ckpt predict
```

## Environment Variables

Create a `.env` file in the project root to customize behavior:

```bash
# GPU configuration
CUDA_VISIBLE_DEVICES=0

# Optional: Auto-download VST during build
DOWNLOAD_VST=false

# Hydra configuration
HYDRA_FULL_ERROR=1
```

## Performance Optimization

### 1. Multi-GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./docker/run_training.sh surge/baseline trainer.devices=[0,1,2,3] trainer.strategy=ddp
```

### 2. Mixed Precision
```bash
./docker/run_training.sh surge/baseline trainer.precision=16
```

### 3. Gradient Accumulation
```bash
./docker/run_training.sh surge/baseline trainer.accumulate_grad_batches=4
```

### 4. Data Loading Optimization
```bash
./docker/run_training.sh surge/baseline data.num_workers=8 data.persistent_workers=true
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support
docker-compose run --rm train nvidia-smi
```

### Permission Issues
```bash
# Fix ownership of mounted volumes
sudo chown -R 1000:1000 data/ logs/ vsts/ checkpoints/
```

### CUDA Out of Memory
```bash
# Reduce batch size
./docker/run_training.sh surge/baseline data.batch_size=32

# Use gradient accumulation
./docker/run_training.sh surge/baseline trainer.accumulate_grad_batches=2

# Enable mixed precision
./docker/run_training.sh surge/baseline trainer.precision=16
```

### VST Plugin Not Found
```bash
# Check VST path
ls -la /workspace/vsts/

# Verify plugin in container
docker-compose run --rm train ls -la /workspace/vsts/
```

### Slow Data Loading
```bash
# Increase number of workers
./docker/run_training.sh surge/baseline data.num_workers=8

# Use persistent workers
./docker/run_training.sh surge/baseline data.persistent_workers=true
```

### Container Build Issues
```bash
# Clean build
docker-compose build --no-cache

# Check build logs
docker-compose build --progress=plain
```

## Monitoring

### TensorBoard
```bash
# Start TensorBoard
docker-compose run --rm train tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006

# Access at http://localhost:6006
```

### Resource Usage
```bash
# Check container stats
docker stats

# Check GPU usage
docker-compose run --rm train nvidia-smi -l 1
```

## Development

### Adding Dependencies
1. Update `requirements.txt` or `pyproject.toml`
2. Rebuild image: `docker-compose build`
3. Test in container: `./docker/shell.sh`

### Debugging
```bash
# Interactive debugging
./docker/shell.sh

# Run with debug flags
./docker/run_training.sh surge/baseline trainer.fast_dev_run=true

# Check logs
docker-compose logs train
```

## File Structure

```
.
├── Dockerfile                 # Main Docker image definition
├── docker-compose.yml        # Service orchestration
├── .dockerignore            # Build context exclusions
├── docker/
│   ├── README.md            # This file
│   ├── run_training.sh      # Training wrapper script
│   ├── run_eval.sh          # Evaluation wrapper script
│   └── shell.sh             # Interactive shell script
└── configs/paths/docker.yaml # Docker-specific paths
```

## Best Practices

1. **Use volume mounts** for large datasets and checkpoints
2. **Set CUDA_VISIBLE_DEVICES** to control GPU usage
3. **Use experiment configs** for reproducible training
4. **Monitor resource usage** during training
5. **Keep container images updated** with latest dependencies
6. **Use helper scripts** for common operations
7. **Clean up unused containers** regularly: `docker system prune`

