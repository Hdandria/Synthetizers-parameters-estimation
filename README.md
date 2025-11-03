# Synthesizer Parameters Estimation

Estimate synthesizer parameters from audio using conditional flow matching. Given an audio file, predict the VST parameters that could have generated it.

> Forked from [synth-permutations](https://github.com/ben-hayes/synth-permutations) by Benjamin Hayes

## Quick Start

### Automated Setup

```bash
./scripts/ovh/setup.sh
```

This installs all dependencies, creates your `.env` file, and configures OVH S3 datastore.

### Manual Setup

```bash
# 1. Install dependencies
sudo apt-get install -y docker.io jq python3-pip  # Ubuntu/Debian
pip install uv
uv pip install --system -r pyproject.toml

# 2. Install OVH CLI
curl -fsSL https://cli.bhs.ai.cloud.ovh.net/install.sh | bash
export PATH="$HOME/.ovhai/bin:$PATH"

# 3. Configure environment
cp .env.example .env
nano .env  # Add your credentials
ovhai login

# 4. Configure S3 datastore
source .env
ovhai datastore add s3 s3-GRA \
  "https://s3.gra.io.cloud.ovh.net" \
  "gra" \
  "${AWS_ACCESS_KEY_ID}" \
  "${AWS_SECRET_ACCESS_KEY}" \
  --store-credentials-locally
```

## Configuration

Create `.env` with your credentials:

```bash
# Weights & Biases
WANDB_API_KEY=your_wandb_api_key

# OVH S3 Storage
AWS_ACCESS_KEY_ID=your_s3_access_key
AWS_SECRET_ACCESS_KEY=your_s3_secret_key
AWS_ENDPOINT_URL=https://s3.gra.io.cloud.ovh.net
AWS_DEFAULT_REGION=gra

# S3 Buckets
S3_BUCKET=uniform-100k
S3_BUCKET_OUTPUTS=synth-outputs

# Docker Hub
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_access_token

# Training
FLAVOR=ai1-1-gpu
NUM_GPUS=1
GPU_IDS=0
DATA_NUM_WORKERS=11
```

Get credentials:
- Weights & Biases: https://wandb.ai/authorize
- OVH S3: OVH Manager → Public Cloud → Object Storage → Users
- Docker Hub: https://hub.docker.com/settings/security

## Dataset Preparation

### Download Datasets

```bash
# List available datasets
ovhai bucket object list uniform-100k@s3-GRA --prefix datasets/

# Download locally (if needed)
ovhai bucket object download uniform-100k@s3-GRA \
  datasets/surge-20k/train.h5 \
  datasets/surge-20k/val.h5 \
  datasets/surge-20k/test.h5 \
  --output datasets/surge-20k/
```

### Create Dataset Subsets

```bash
# 20k samples
python scripts/dataset/create_subset_dataset.py \
  ./datasets/surge-100k \
  ./datasets/surge-20k \
  --train-shards 0,1 \
  --val-shards 10 \
  --test-shards 11

# 50k samples
python scripts/dataset/create_subset_dataset.py \
  ./datasets/surge-100k \
  ./datasets/surge-50k \
  --train-shards 0,1,2,3,4 \
  --val-shards 10 \
  --test-shards 11
```

### Generate New Data

```bash
python scripts/dataset/generate_surge_xt_data.py \
  --surge-path "vsts/Surge XT.vst3" \
  --output-dir datasets/surge-custom \
  --num-samples 10000
```

### Upload to S3

```bash
ovhai bucket object upload uniform-100k@s3-GRA \
  datasets/surge-20k/train.h5 \
  datasets/surge-20k/val.h5 \
  datasets/surge-20k/test.h5 \
  datasets/surge-20k/stats.npz
```

## Training

### Cloud Training (OVH)

```bash
./launch.sh flow_multi/dataset_20k_40k
```

Builds Docker image, submits job to OVH, and streams logs.

### Local Training

```bash
# With Docker
./launch.sh flow_multi/dataset_20k_40k --local

# Direct (development)
python src/train.py experiment=flow_multi/dataset_20k_40k
```

### Available Experiments

Located in `configs/experiment/flow_multi/`:

```bash
# Dataset sizes
dataset_20k_40k         # 20k samples (good for testing)
dataset_50k             # 50k samples
dataset_100k            # 100k samples

# Hyperparameter variations
batch_size_low, batch_size_high
lr_low, lr_high

# Model sizes
model_enlarged
model_reduced_50, model_reduced_75

# Token counts
tokens_64, tokens_256

# Full config
important               # All hyperparameters documented
```

### Monitoring

```bash
./scripts/ovh/status.sh <job-id>        # Job status
./scripts/ovh/logs.sh <job-id>          # Stream logs
./scripts/ovh/list-jobs.sh              # List all jobs
ovhai job stop <job-id>                 # Stop job
```

### Evaluation

```bash
# Evaluate checkpoint
python src/eval.py \
  experiment=flow_multi/dataset_20k_40k \
  ckpt_path=path/to/checkpoint.ckpt

# With audio metrics
./scripts/eval/evaluate_locally.sh <checkpoint> <dataset>
```


## Dataset Structure

HDF5 files with Virtual Dataset (VDS) architecture:

```
datasets/
├── surge-100k/           # Full dataset (shards)
│   ├── shard-0.h5       # 10k samples per shard
│   ├── ...
│   ├── shard-11.h5
│   ├── train.h5         # VDS → shards 0-9
│   ├── val.h5           # VDS → shard 10
│   └── test.h5          # VDS → shard 11
├── surge-20k/           # Subset (20k samples)
│   ├── train.h5         # VDS → ../surge-100k/shard-{0,1}.h5
│   ├── val.h5
│   └── test.h5
└── surge-50k/           # Subset (50k samples)
    └── ...
```

### Utilities

```bash
# Check dataset readability
python scripts/dataset/test_readability.py datasets/surge-20k

# Get statistics
python scripts/dataset/get_dataset_stats.py datasets/surge-20k
```

## Model Architecture

Uses conditional flow matching for parameter estimation:

**Components:**
- Audio Encoder: Converts mel spectrograms to conditioning (AST, CNN)
- Vector Field Network: Transformer-based (`ApproxEquivTransformer`)
- Flow Matching: Learns optimal transport trajectories with classifier-free guidance

**Training:**
1. Sample time steps t ∈ [0, 1]
2. Interpolate between noise and target parameters
3. Predict velocity field
4. Minimize MSE loss

**Key Config** (`configs/model/surge_flow.yaml`):
- d_model: 512
- num_layers: 8
- num_tokens: 128
- learning_rate: 1e-4
- cfg_dropout_rate: 0.1

## Project Structure

```
configs/
├── experiment/flow_multi/  # All experiments (only these are used)
├── model/surge_flow.yaml   # Main model config
├── model/encoder/          # Audio encoder configs
├── data/surge.yaml         # Dataset config
├── callbacks/              # Training callbacks
└── trainer/                # PyTorch Lightning configs

src/
├── models/
│   ├── surge_flow_matching_module.py
│   └── components/transformer.py
├── data/
│   ├── surge_datamodule.py
│   └── vst/                # VST rendering
├── train.py                # Training entry point
└── eval.py                 # Evaluation entry point

scripts/
├── dataset/                # Dataset tools
├── eval/                   # Evaluation scripts
├── ovh/                    # Cloud utilities
└── render/                 # Audio rendering

datasets/                   # HDF5 training data
tests/                      # Unit tests
launch.sh                   # Main launcher
Dockerfile                  # Container definition
pyproject.toml             # Dependencies
```


## Troubleshooting

### Setup

**"ovhai: command not found"**
```bash
# Add to PATH
export PATH="$HOME/.ovhai/bin:$PATH"
echo 'export PATH="$HOME/.ovhai/bin:$PATH"' >> ~/.bashrc
```

**"Permission denied" with Docker**
```bash
sudo usermod -aG docker $USER
newgrp docker  # Or log out/in
```

### Datasets

**"No readable non-zero audio found"**
- VDS files must use forward slashes (not Windows backslashes)
- Check `HDF5_VDS_PREFIX` is set correctly
- Run `python scripts/dataset/test_readability.py <path>` to diagnose
- Regenerate VDS if needed

**"Permission denied" on S3**
- Verify credentials in `.env`
- Check buckets exist: `ovhai bucket list s3-GRA`
- Reconfigure datastore (see setup section)

**"Bucket not found"**
```bash
ovhai bucket create s3-GRA <bucket-name>
```

### Training

**"DATASYNC_FAILED"**
- Check S3 bucket names in `.env` and `launch.sh`
- Verify output bucket exists with write permissions
- Check `ovhai volume list`

**Out of memory**
- Use `batch_size_low` experiment
- Try `model_reduced_50` or `model_reduced_75`
- Reduce `data.num_workers`

### Docker

**Build fails**
```bash
sudo systemctl start docker     # Start daemon
docker system prune -a          # Free space
```


## Quick Reference

### Common Commands

```bash
# Setup
./scripts/ovh/setup.sh
ovhai login
ovhai me

# Datasets
ovhai bucket list s3-GRA
ovhai bucket object list <bucket>@s3-GRA
python scripts/dataset/test_readability.py <path>

# Training
./launch.sh flow_multi/<experiment>           # Cloud
./launch.sh flow_multi/<experiment> --local   # Local Docker
python src/train.py experiment=flow_multi/<experiment>

# Monitoring
./scripts/ovh/list-jobs.sh
./scripts/ovh/status.sh <job-id>
./scripts/ovh/logs.sh <job-id>
ovhai job stop <job-id>

# Evaluation
python src/eval.py experiment=flow_multi/<experiment> ckpt_path=<path>
./scripts/eval/evaluate_locally.sh <checkpoint> <dataset>
```

### Workflows

**Typical Training Cycle:**

1. Test locally first:
   ```bash
   ./launch.sh flow_multi/dataset_20k_40k --local
   ```

2. Launch cloud training:
   ```bash
   ./launch.sh flow_multi/dataset_100k
   ```

3. Monitor:
   ```bash
   ./scripts/ovh/logs.sh <job-id>
   ```

4. Evaluate:
   ```bash
   ovhai bucket object download synth-outputs@s3-GRA <path>/checkpoints/best.ckpt
   python src/eval.py experiment=flow_multi/dataset_100k ckpt_path=best.ckpt
   ```

**Tips:**
- Start with `dataset_20k_40k` before scaling to `dataset_100k`
- Monitor OVH costs (V100s = €1.93/hour)
- Use W&B for experiment tracking
- Stop unused jobs to avoid charges

## License

MIT License (inherited from synth-permutations)

## Citation

If you use this code, please cite the original work:

```bibtex
@inproceedings{hayes2024synth,
  title={Synth Permutations: Learning Deep Synthesis from Audio},
  author={Hayes, Benjamin},
  year={2024}
}
```

## Contributing

This is a research project. Feel free to fork and adapt for your own experiments.

For issues or questions, please open a GitHub issue.
