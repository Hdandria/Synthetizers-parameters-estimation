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

## Dataset Preparation

### Generate Shards

Generate 12 shards of 10k samples each (120k total):

```bash
mkdir -p datasets/surge-100k
for i in {0..11}; do
  python -m src.data.vst.generate_vst_dataset \
    datasets/surge-100k/shard-$i.h5 \
    10000 \
    -p "vsts/Surge XT.vst3" \
    -r "presets/surge-base.vstpreset" &
done
wait
```

### Create Virtual Dataset Files

```bash
python scripts/dataset/create_subset_dataset.py \
  ./datasets/surge-100k \
  ./datasets/surge-100k \
  --train-shards 0,1,2,3,4,5,6,7,8,9 \
  --val-shards 10 \
  --test-shards 11
```

### Compute Statistics

```bash
python scripts/dataset/get_dataset_stats.py datasets/surge-100k
```

### Create Subsets

```bash
python scripts/dataset/create_subset_dataset.py \
  ./datasets/surge-100k \
  ./datasets/surge-20k \
  --train-shards 0,1 \
  --val-shards 10 \
  --test-shards 11
```

### Upload to S3

```bash
# Upload full dataset
ovhai bucket object upload uniform-100k@s3-GRA \
  datasets/surge-100k/

# Upload subsets
ovhai bucket object upload uniform-100k@s3-GRA \
  datasets/surge-20k/
```

## Training

```bash
./launch.sh flow_multi/dataset_20k_40k
```

Builds Docker image, submits job to OVH, and streams logs.


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

HDF5 files with Virtual Dataset architecture:

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