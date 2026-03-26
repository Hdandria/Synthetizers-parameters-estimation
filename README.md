# Synthesizer Parameters Estimation via Semantic Flow Matching

[](example.com)[placeholder paper]

Estimate complex commercial synthesizer parameters from audio recordings using **Continuous Normalizing Flows (CNF)** and **Semantic Feature Distillation**.

## ⚠️ Provenance & Contributions

This is a fork and extension of [**synth-permutations**](https://github.com/ben-hayes/synth-permutations) by Benjamin Hayes.

- **The Foundation:** We utilize their core generative architecture—Equivariant CNFs, Diffusion Transformers (DiT), and the `PARAM2TOK` module—to handle synthesizer non-injectivity.
- **Our Contribution:** Standard flow models overfit to synthetic data. To bridge the **synthetic-to-real domain gap** and handle Out-of-Distribution (OOD) acoustic instruments, we introduce:
  1.  **Semantic Feature Distillation:** Aligning the AST encoder with a frozen CLAP teacher.
  2.  **Preset-Centered Data:** Constraining training to a human-curated preset space to bypass the computational limits of uniform sampling.

## 📖 Overview

Inverting a VST is a hostile optimization task due to massive parameter spaces, non-linearities, and **non-injectivity** (different parameters making the exact same sound). We solve this by treating parameter estimation as a probabilistic generative task rather than direct regression. _(Note: Scripts should support both `Surge XT` and `Vital` but everything has only been tested for Vital)._

## 🔬 Key Findings

We evaluated our strategies on OOD real-world audio (NSynth dataset).

1.  **The Scaling Wall:** Uniform parameter sampling improves logarithmically with scale but yields mostly unmusical noise, making it computationally intractable. Constraining the dataset to **Presets** bypasses this scaling wall but fails to generalize to real-world sounds.
2.  **Semantic Distillation:** Distilling features from a frozen CLAP model creates a successful trade-off: it sacrifices minor synthetic spectral accuracy for **significantly improved alignment and robustness on physical acoustic instruments**.

### Results Summary

**In-Distribution (Synthetic Eval Dataset, N=10000)**
| Metric | Uniform (Base) | Preset Centered | Distill (Ours) |
| :--- | :--- | :--- | :--- |
| **MSS** ↓ | 8.461 ± 0.090 | **5.853 ± 0.075** | 7.208 ± 0.079 |
| **wMFCC** ↓ | 26.817 ± 0.241 | **16.168 ± 0.147** | 19.665 ± 0.188 |
| **SOT** ↓ | 0.154 ± 0.002 | **0.090 ± 0.002** | 0.110 ± 0.002 |
| **RMS** ↑ | 0.880 ± 0.002 | **0.939 ± 0.001** | 0.878 ± 0.003 |

**Out-of-Distribution (Real-World NSynth Dataset, N=20000)**
| Metric | Uniform (Base) | Preset Centered | Distill (Ours) |
| :--- | :--- | :--- | :--- |
| **MSS** ↓ | 38.894 ± 0.351 | 41.764 ± 0.327 | **34.705 ± 0.313** |
| **wMFCC** ↓ | 49.396 ± 0.292 | 51.875 ± 0.298 | **48.221 ± 0.277** |
| **SOT** ↓ | **0.464 ± 0.004** | 0.512 ± 0.004 | 0.484 ± 0.004 |
| **RMS** ↑ | 0.442 ± 0.006 | 0.342 ± 0.006 | **0.446 ± 0.006** |

---

## 🚀 Quick Start

### Automated Setup

```bash
./scripts/ovh/setup.sh
```

Installs dependencies, creates `.env`, and configures the OVH S3 datastore.

### Manual Setup

```bash
# 1. Install dependencies
sudo apt-get install -y docker.io jq python3-pip
pip install uv
uv pip install --system -r pyproject.toml

# 2. Install OVH CLI
curl -fsSL https://cli.bhs.ai.cloud.ovh.net/install.sh | bash
export PATH="$HOME/.ovhai/bin:$PATH"

# 3. Configure environment
cp .env.example .env
nano .env  # Add credentials
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

## ⌨️ Script Usage

All primary entry points in this repository (`scripts/dataset/generate.sh`, `scripts/train/launch.sh`, and `scripts/eval/evaluate.sh`) are equipped with built-in help menus. To view all configurable arguments and defaults for any script, simply append `--help`:

```bash
./scripts/dataset/generate.sh --help
./scripts/train/launch.sh --help
./scripts/eval/evaluate.sh --help
```

## 💽 Dataset Preparation

### 1\. Generate Shards (120k samples total)

```bash
# Generate 12 shards (120k samples total) using Vital
./scripts/dataset/generate.sh \
  --first-shard 0 --last-shard 11 \
  --samples 10000 \
  --output-dir datasets/vital-100k \
  --plugin-path "plugins/Vital.vst3" \
  --preset-dir "data/presets/vital_single" 
```

### 2\. Create Virtual Datasets & Subsets

```bash
# Full 100k Dataset
python scripts/dataset/create_subset_dataset.py \
  ./datasets/vital-100k \
  ./datasets/vital-100k \
  --train-shards 0,1,2,3,4,5,6,7,8,9 \
  --val-shards 10 \
  --test-shards 11

# 20k Subset
python scripts/dataset/create_subset_dataset.py \
  ./datasets/vital-100k \
  ./datasets/vital-20k \
  --train-shards 0,1 \
  --val-shards 10 \
  --test-shards 11
```

### 3\. Upload to S3

```bash
ovhai bucket object upload uniform-100k@s3-GRA datasets/vital-100k/
ovhai bucket object upload uniform-100k@s3-GRA datasets/vital-20k/
```

## 🧠 Training & Evaluation

```bash
# Launch training (Builds Docker, submits to OVH, streams logs)
./scripts/train/launch.sh flow_multi/dataset_20k_40k
```

### Resuming Training

Pass the `ckpt_path` and a higher `trainer.max_steps`:

```bash
./scripts/train/launch.sh <experiment> \
  ckpt_path=/workspace/outputs/train/.../checkpoints/last.ckpt \
  trainer.max_steps=1600000
```

### Monitoring Jobs

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

# Run cloud evaluation
./scripts/eval/evaluate.sh \
  --experiment flow_multi/dataset_20k_40k \
  --ckpt path/to/checkpoint.ckpt \
  --dataset vital-100k
```

## 📂 Project Structure

```text
configs/
├── experiment/flow_multi/  # All experiments
├── model/surge_flow.yaml   # Main model config
├── model/encoder/          # Audio encoder configs
├── data/synth.yaml         # Dataset config
├── callbacks/              # Training callbacks
└── trainer/                # PyTorch Lightning configs

src/
├── models/                 # Core architecture & components
├── data/                   # Datamodules & VST rendering
├── train.py                # Training entry point
└── eval.py                 # Evaluation entry point

scripts/
├── dataset/                # Dataset tools
├── eval/                   # Evaluation scripts
├── ovh/                    # Cloud utilities
└── render/                 # Audio rendering

datasets/                   # HDF5 training data
tests/                      # Unit tests
├── train/                  # Training entry points
Dockerfile                  # Container definition
pyproject.toml              # Dependencies
```
