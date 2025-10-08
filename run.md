Perfect! Let me update the commands with your specific dataset path `datasets/experiment_1`. Here's the complete reproduction guide with your path:

# Reproducing Surge XT Simple Results with 20,000 Examples

## Step-by-Step Reproduction Guide with Your Path

### 1. Environment Setup

```bash
# Clone and setup the repository
git clone https://github.com/ben-hayes/synth-permutations.git
cd synth-permutations

# Install dependencies
uv sync
# or
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Download Surge XT VST Plugin

```bash
# Download Surge XT from https://surge-synthesizer.github.io/
# Place the VST3 file in the project directory
mkdir -p vsts
# Copy Surge XT.vst3 to vsts/
```

### 3. Generate Surge XT Simple Dataset (20,000 samples)

**Note: CPU Core Control for Dataset Generation**

The dataset generation process uses CPU-intensive operations (audio processing, spectrogram computation). You can control the number of CPU cores used by setting environment variables before running the generation commands:

```bash
# Set the number of CPU cores to use (adjust as needed)
export OMP_NUM_THREADS=8      # OpenMP threads (for general parallel processing)
export MKL_NUM_THREADS=8      # Intel MKL threads (if using Intel libraries)
export TORCH_NUM_THREADS=8    # PyTorch threads (for tensor operations)
```

**Recommended settings:**
- For systems with 8+ cores: Use 6-8 cores
- For systems with 4 cores: Use 2-3 cores  
- Leave some cores free for system processes

```bash
# Create dataset directory
mkdir -p datasets/experiment_1

# Generate the training dataset
# Set CPU cores to use (adjust the number as needed)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

LOGURU_LEVEL=ERROR python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/train.h5 \
    20000 \
    --plugin_path vsts/Surge\ XT.vst3 \
    --preset_path presets/surge-simple.vstpreset \
    --param_spec surge_simple \
    --sample_batch_size 16 \
    --signal_duration_seconds 4.0 \
    --sample_rate 44100.0 \
    --channels 2 \
    --velocity 100 \
    --min_loudness -50.0
    
python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/train.h5 \
    20000 \
    --plugin_path vsts/Surge\ XT.vst3 \
    --preset_path presets/surge-simple.vstpreset \
    --param_spec surge_simple \
    --sample_batch_size 16 \
    --signal_duration_seconds 4.0 \
    --sample_rate 44100.0 \
    --channels 2 \
    --velocity 100 \
    --min_loudness -50.0

# Generate validation set (10% of training)
python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/val.h5 \
    2000 \
    --plugin_path vsts/Surge\ XT.vst3 \
    --preset_path presets/surge-simple.vstpreset \
    --param_spec surge_simple \
    --sample_batch_size 16 \
    --signal_duration_seconds 4.0 \
    --sample_rate 44100.0 \
    --channels 2 \
    --velocity 100 \
    --min_loudness -50.0

# Generate test set (10% of training)
python src/data/vst/generate_vst_dataset.py \
    datasets/experiment_1/test.h5 \
    2000 \
    --plugin_path vsts/Surge\ XT.vst3 \
    --preset_path presets/surge-simple.vstpreset \
    --param_spec surge_simple \
    --sample_batch_size 16 \
    --signal_duration_seconds 4.0 \
    --sample_rate 44100.0 \
    --channels 2 \
    --velocity 100 \
    --min_loudness -50.0
```

### 4. Compute Dataset Statistics

```bash
# Compute mel spectrogram statistics for normalization
python scripts/get_dataset_stats.py datasets/experiment_1/train.h5
```

### 5. Update Configuration

Create a local configuration file for your dataset path:

```bash
# Create local config
mkdir -p configs/local
cat > configs/local/local.yaml << EOF
# @package _global_

# Override paths for your local setup
paths:
  data_dir: "datasets"
  log_dir: "logs"
  checkpoint_dir: "checkpoints"
EOF
```

### 6. Train the Models

#### Flow Matching (Main Approach)
```bash
# Train flow matching model
python src/train.py experiment=surge/flow_simple \
    data.dataset_root=datasets/experiment_1 \
    data.num_workers=8 \
    trainer.max_epochs=200 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=0.5 \
    trainer.devices=[2] \
    tags="[surge_simple, flow_matching, 20k]"
```

#### Feed-Forward Network (Baseline)
```bash
# Train FFN baseline
python src/train.py experiment=surge/ffn_simple \
    data.dataset_root=datasets/experiment_1 \
    trainer.max_epochs=200 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=0.5 \
    trainer.devices=[2] \
    tags="[surge_simple, ffn, 20k]"
```

#### VAE Baseline
```bash
# Train VAE baseline
python src/train.py experiment=surge/vae_simple \
    data.dataset_root=datasets/experiment_1 \
    trainer.max_epochs=200 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=0.5 \
    trainer.devices=[2] \
    tags="[surge_simple, vae, 20k]"
```

### 7. Evaluation

After training, evaluate each model:

```bash
# Evaluate flow matching model
python src/eval.py \
    ckpt_path=checkpoints/flow_simple/best.ckpt \
    data.dataset_root=datasets/experiment_1 \
    mode=test

# Evaluate FFN model
python src/eval.py \
    ckpt_path=checkpoints/ffn_simple/best.ckpt \
    data.dataset_root=datasets/experiment_1 \
    mode=test

# Evaluate VAE model
python src/eval.py \
    ckpt_path=checkpoints/vae_simple/best.ckpt \
    data.dataset_root=datasets/experiment_1 \
    mode=test
```

## Alternative: Use Pre-configured Data Config

You can also create a custom data configuration file:

```bash
# Create custom data config
cat > configs/data/surge_simple_custom.yaml << EOF
# @package _global_

_target_: src.data.surge_datamodule.SurgeDataModule
dataset_root: datasets/experiment_1
use_saved_mean_and_variance: true
batch_size: 128
ot: true
num_workers: 8  # Change this to your desired number of CPU cores
fake: false
repeat_first_batch: false
predict_file: null
conditioning: "mel"
EOF
```

Then use it in training:

```bash
# Train with custom data config
python src/train.py data=surge_simple_custom experiment=surge/flow_simple \
    trainer.max_epochs=200 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=0.5 \
    trainer.devices=[2] \
    tags="[surge_simple, flow_matching, 20k]"
```

## Monitoring Training

### With Weights & Biases
```bash
# Enable W&B logging
python src/train.py experiment=surge/flow_simple \
    data.dataset_root=datasets/experiment_1 \
    logger=wandb \
    trainer.devices=[2] \
    tags="[surge_simple, flow_matching, 20k]"
```

### With TensorBoard
```bash
# Enable TensorBoard logging
python src/train.py experiment=surge/flow_simple \
    data.dataset_root=datasets/experiment_1 \
    logger=tensorboard \
    trainer.devices=[2] \
    tags="[surge_simple, flow_matching, 20k]"

# View results
tensorboard --logdir logs
```

## Expected File Structure

After setup, your directory structure should look like:

```
synth-permutations/
├── datasets/
│   └── experiment_1/
│       ├── train.h5          # 20,000 training samples
│       ├── val.h5            # 2,000 validation samples
│       ├── test.h5           # 2,000 test samples
│       └── stats.npz         # Dataset statistics
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
├── vsts/
│   └── Surge XT.vst3        # VST plugin
└── configs/local/
    └── local.yaml           # Local configuration
```

## Quick Start Commands

If you want to get started quickly, here are the essential commands:

```bash
# 1. Generate dataset (this takes the longest time)
# Set CPU cores to use (adjust the number as needed)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_NUM_THREADS=8

python src/data/vst/generate_vst_dataset.py datasets/experiment_1/train.h5 20000 --param_spec surge_simple

# 2. Compute statistics
python scripts/get_dataset_stats.py datasets/experiment_1/train.h5

# 3. Train flow matching model
python src/train.py experiment=surge/flow_simple data.dataset_root=datasets/experiment_1 trainer.precision=bf16-mixed trainer.devices=[2]

# 4. Evaluate
python src/eval.py ckpt_path=checkpoints/flow_simple/best.ckpt data.dataset_root=datasets/experiment_1
```

This setup will reproduce the paper's results using your `datasets/experiment_1` path. The key is ensuring the VST plugin is properly installed and the dataset generation completes successfully.



Notes:
- No hd5f / hdf5plugin in requirements.txt
- NO HEADLESS