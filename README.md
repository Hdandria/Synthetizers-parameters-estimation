# Synthesizer Parameter Estimation

Train ML models to predict synthesizer parameters from audio on OVH AI Training.

```bash
./scripts/setup.sh
cp .env.example .env && nano .env
./launch.sh surge/base
```

See [QUICKSTART.md](QUICKSTART.md).

## Quick Start

### Option 1: Docker (Recommended)
For GPU-optimized local training:

```bash
# Clone repository
git clone <repository-url>
cd Synthetizers-parameters-estimation

# Configure environment
cp env.example .env
# Edit .env with your Wandb credentials (optional)

# Build and run
./docker/build.bat
./docker/run_local.bat surge/base
```

See [DOCKER.md](DOCKER.md) for Docker setup instructions.

### Option 2: Local Installation
For development and customization:

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run training
python src/train.py experiment=surge/baseline
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

## Requirements

- Python 3.10+
- PyTorch with GPU support recommended
- VST3 plugins (for dataset generation with Surge XT)
