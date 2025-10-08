# Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow Matching

## Notes
- No multi threading for dataset generation
- hd5f and hdf5plugin not in requirements.txt
- GUI necessary (no headless)
---

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)

This repository contains the implementation for the research paper "Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow Matching" submitted to ISMIR 2025. The project explores the effect of permutation symmetry on audio synthesizer parameter inference using flow matching techniques.

## 🎯 Overview

This research addresses the challenge of inferring synthesizer parameters from audio signals, with a particular focus on handling permutation-invariant parameter spaces. The work introduces approximately equivariant flow matching models that can handle symmetric parameter spaces more effectively than traditional approaches.

### Key Contributions

- **Approximately Equivariant Flow Matching**: Novel flow matching approach that respects permutation symmetries in synthesizer parameter spaces
- **Symmetric Parameter Space Handling**: Methods to handle permutation-invariant labels in audio synthesis parameter estimation
- **Comprehensive Evaluation**: Evaluation on both synthetic (k-sin) and real-world (Surge XT synthesizer) datasets
- **Multiple Model Architectures**: Implementation of various model types including transformers, CNNs, and MLPs

## 🏗️ Project Structure

```
synth-permutations/
├── src/                          # Main source code
│   ├── data/                     # Data modules and datasets
│   │   ├── ksin_datamodule.py    # k-sin synthetic dataset
│   │   ├── surge_datamodule.py   # Surge XT synthesizer dataset
│   │   ├── vst/                  # VST plugin integration
│   │   └── ot.py                 # Optimal transport utilities
│   ├── models/                   # Model implementations
│   │   ├── components/           # Reusable model components
│   │   │   ├── transformer.py    # DiT and AST implementations
│   │   │   ├── cnn.py            # CNN encoder implementations
│   │   │   ├── residual_mlp.py   # Residual MLP implementations
│   │   │   └── vae.py            # VAE+RealNVP baseline
│   │   ├── surge_flow_matching_module.py  # Main Surge flow matching
│   │   └── ksin_flow_matching_module.py   # k-sin flow matching
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation script
│   └── metrics.py                # Evaluation metrics
├── configs/                      # Hydra configuration files
│   ├── data/                     # Dataset configurations
│   ├── model/                    # Model configurations
│   ├── experiment/               # Experiment-specific configs
│   └── trainer/                  # Training configurations
├── scripts/                      # Utility scripts
├── jobs/                         # Training and evaluation job scripts
└── notebooks/                    # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ben-hayes/synth-permutations.git
   cd synth-permutations
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   # Copy and modify local configuration
   cp configs/local/default.yaml configs/local/local.yaml
   # Edit paths and settings as needed
   ```

### Basic Usage

#### Training a Model

```bash
# Train on k-sin dataset with flow matching
python src/train.py data=ksin model=ksin_flow_matching

# Train on Surge XT dataset
python src/train.py data=surge model=surge_flow_matching

# Override specific parameters
python src/train.py data=ksin model=ksin_flow_matching data.k=4 model.d_model=512
```

#### Evaluation

```bash
# Evaluate a trained model
python src/eval.py ckpt_path=path/to/checkpoint.ckpt

# Run predictions
python src/eval.py ckpt_path=path/to/checkpoint.ckpt mode=predict
```

## 📊 Datasets

### k-Sin Dataset
A synthetic dataset designed to test permutation-invariant parameter estimation:
- **Purpose**: Simple synthetic synthesizer with k sinusoidal components
- **Parameters**: Frequency and amplitude for each sinusoid
- **Challenge**: Permutation symmetry in parameter ordering
- **Configuration**: `configs/data/ksin.yaml`

### Surge XT Dataset
Real-world synthesizer parameter estimation:
- **Synthesizer**: Surge XT open-source synthesizer
- **Parameters**: 189 continuous and categorical parameters
- **Audio**: 4-second stereo audio samples at 44.1kHz
- **Features**: Mel spectrograms and Music2Latent embeddings
- **Configuration**: `configs/data/surge.yaml`

## 🧠 Models

### Flow Matching Models
- **SurgeFlowMatchingModule**: Main flow matching implementation for Surge XT
- **KSinFlowMatchingModule**: Flow matching for k-sin dataset
- **Features**: 
  - Conditional generation with classifier-free guidance
  - Approximately equivariant architectures
  - Multiple sampling strategies

### Encoder Architectures
- **AudioSpectrogramTransformer (AST)**: Transformer-based audio encoder
- **CNN Encoders**: Convolutional encoders for audio features
- **Residual MLPs**: Multi-layer perceptron encoders

### Vector Field Networks
- **ApproxEquivTransformer**: Approximately equivariant transformer
- **LearntProjection**: Learnable parameter-to-token projections
- **PositionalEncoding**: Various positional encoding strategies

## ⚙️ Configuration

The project uses Hydra for configuration management. Key configuration categories:

### Data Configuration (`configs/data/`)
- `ksin.yaml`: k-sin dataset settings
- `surge.yaml`: Surge XT dataset settings
- `mnist.yaml`: MNIST baseline (for comparison)

### Model Configuration (`configs/model/`)
- `surge_flow_matching.yaml`: Surge flow matching model
- `ksin_flow_matching.yaml`: k-sin flow matching model
- `flow.yaml`: Generic flow matching model

### Experiment Configuration (`configs/experiment/`)
- Pre-configured experiments with optimal hyperparameters
- Organized by dataset (surge/, ksin/, etc.)

## 🔬 Key Features

### Approximately Equivariant Architectures
- **LearntProjection**: Learnable parameter-to-token mappings
- **PositionalEncoding**: Structured positional encodings
- **Penalty Terms**: Regularization for equivariance

### Flow Matching Implementation
- **Rectified Flow**: Improved probability paths
- **Classifier-Free Guidance**: Conditional generation
- **Multiple Sampling**: RK4 and Euler integration

### Evaluation Metrics
- **ChamferDistance**: Permutation-invariant distance
- **LinearAssignmentDistance**: Optimal assignment distance
- **SpectralDistance**: Audio quality metrics
- **LogSpectralDistance**: Perceptual audio quality

### Optimal Transport
- **Hungarian Matching**: Optimal minibatch coupling
- **Sinkhorn Algorithm**: Approximate optimal transport

## 📈 Training and Evaluation

### Training Process
1. **Data Loading**: Efficient HDF5-based data loading
2. **Flow Matching**: Train vector field to match probability paths
3. **Regularization**: Apply equivariance penalties
4. **Logging**: Comprehensive experiment tracking

### Evaluation Process
1. **Sampling**: Generate parameters from trained model
2. **Audio Synthesis**: Render audio from predicted parameters
3. **Metrics**: Compute permutation-invariant distances
4. **Analysis**: Compare with ground truth parameters

## 🛠️ Development

### Running Tests
```bash
# Run all tests
make test-full

# Run fast tests only
make test

# Run specific test
pytest tests/test_datamodules.py
```

### Code Quality
```bash
# Format code
make format

# Clean generated files
make clean
```

### Adding New Models
1. Create model class in `src/models/`
2. Add configuration in `configs/model/`
3. Update experiment configs if needed
4. Add tests in `tests/`

## 📚 Documentation

### API Reference
- **Data Modules**: `src/data/` - Dataset implementations
- **Models**: `src/models/` - Model architectures
- **Metrics**: `src/metrics.py` - Evaluation metrics
- **Utilities**: `src/utils/` - Helper functions

### Configuration Reference
- **Hydra Configs**: `configs/` - All configuration files
- **Experiment Configs**: `configs/experiment/` - Pre-configured experiments
- **Model Configs**: `configs/model/` - Model architectures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hayes2025audio,
  title={Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow Matching},
  author={Hayes, Benjamin and others},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2025}
}
```

## 📞 Contact

- **Author**: Benjamin Hayes
- **Email**: [Contact information]
- **Website**: [https://benhayes.net/synth-perm/](https://benhayes.net/synth-perm/)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Surge XT synthesizer team for the open-source synthesizer
- PyTorch Lightning team for the training framework
- Hydra team for the configuration management
- The broader MIR and ML communities for inspiration and feedback

---

**Note**: This repository accompanies a submission to ISMIR 2025. Audio examples and additional resources are available at the [online supplement](https://benhayes.net/synth-perm/).