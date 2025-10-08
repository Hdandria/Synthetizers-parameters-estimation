# Documentation Index

Welcome to the synth-permutations documentation! This directory contains comprehensive documentation for the project.

## 📚 Documentation Overview

### Core Documentation

- **[Main README](../README.md)** - Project overview, installation, and quick start guide
- **[API Documentation](API.md)** - Complete API reference for all modules and functions
- **[Configuration Guide](CONFIGURATION.md)** - Comprehensive guide to the Hydra configuration system
- **[Usage Guide](USAGE_GUIDE.md)** - Detailed tutorials and examples for using the project

## 🚀 Quick Navigation

### For New Users
1. Start with the [Main README](../README.md) for project overview
2. Follow the [Usage Guide](USAGE_GUIDE.md) for step-by-step tutorials
3. Refer to [API Documentation](API.md) for detailed function references

### For Developers
1. Review the [Configuration Guide](CONFIGURATION.md) for system architecture
2. Use [API Documentation](API.md) for implementation details
3. Check [Usage Guide](USAGE_GUIDE.md) for advanced techniques

### For Researchers
1. Read the [Main README](../README.md) for research context
2. Explore [Configuration Guide](CONFIGURATION.md) for experiment setup
3. Use [Usage Guide](USAGE_GUIDE.md) for reproducible experiments

## 📖 Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── API.md                 # Complete API reference
├── CONFIGURATION.md       # Configuration system guide
└── USAGE_GUIDE.md         # Tutorials and examples
```

## 🔍 What's Documented

### API Documentation (`API.md`)
- **Training and Evaluation**: `src.train.train`, `src.eval.evaluate`
- **Data Modules**: `KSinDataModule`, `SurgeDataModule`
- **Models**: `SurgeFlowMatchingModule`, `ApproxEquivTransformer`
- **Metrics**: `LogSpectralDistance`, `ChamferDistance`, `LinearAssignmentDistance`
- **Utilities**: Helper functions and classes

### Configuration Guide (`CONFIGURATION.md`)
- **Main Configuration Files**: `train.yaml`, `eval.yaml`
- **Data Configurations**: Dataset-specific settings
- **Model Configurations**: Architecture and training parameters
- **Experiment Configurations**: Pre-configured experiments
- **Advanced Usage**: Command-line overrides, multi-run experiments

### Usage Guide (`USAGE_GUIDE.md`)
- **Quick Start**: Installation and first run
- **Basic Training**: k-sin and Surge XT datasets
- **Evaluation**: Model testing and prediction
- **Data Preparation**: Dataset generation and setup
- **Advanced Techniques**: Hyperparameter optimization, distributed training
- **Troubleshooting**: Common issues and solutions

## 🎯 Key Concepts

### Flow Matching
The project implements flow matching for parameter estimation, which is a generative modeling technique that learns to transform noise into target parameters through a learned vector field.

### Permutation Invariance
A key challenge addressed by this project is handling permutation-invariant parameter spaces, where parameters can be reordered without changing the resulting audio signal.

### Approximately Equivariant Architectures
The project uses approximately equivariant neural networks that respect permutation symmetries in the parameter space while maintaining expressiveness.

## 🛠️ Getting Started

1. **Install the project** (see [Main README](../README.md))
2. **Run your first experiment** (see [Usage Guide](USAGE_GUIDE.md))
3. **Customize configurations** (see [Configuration Guide](CONFIGURATION.md))
4. **Explore the API** (see [API Documentation](API.md))

## 📝 Contributing to Documentation

If you find issues with the documentation or want to contribute:

1. **Report issues** via GitHub issues
2. **Suggest improvements** via pull requests
3. **Add examples** to the usage guide
4. **Improve API documentation** with better descriptions

## 🔗 External Resources

- **Project Repository**: [GitHub](https://github.com/ben-hayes/synth-permutations)
- **Research Paper**: [ISMIR 2025 Submission](https://benhayes.net/synth-perm/)
- **Audio Examples**: [Online Supplement](https://benhayes.net/synth-perm/)
- **PyTorch Lightning**: [Documentation](https://lightning.ai/docs/pytorch/stable/)
- **Hydra**: [Documentation](https://hydra.cc/)

## 📞 Support

For questions about the documentation or the project:

1. **Check existing documentation** first
2. **Search GitHub issues** for similar problems
3. **Open a new issue** with detailed information
4. **Contact the authors** for research-related questions

---

**Note**: This documentation is actively maintained and updated with the project. If you find any inconsistencies or missing information, please report them via GitHub issues.
