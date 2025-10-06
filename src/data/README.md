# Simple VST Dataset Generator

This tool creates audio datasets from the surge_xt synth.

## Quick Start

1. **Edit the config file** - Open `dataset_config.yaml` and change the plugin path:
   ```yaml
   plugin_path: "/path/to/your/plugin.vst3"
   ```

2. **Generate your dataset**:
   ```bash
   python generate_vst_dataset.py --config dataset_config.yaml
   ```

3. **Check what you got**:
   Use the notebook `analyse_dataset.ipynb` to get some basic informations.

## What it does

The tool will:
- Load your VST plugin
- Generate random parameter settings
- Play random MIDI notes through the plugin
- Save the audio as an HDF5 file with all the parameter info

## Configuration

The `dataset_config.yaml` file controls everything:

### Load in Python
```python
import h5py
import numpy as np

# Load the dataset
with h5py.File('datasets/vst_samples/dataset.h5', 'r') as f:
    audio = f['audio'][:]           # Shape: (num_samples, channels, samples)
    parameters = f['parameters'][:] # Parameter strings
    midi_notes = f['midi_notes'][:] # MIDI note numbers
    velocities = f['velocities'][:] # MIDI velocities

print(f"Loaded {len(audio)} samples")
```

## Dataset structure:

Each sample contains:
- **Audio data**: The actual sound (stereo, 44.1kHz)
- **Parameters**: Values of VST parameters
- **MIDI info**: What note was played and how hard
- **Metadata**: Sample rate, duration, etc.