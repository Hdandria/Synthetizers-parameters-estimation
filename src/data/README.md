# Scalable VST Dataset Generator

This tool creates audio datasets from the surge_xt synth with support for both small and large-scale generation.

## Quick Start

### Small Datasets (< 1000 samples)

1. **Edit the config file** - Open `dataset_config.yaml` and change the plugin path:
   ```yaml
   plugin_path: "/path/to/your/plugin.vst3"
   ```

2. **Generate your dataset**:
   ```bash
   python generate_vst_dataset.py --config dataset_config.yaml
   ```

### Large Datasets (1000+ samples) - Recommended

1. **Use the large-scale config** - Edit `dataset_config_large.yaml`:
   ```yaml
   plugin_path: "/path/to/your/plugin.vst3"
   num_samples: 100000
   samples_per_file: 1000
   num_workers: 8
   ```

2. **Generate with chunked approach**:
   ```bash
   python generate_vst_dataset.py --config dataset_config_large.yaml
   ```

3. **Check what you got**:
   Use the notebook `analyse_dataset.ipynb` or the new `ChunkedDatasetReader` for multi-file datasets.

## What it does

The tool will:
- Load your VST plugin
- Generate random parameter settings
- Play random MIDI notes through the plugin
- Save the audio as HDF5 file(s) with all the parameter info
- Use parallel processing for large datasets
- Split large datasets into manageable chunks

## Configuration

### Small Datasets
- `dataset_config.yaml` - Standard configuration
- `original_config.yaml` - 1-1 reproduction of the original paper generation

### Large Datasets
- `dataset_config_large.yaml` - Optimized for 100k+ samples with chunked generation

### Load in Python

#### Single File Dataset
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

#### Chunked Dataset (Recommended for Large Datasets)
```python
from chunked_dataset_reader import ChunkedDatasetReader

# Load chunked dataset
reader = ChunkedDatasetReader('datasets/vst_samples_large')

# Get dataset info
info = reader.get_dataset_info()
print(f"Total samples: {info['total_samples']}")
print(f"Number of chunks: {info['num_chunks']}")

# Get random samples
samples = reader.get_random_samples(10, seed=42)
for sample in samples:
    print(f"Sample {sample['global_id']}: MIDI {sample['midi_note']}")

# Iterate through chunks
for chunk in reader.iterate_chunks():
    audio = chunk['audio'][:]
    # Process chunk...
```

## Dataset Structure

Each sample contains:
- **Audio data**: The actual sound (stereo, 44.1kHz)
- **Parameters**: Values of VST parameters
- **MIDI info**: What note was played and how hard
- **Metadata**: Sample rate, duration, etc.

## Performance & Scalability

### Chunked Generation Benefits
- **Memory efficient**: Only loads chunks as needed
- **Parallel processing**: Multiple workers generate chunks simultaneously
- **Fault tolerant**: Failed chunks don't affect others
- **Resumable**: Can restart from specific chunks (future feature)

### Performance Estimates
Based on current setup:
- **Single core**: ~2.6s per sample
- **4 cores**: ~900 samples/hour
- **8 cores**: ~1800 samples/hour
- **40 cores**: ~9000 samples/hour

For 100k samples with 8 cores: ~55 hours
For 100k samples with 40 cores: ~11 hours

### File Sizes
- **1k samples**: ~1.4GB per chunk
- **100k samples**: ~140GB total (100 chunks of 1.4GB each)
- **1M samples**: ~1.4TB total (1000 chunks of 1.4GB each)

## Command Line Options

```bash
# Basic usage (uses config to determine generation method)
python generate_vst_dataset.py --config dataset_config.yaml

# Large dataset with chunked generation
python generate_vst_dataset.py --config dataset_config_large.yaml

# Test chunked dataset reader
python chunked_dataset_reader.py /path/to/dataset --validate --random-samples 10
```

## Configuration Options

### Chunked Generation Settings
- `force_chunked: true/false` - Force chunked generation even for small datasets
- `samples_per_file: 1000` - Number of samples per HDF5 file
- `file_naming: "dataset_{chunk:04d}.h5"` - File naming pattern
- `num_workers: 8` - Number of parallel workers

### Automatic Mode Selection
- **Chunked mode** is used when:
  - `force_chunked: true` is set, OR
  - `num_samples > 1000`
- **Legacy mode** is used for smaller datasets unless `force_chunked: true`