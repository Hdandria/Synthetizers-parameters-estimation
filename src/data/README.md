# Scalable VST Dataset Generator

This tool creates audio datasets from the surge_xt synth using a chunked approach for all dataset sizes.

## Quick Start

1. **Edit the config file** - Choose your configuration:
   - `original_config.yaml` - Small datasets (20 samples)
   - `dataset_config_large.yaml` - Large datasets (100k+ samples)
   
   Update the plugin path:
   ```yaml
   plugin_path: "/path/to/your/plugin.vst3"
   ```

2. **Generate your dataset**:
   ```bash
   python generate_vst_dataset.py --config original_config.yaml
   ```

3. **Check what you got**:
   Use the notebook `analyse_dataset.ipynb` or the `ChunkedDatasetReader` for analysis.

## What it does

The tool will:
- Load your VST plugin
- Generate random parameter settings
- Play random MIDI notes through the plugin
- Save the audio as HDF5 file(s) with all the parameter info
- Use parallel processing for all dataset sizes
- Split datasets into manageable chunks (even for small datasets)

## Configuration

### Available Configurations
- `original_config.yaml` - Small datasets (20 samples) - good for testing
- `dataset_config_large.yaml` - Large datasets (100k+ samples) - for production use

### Load in Python

#### Chunked Dataset (All datasets now use this format)
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

Each dataset chunk contains:

### Datasets
- **`audio`**: Raw audio waveform (float16) — shape `(num_samples, channels, sample_rate * signal_duration_seconds)`
- **`mel_spec`**: Mel-spectrogram (float32) — shape `(num_samples, 2, 128, 401)` (128 mels, ~401 frames; second dimension = 2)
- **`param_array`**: Encoded parameters (float32) — shape `(num_samples, num_params)`. This is the concatenated encoded synth + note parameters from the ParamSpec
- **`parameters`**: Parameter strings (legacy compatibility)
- **`midi_notes`**: MIDI note numbers (int16)
- **`velocities`**: MIDI velocities (int16)

### Audio Dataset Attributes
- **`velocity`**: MIDI velocity used
- **`signal_duration_seconds`**: Duration of each audio sample
- **`sample_rate`**: Audio sample rate (44100 Hz)
- **`min_loudness`**: Minimum loudness threshold used during generation

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
- **1k samples**: ~2.5GB per chunk (includes audio, mel-spec, and parameters)
- **100k samples**: ~250GB total (100 chunks of 2.5GB each)
- **1M samples**: ~2.5TB total (1000 chunks of 2.5GB each)

### Data Breakdown per Sample (4 seconds, 44.1kHz, stereo)
- **Audio**: 2 channels × 176,400 samples × 2 bytes (float16) = ~705KB
- **Mel-spec**: 2 channels × 128 mels × 401 frames × 4 bytes (float32) = ~410KB  
- **Parameters**: ~50 parameters × 4 bytes (float32) = ~200 bytes
- **Total per sample**: ~1.1MB

## Command Line Options

```bash
# Basic usage
python generate_vst_dataset.py --config original_config.yaml
```