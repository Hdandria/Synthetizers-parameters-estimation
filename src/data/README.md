# Scalable VST Dataset Generator

> To be added: the plugin under /plugins/ is Linux only

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

Look at `notebooks/analyze_dataset.ipynb` to view how to load the samples.

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
Empirically, on AMD EPYC 7542 (32 cores):

On each core, we have the approximate law:
t = 3.7 * nb_sample + 16

Generally, based on current samples:
- **Single core**: ~1000 samples/hours

For 100k samples with 8 cores: ~12.5h
For 100k samples with 40 cores: ~2.5h

### File Sizes
- **1k samples**: ~1.1GB (includes audio, mel-spec, and parameters)
- **100k samples**: ~110GB total (100 chunks of 1.1GB each)
- **1M samples**: ~1.1TB total (1000 chunks of 1.1GB each)

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