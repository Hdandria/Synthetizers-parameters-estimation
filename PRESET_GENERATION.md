# Preset-Based Dataset Generation

This document explains the methodology and usage of the preset-based dataset generation tools.

## Methodology

The goal is to generate a dataset of synthesizer sounds that are representative of real-world usage, rather than random parameter combinations. To achieve this, we use a "sampling near presets" approach.

1.  **Preset Loading**: We load existing `.vstpreset` or `.vital` files from a library.
2.  **Parameter Extraction**: We extract the parameter values from the loaded preset.
3.  **Perturbation**: We apply Gaussian noise to the **continuous** parameters (e.g., cutoff, resonance, envelope times).
    *   **Categorical** parameters (e.g., filter types, oscillator waveforms, switches) are kept **fixed**. This preserves the fundamental character and structure of the sound (the "patch").
    *   The noise level is configurable (default standard deviation: 0.1).
4.  **Rendering**: We render audio samples using the perturbed parameters.

This approach ensures that the generated dataset covers the parameter space around "good" sounds, providing a more relevant training set for parameter estimation.

## Usage

### 1. Generating the Dataset

Use the `src/data/vst/generate_preset_dataset.py` script.

```bash
uv run src/data/vst/generate_preset_dataset.py \
    <OUTPUT_H5_FILE> \
    <NUM_SAMPLES> \
    --plugin_path <PATH_TO_PLUGIN> \
    --preset_dir <PATH_TO_PRESETS> \
    --param_spec <PARAM_SPEC_NAME> \
    [--noise_level <FLOAT>] \
    [--num_workers <INT>]
```

**Arguments:**
*   `OUTPUT_H5_FILE`: Path to the output HDF5 file (e.g., `datasets/my-dataset/shard-0.h5`).
*   `NUM_SAMPLES`: Number of samples to generate.

**Options:**
*   `-p, --plugin_path`: Path to the VST3 plugin file (default: `vsts/Surge XT.vst3`).
*   `-D, --preset_dir`: Directory containing `.vstpreset` or `.vital` files.
*   `-t, --param_spec`: Parameter specification name (e.g., `vital_simple`, `surge_xt`).
*   `-n, --noise_level`: Standard deviation of Gaussian noise (default: 0.1).
*   `-w, --num_workers`: Number of parallel workers (default: 1).

**Example:**

```bash
uv run src/data/vst/generate_preset_dataset.py \
    datasets/vital-presets/shard-0.h5 \
    10000 \
    -p "plugins/Vital.vst3" \
    -D "presets" \
    -t "vital_simple" \
    -n 0.1 \
    -w 4
```

### 2. Verifying the Dataset

Use the `scripts/dataset/verify_dataset_audio.py` script to inspect the generated audio.

This script reads the first few samples from the dataset, checks for silence, and exports them as WAV files for manual listening.

```bash
# Edit the script to point to your dataset file if needed, or run it directly if it accepts args (currently hardcoded in main block)
# TODO: Update script to accept CLI args for flexibility.
```

**Current Usage:**
The script currently has the dataset path hardcoded in the `if __name__ == "__main__":` block. You may need to edit `scripts/dataset/verify_dataset_audio.py` to point to your specific dataset file.

```python
if __name__ == "__main__":
    verify_and_export_audio("datasets/your-dataset/shard-0.h5", "debug_audio")
```

Then run:

```bash
uv run scripts/dataset/verify_dataset_audio.py
```

The WAV files will be saved to `debug_audio/`.
