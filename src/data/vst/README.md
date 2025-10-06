# VST Parameter Selection

This module provides structured parameter specifications for VST plugins, specifically Surge XT.

## Parameter Sets

### `surge_simple` (92 audio parameters)
Essential parameters for basic sound generation:
- Core envelope parameters (attack, decay, sustain, release)
- Filter parameters (cutoff, resonance, FEG modulation)
- Oscillator parameters (waveforms, volume, pitch, sync)
- Complete LFO parameters (all 6 LFOs with full envelope)
- Noise parameters
- Oscillator drift and width

### `surge_full` (165 audio parameters)
Comprehensive parameter set including:
- All parameters from `surge_simple`
- Advanced filter configurations and types
- Waveshaper parameters
- Ring modulation
- Effects parameters (chorus, delay, reverb)
- Categorical parameters (envelope modes, filter types, LFO types)
- Unison and FM parameters
- Complete oscillator routing and mute controls

## Usage

In your `dataset_config.yaml`:

```yaml
# Use simple parameter set
param_set: "surge_simple"

# Or use full parameter set
param_set: "surge_full"
```

## Parameter Types

- **ContinuousParameter**: Continuous values with min/max bounds
- **CategoricalParameter**: Discrete values with optional weights
- **DiscreteLiteralParameter**: Integer values (like MIDI notes)
- **NoteDurationParameter**: Note timing parameters

## Adding New Parameter Sets

1. Define your parameter specification in `surge_params.py`
2. Add it to the `get_param_spec()` function
3. Update your configuration file to use the new set
