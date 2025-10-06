#!/usr/bin/env python3
"""
Simple VST Dataset Generator

This script creates audio datasets from VST plugins. Just point it at your plugin
and it will generate random audio samples with different parameter settings.

Usage:
    python generate_vst_dataset.py --config dataset_config.yaml
"""

import argparse
from pathlib import Path
import random
from typing import Dict, List, Tuple

import h5py
import numpy as np
from pedalboard import VST3Plugin
from pyloudnorm import Meter
from tqdm import tqdm
from vst.surge_params import SURGE_SIMPLE_PARAM_SPEC, SURGE_XT_PARAM_SPEC
import yaml


def load_config(config_path: str) -> Dict:
    """Load the configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_plugin(plugin_path: str) -> VST3Plugin:
    """Load the VST plugin"""
    print(f"Loading plugin: {plugin_path}")
    plugin = VST3Plugin(plugin_path)

    # Give the plugin a moment to initialize
    plugin.process([], 0.1, 44100, 2, 2048, True)
    print("Plugin loaded successfully!")
    return plugin


def get_param_spec(param_set: str):
    """Get parameter specification based on config"""
    if param_set == "surge_full":
        return SURGE_XT_PARAM_SPEC
    elif param_set == "surge_simple":
        return SURGE_SIMPLE_PARAM_SPEC
    else:
        raise ValueError(f"Unknown parameter set: {param_set}")


def make_midi_note(pitch: int, velocity: int, start_time: float, end_time: float):
    """Create a simple MIDI note"""
    import mido

    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)

    return [(note_on.bytes(), start_time), (note_off.bytes(), end_time)]


def generate_audio_sample(plugin: VST3Plugin, config: Dict) -> Tuple[np.ndarray, Dict, int, int]:
    """Generate one audio sample with random settings, filtering by loudness"""

    while True:
        # Reset plugin to clean state
        plugin.reset()
        plugin.process([], 0.1, 44100, 2, 2048, True)

        # Get parameters using structured specification
        if "param_set" not in config:
            raise ValueError("param_set is required in configuration. Please specify 'surge_simple' or 'surge_full'")

        param_spec = get_param_spec(config["param_set"])
        synth_params, note_params = param_spec.sample()
        all_params = {**synth_params, **note_params}
        # print(f"Using parameter set: {config['param_set']} ({len(all_params)} parameters)")

        # Extract note parameters
        midi_note = int(note_params["pitch"])
        note_start, note_end = note_params["note_start_and_end"]

        # Set the synth parameters (skip note parameters)
        for param_name, value in all_params.items():
            if param_name in plugin.parameters:
                plugin.parameters[param_name].raw_value = value

        # Let the plugin process the parameter changes
        plugin.process([], 0.1, 44100, 2, 2048, True)
        plugin.reset()

        # Read velocity from config
        velocity = config["velocity"]

        # Create MIDI note using the sampled timing
        duration = config["duration_seconds"]
        midi_events = make_midi_note(midi_note, velocity, note_start, note_end)

        # Generate the audio
        audio = plugin.process(midi_events, duration, config["sample_rate"], 2, 2048, True)

        # Check loudness if min_loudness is specified
        if "min_loudness" in config:
            meter = Meter(config["sample_rate"])
            loudness = meter.integrated_loudness(audio.T)
            if loudness < config["min_loudness"]:
                # print(f"Sample too quiet (loudness: {loudness:.1f} LUFS), regenerating...")
                continue

        # Clean up
        plugin.process([], 0.1, 44100, 2, 2048, True)
        plugin.reset()

        return audio, all_params, midi_note, velocity


def save_dataset(samples: List[Tuple], config: Dict):
    """Save all samples to an HDF5 file"""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / "dataset.h5"

    print(f"Saving {len(samples)} samples to {filename}")

    with h5py.File(filename, "w") as f:
        # Prepare data arrays
        n_samples = len(samples)
        audio_shape = samples[0][0].shape  # (channels, samples)

        # Create datasets
        audio_ds = f.create_dataset("audio", (n_samples, audio_shape[0], audio_shape[1]), dtype=np.float32)
        param_ds = f.create_dataset("parameters", (n_samples,), dtype=h5py.special_dtype(vlen=str))
        midi_ds = f.create_dataset("midi_notes", (n_samples,), dtype=np.int16)
        velocity_ds = f.create_dataset("velocities", (n_samples,), dtype=np.int16)

        # Fill datasets
        for i, (audio, params, midi_note, velocity) in enumerate(samples):
            audio_ds[i] = audio.astype(np.float32)
            param_ds[i] = str(params)  # Store as string for simplicity
            midi_ds[i] = midi_note
            velocity_ds[i] = velocity

    print(f"Dataset saved successfully!")


def main():
    """Main function - does the work"""
    parser = argparse.ArgumentParser(description="Generate VST dataset")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducible results
    random.seed(42)

    # Load the plugin
    plugin = load_plugin(config["plugin_path"])

    # Show parameter selection info
    print(f"Parameter set: {config['param_set']}")
    param_spec = get_param_spec(config["param_set"])
    print(f"Synth parameters: {len(param_spec.synth_params)}")
    print(f"Note parameters: {len(param_spec.note_params)}")

    # Generate samples
    print(f"Generating {config['num_samples']} samples...")
    samples = []

    for i in tqdm(range(config["num_samples"]), desc="Creating samples"):
        try:
            audio, params, midi_note, velocity = generate_audio_sample(plugin, config)
            samples.append((audio, params, midi_note, velocity))
        except Exception as e:
            print(f"Failed to generate sample {i}: {e}")
            continue

    # Save the dataset
    if samples:
        save_dataset(samples, config)
        print(f"Successfully created dataset with {len(samples)} samples!")
    else:
        print("No samples were generated successfully.")


if __name__ == "__main__":
    main()
