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
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
from pedalboard import VST3Plugin
from tqdm import tqdm
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


def get_random_parameters(plugin: VST3Plugin, num_params: int) -> Dict[str, float]:
    """Pick some random parameters and set them to random values"""
    # Get all available parameters
    all_params = list(plugin.parameters.keys())

    # Skip empty or problematic parameters
    safe_params = [p for p in all_params if p and p != "bypass"]

    # Pick random parameters
    if len(safe_params) < num_params:
        print(f"Warning: Only {len(safe_params)} parameters available, wanted {num_params}")
        selected_params = safe_params
    else:
        selected_params = random.sample(safe_params, num_params)

    # Set random values (0.0 to 1.0)
    params = {}
    for param_name in selected_params:
        params[param_name] = random.uniform(0.0, 1.0)

    return params


def make_midi_note(pitch: int, velocity: int, start_time: float, end_time: float):
    """Create a simple MIDI note"""
    import mido

    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)

    return [(note_on.bytes(), start_time), (note_off.bytes(), end_time)]


def generate_audio_sample(plugin: VST3Plugin, config: Dict) -> Tuple[np.ndarray, Dict, int, int]:
    """Generate one audio sample with random settings"""

    # Reset plugin to clean state
    plugin.reset()
    plugin.process([], 0.1, 44100, 2, 2048, True)

    # Get random parameters
    params = get_random_parameters(plugin, config["num_random_params"])

    # Set the parameters
    for param_name, value in params.items():
        if param_name in plugin.parameters:
            plugin.parameters[param_name].raw_value = value

    # Let the plugin process the parameter changes
    plugin.process([], 0.1, 44100, 2, 2048, True)
    plugin.reset()

    # Pick random MIDI note and velocity
    midi_note = random.randint(config["note_range"][0], config["note_range"][1])
    velocity = random.randint(config["velocity_range"][0], config["velocity_range"][1])

    # Create MIDI note (play for 80% of the duration)
    duration = config["duration_seconds"]
    note_end = duration * 0.8
    midi_events = make_midi_note(midi_note, velocity, 0.0, note_end)

    # Generate the audio
    audio = plugin.process(midi_events, duration, config["sample_rate"], 2, 2048, True)

    # Clean up
    plugin.process([], 0.1, 44100, 2, 2048, True)
    plugin.reset()

    return audio, params, midi_note, velocity


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
