#!/usr/bin/env python3
"""
Scalable VST Dataset Generator

This script creates audio datasets from VST plugins using a chunked approach
for generating tens or hundreds of thousands of samples efficiently.

Usage:
    python generate_vst_dataset.py --config dataset_config.yaml
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import random
import time
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
        note_timing = note_params["note_start_and_end"]
        if isinstance(note_timing, (list, tuple)) and len(note_timing) == 2:
            note_start, note_end = note_timing
        else:
            # Fallback if note_timing is not a tuple/list
            note_start, note_end = 0.0, config["duration_seconds"]

        # Set the synth parameters (skip note parameters)
        for param_name, value in all_params.items():
            if hasattr(plugin, "parameters") and param_name in plugin.parameters:  # type: ignore
                plugin.parameters[param_name].raw_value = value  # type: ignore

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


def save_chunk(samples: List[Tuple], chunk_id: int, config: Dict):
    """Save a chunk of samples to an HDF5 file"""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use configurable file naming pattern
    file_pattern = config.get("file_naming", "dataset_{chunk:04d}.h5")
    filename = output_dir / file_pattern.format(chunk=chunk_id)

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

        # Add metadata
        f.attrs["chunk_id"] = chunk_id
        f.attrs["num_samples"] = n_samples
        f.attrs["sample_rate"] = config["sample_rate"]
        f.attrs["duration_seconds"] = config["duration_seconds"]
        f.attrs["param_set"] = config["param_set"]

    return filename


def generate_single_chunk_worker(args):
    """Worker function for generating a single chunk in a separate process"""
    chunk_id, chunk_size, config_path, plugin_path, seed_offset = args

    # Load config in worker process
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set unique seed for this worker
    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)

    # Load plugin in worker process
    plugin = load_plugin(plugin_path)

    # Generate samples for this chunk
    samples = []
    for i in range(chunk_size):
        try:
            audio, params, midi_note, velocity = generate_audio_sample(plugin, config)
            samples.append((audio, params, midi_note, velocity))
        except Exception as e:
            print(f"Worker {chunk_id}: Failed to generate sample {i}: {e}")
            continue

    # Save chunk
    if samples:
        filename = save_chunk(samples, chunk_id, config)
        return chunk_id, len(samples), filename, None
    else:
        return chunk_id, 0, None, "No samples generated"


def generate_chunked_dataset(config: Dict):
    """Generate dataset using chunked approach with parallel processing"""
    total_samples = config["num_samples"]
    samples_per_file = config.get("samples_per_file", 1000)
    num_workers = config.get("num_workers", min(4, mp.cpu_count()))

    # Calculate number of chunks
    total_chunks = (total_samples + samples_per_file - 1) // samples_per_file

    print(f"Generating {total_samples} samples in {total_chunks} chunks")
    print(f"Chunk size: {samples_per_file} samples")
    print(f"Using {num_workers} workers")

    # Prepare arguments for workers
    worker_args = []
    for chunk_id in range(total_chunks):
        chunk_size = min(samples_per_file, total_samples - chunk_id * samples_per_file)
        seed_offset = chunk_id * 1000  # Ensure different seeds for each chunk
        worker_args.append((chunk_id, chunk_size, config.get("_config_path"), config["plugin_path"], seed_offset))

    # Generate chunks in parallel
    completed_chunks = 0
    failed_chunks = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all chunk generation tasks
        future_to_chunk = {executor.submit(generate_single_chunk_worker, args): args[0] for args in worker_args}

        # Process completed chunks with progress bar
        with tqdm(total=total_chunks, desc="Generating chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_id, num_samples, filename, error = future.result()

                if error:
                    print(f"Chunk {chunk_id} failed: {error}")
                    failed_chunks += 1
                else:
                    print(f"Chunk {chunk_id}: {num_samples} samples saved to {filename}")
                    completed_chunks += 1

                pbar.update(1)

    print("\nGeneration complete!")
    print(f"Successfully generated: {completed_chunks} chunks")
    print(f"Failed chunks: {failed_chunks}")
    print(f"Total samples: {completed_chunks * samples_per_file}")


def save_dataset(samples: List[Tuple], config: Dict):
    """Legacy function - save all samples to a single HDF5 file"""
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

    print("Dataset saved successfully!")


def main():
    """Main function - does the work"""
    parser = argparse.ArgumentParser(description="Generate VST dataset")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config["_config_path"] = args.config  # Pass config path to workers

    # Show parameter selection info
    print(f"Parameter set: {config['param_set']}")
    param_spec = get_param_spec(config["param_set"])
    print(f"Synth parameters: {len(param_spec.synth_params)}")

    # Choose generation method based on config
    use_chunked = config.get("force_chunked", False) or config.get("num_samples", 0) > 1000

    if use_chunked:
        print("Using chunked generation (recommended for large datasets)")
        generate_chunked_dataset(config)
    else:
        print("Using legacy single-file generation")

        start_time = time.time()

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

        print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
