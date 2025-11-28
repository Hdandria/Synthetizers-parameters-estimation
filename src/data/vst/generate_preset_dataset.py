import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import click
import h5py
import hdf5plugin
import librosa
import numpy as np
import rootutils
from loguru import logger
from pedalboard import VST3Plugin
from pyloudnorm import Meter
from tqdm import tqdm, trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, param_specs, render_params
from src.data.vst.core import _enforce_minimal_audible_params
from src.data.vst.param_spec import ParamSpec
from src.data.vst.vital_preset_converter import convert_vital_preset_to_params


@dataclass
class VSTDataSample:
    synth_params: dict[str, float]
    note_params: dict[str, float]
    sample_rate: float
    channels: int
    param_spec: ParamSpec
    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = None

    def __post_init__(self):
        self.param_array = self.param_spec.encode(self.synth_params, self.note_params)


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    window = "hamming"

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def perturb_params(
    params: dict[str, float], variance: float, param_spec: ParamSpec
) -> dict[str, float]:
    """Apply Gaussian noise to continuous parameters."""
    new_params = params.copy()
    
    # Identify continuous parameters from spec
    continuous_names = set()
    for p in param_spec.synth_params:
        if hasattr(p, "min") and hasattr(p, "max") and not hasattr(p, "values"):
             # It's a continuous parameter (roughly)
             continuous_names.add(p.name)

    for k, v in new_params.items():
        if k in continuous_names:
            # Add noise
            noise = random.gauss(0, variance)
            new_val = v + noise
            # Clamp
            new_val = max(0.0, min(1.0, new_val))
            new_params[k] = new_val
            
    return new_params


def generate_sample(
    plugin: VST3Plugin,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    base_presets: list[dict[str, float]],
    perturbation_variance: float,
) -> VSTDataSample:
    while True:
        # 1. Pick a random base preset
        base_params = random.choice(base_presets)
        
        # 2. Perturb it
        synth_params = perturb_params(base_params, perturbation_variance, param_spec)
        
        # 3. Sample note params (pitch, duration) randomly as before
        _, note_params = param_spec.sample()

        # 4. Render
        output = render_params(
            plugin,
            synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            preset_path=None, # Params already loaded
        )

        # 5. Check loudness
        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(output.T)
        if loudness < min_loudness:
            continue

        break

    spectrogram = make_spectrogram(output, sample_rate)

    return VSTDataSample(
        synth_params=synth_params,
        note_params=note_params,
        audio=output.T,
        mel_spec=spectrogram,
        sample_rate=sample_rate,
        channels=channels,
        param_spec=param_spec,
    )


def worker_generate_samples(
    worker_id: int,
    sample_indices: list[int],
    plugin_path: str,
    base_presets: list[dict[str, float]],
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    perturbation_variance: float,
    worker_output_path: str,
    progress_queue: multiprocessing.Queue,
) -> None:
    logger.info(f"Worker {worker_id} starting with {len(sample_indices)} samples")
    
    plugin = load_plugin(plugin_path)

    with h5py.File(worker_output_path, "w") as worker_file:
        num_samples = len(sample_indices)
        
        # Calculate mel frames
        n_samples = int(sample_rate * signal_duration_seconds)
        n_fft = int(0.025 * sample_rate)
        hop_length = int(sample_rate / 100.0)
        if n_samples > n_fft:
            mel_frames = 1 + (n_samples - n_fft) // hop_length
        else:
            mel_frames = 1

        audio_dataset = worker_file.create_dataset(
            "audio",
            shape=(num_samples, channels, n_samples),
            dtype=np.float16,
            compression=hdf5plugin.Blosc2(),
        )
        mel_dataset = worker_file.create_dataset(
            "mel_spec",
            shape=(num_samples, 2, 128, mel_frames),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )
        param_dataset = worker_file.create_dataset(
            "param_array",
            shape=(num_samples, len(param_spec)),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )

        for i, sample_idx in enumerate(sample_indices):
            sample = generate_sample(
                plugin,
                velocity,
                signal_duration_seconds,
                sample_rate,
                channels,
                min_loudness,
                param_spec,
                base_presets,
                perturbation_variance,
            )

            audio_dataset[i, :, :] = sample.audio.T
            mel_dataset[i, :, :] = sample.mel_spec
            param_dataset[i, :] = sample.param_array

            progress_queue.put(("generated", worker_id, sample_idx))

    progress_queue.put(("worker_done", worker_id, len(sample_indices)))
    logger.info(f"Worker {worker_id} finished")


import pickle

def load_all_presets(preset_dir: str, plugin_path: str, limit: int = None) -> list[dict[str, float]]:
    cache_path = Path(preset_dir) / "presets_cache.pkl"
    if cache_path.exists():
        logger.info(f"Loading presets from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            presets = pickle.load(f)
        logger.info(f"Loaded {len(presets)} presets from cache.")
        if limit:
            return presets[:limit]
        return presets

    logger.info(f"Loading all presets from {preset_dir}...")
    plugin = load_plugin(plugin_path)
    
    presets = []
    files = list(Path(preset_dir).glob("*.vital"))
    
    # Sort for determinism
    files.sort()

    if limit:
        files = files[:limit]
    
    for p in tqdm(files, desc="Loading presets"):
        try:
            params = convert_vital_preset_to_params(str(p), plugin)
            if params:
                params = _enforce_minimal_audible_params(plugin, params)
                presets.append(params)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            
    logger.info(f"Loaded {len(presets)} usable presets.")
    
    # Only cache if we loaded everything (no limit)
    if not limit:
        logger.info(f"Saving cache to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(presets, f)
            
    return presets


def merge_worker_files(worker_files: list[str], output_file: h5py.File) -> None:
    logger.info(f"Merging {len(worker_files)} worker files...")
    
    all_audio = []
    all_mel = []
    all_params = []

    for worker_file_path in worker_files:
        if not os.path.exists(worker_file_path):
            continue
            
        with h5py.File(worker_file_path, "r") as wf:
            all_audio.append(wf["audio"][:])
            all_mel.append(wf["mel_spec"][:])
            all_params.append(wf["param_array"][:])

    if all_audio:
        output_file["audio"][:] = np.concatenate(all_audio, axis=0)
        output_file["mel_spec"][:] = np.concatenate(all_mel, axis=0)
        output_file["param_array"][:] = np.concatenate(all_params, axis=0)


@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="vsts/Surge XT.vst3")
@click.option("--preset_dir", "-d", type=str, required=True, help="Directory containing .vital presets")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-t", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-55.0)
@click.option("--param_spec", "-k", type=str, default="vital")
@click.option("--perturbation_variance", "-z", type=float, default=0.1)
@click.option("--num_workers", "-w", type=int, default=1)
@click.option("--limit_presets", type=int, default=None, help="Limit number of presets to load (for testing)")
def main(
    data_file: str,
    num_samples: int,
    plugin_path: str,
    preset_dir: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: str,
    perturbation_variance: float,
    num_workers: int,
    limit_presets: int,
):
    spec = param_specs[param_spec]
    
    # 1. Load all presets into memory
    base_presets = load_all_presets(preset_dir, plugin_path, limit=limit_presets)
    if not base_presets:
        logger.error("No presets loaded! Exiting.")
        return

    # 2. Prepare Output File
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    with h5py.File(data_file, "a") as f:
        # Calculate dimensions
        n_samples = int(sample_rate * signal_duration_seconds)
        n_fft = int(0.025 * sample_rate)
        hop_length = int(sample_rate / 100.0)
        mel_frames = 1 + (n_samples - n_fft) // hop_length if n_samples > n_fft else 1
        
        # Create datasets
        if "audio" in f: del f["audio"]
        if "mel_spec" in f: del f["mel_spec"]
        if "param_array" in f: del f["param_array"]
        
        f.create_dataset("audio", (num_samples, channels, n_samples), dtype=np.float16, compression=hdf5plugin.Blosc2())
        f.create_dataset("mel_spec", (num_samples, 2, 128, mel_frames), dtype=np.float32, compression=hdf5plugin.Blosc2())
        f.create_dataset("param_array", (num_samples, len(spec)), dtype=np.float32, compression=hdf5plugin.Blosc2())

        # Attributes
        f["audio"].attrs.update({
            "velocity": velocity,
            "signal_duration_seconds": signal_duration_seconds,
            "sample_rate": sample_rate,
            "channels": channels,
            "min_loudness": min_loudness
        })

        # 3. Multiprocessing
        progress_queue = multiprocessing.Queue()
        sample_indices = list(range(num_samples))
        indices_per_worker = len(sample_indices) // num_workers
        
        worker_files = []
        processes = []
        
        for i in range(num_workers):
            start = i * indices_per_worker
            end = (i + 1) * indices_per_worker if i < num_workers - 1 else len(sample_indices)
            indices = sample_indices[start:end]
            
            if not indices: continue
            
            w_file = f"{data_file}.worker_{i}.h5"
            worker_files.append(w_file)
            
            p = multiprocessing.Process(
                target=worker_generate_samples,
                args=(
                    i, indices, plugin_path, base_presets, sample_rate, channels,
                    velocity, signal_duration_seconds, min_loudness, spec,
                    perturbation_variance, w_file, progress_queue
                )
            )
            p.start()
            processes.append(p)

        # Monitor
        generated = 0
        finished = 0
        with trange(num_samples) as pbar:
            while finished < len(processes):
                try:
                    while True:
                        msg = progress_queue.get_nowait()
                        if msg[0] == "generated":
                            generated += 1
                            pbar.update(1)
                        elif msg[0] == "worker_done":
                            finished += 1
                except:
                    pass
                
                import time
                time.sleep(0.1)

        for p in processes:
            p.join()

        # 4. Merge
        merge_worker_files(worker_files, f)
        
        # Cleanup
        for wf in worker_files:
            if os.path.exists(wf):
                os.remove(wf)

    logger.info("Done!")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
