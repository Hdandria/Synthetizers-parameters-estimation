import multiprocessing
import os
import random
from dataclasses import dataclass
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
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, param_specs, render_params
from src.data.vst.param_spec import ParamSpec, ContinuousParameter, CategoricalParameter, DiscreteLiteralParameter, NoteDurationParameter
from src.data.vst.core import load_preset

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

def get_params_from_plugin(plugin: VST3Plugin, param_spec: ParamSpec) -> dict[str, float]:
    """Extract current parameter values from the plugin."""
    params = {}
    plugin_params = getattr(plugin, "parameters", {})
    
    for param in param_spec.synth_params:
        if param.name in plugin_params:
            # We prefer raw_value as it is normalized [0, 1]
            try:
                params[param.name] = float(plugin_params[param.name].raw_value)
            except Exception:
                # Fallback if raw_value is not available (unlikely for VST3)
                params[param.name] = 0.0
        else:
            # If parameter is missing in plugin, use default/min
            if isinstance(param, ContinuousParameter):
                params[param.name] = param.min
            elif isinstance(param, CategoricalParameter):
                params[param.name] = param.raw_values[0]
            elif isinstance(param, DiscreteLiteralParameter):
                 params[param.name] = param.min
            else:
                params[param.name] = 0.0
                
    return params

def perturb_params(
    params: dict[str, float], 
    param_spec: ParamSpec, 
    noise_level: float = 0.1
) -> dict[str, float]:
    """Perturb continuous parameters with Gaussian noise."""
    new_params = params.copy()
    
    for param in param_spec.synth_params:
        if isinstance(param, ContinuousParameter):
            val = new_params.get(param.name, 0.0)
            # Add noise
            noise = random.gauss(0, noise_level)
            new_val = val + noise
            # Clamp to [0, 1]
            new_val = max(0.0, min(1.0, new_val))
            new_params[param.name] = new_val
            
    return new_params

def generate_sample_near_preset(
    plugin: VST3Plugin,
    preset_path: str,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    noise_level: float,
) -> VSTDataSample:
    
    # 1. Load Preset
    load_preset(plugin, preset_path)
    
    # 2. Extract Parameters
    base_params = get_params_from_plugin(plugin, param_spec)
    
    # 3. Sample Note Parameters (randomly, as they are not part of the preset usually)
    _, note_params = param_spec.sample()
    
    while True:
        # 4. Perturb Parameters
        synth_params = perturb_params(base_params, param_spec, noise_level)
        
        # 5. Render
        output = render_params(
            plugin,
            synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            preset_path=None, # Already loaded and modified
        )

        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(output.T)
        
        if loudness < min_loudness:
            # If too quiet, maybe try again with different perturbation or just skip?
            # For now, let's just retry perturbation
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

def save_samples(
    samples: list[VSTDataSample],
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    start_idx: int,
) -> None:
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    audio_dataset[start_idx : start_idx + len(samples), :, :] = audios
    mel_dataset[start_idx : start_idx + len(samples), :, :] = mel_specs
    param_dataset[start_idx : start_idx + len(samples), :] = param_arrays

def create_dataset_and_get_first_unwritten_idx(
    h5py_file: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    compression: Any,
) -> tuple[h5py.Dataset, int]:
    if name in h5py_file:
        dataset = h5py_file[name]
        if dataset.shape != shape:
            del h5py_file[name]
            dataset = h5py_file.create_dataset(name, shape=shape, dtype=dtype, compression=compression)
            return dataset, 0
        
        # Simple check for first unwritten
        # Assuming sequential writing from 0
        # We can check the last written index by looking for non-zeros
        # But for speed, let's just assume we append if we want to restart? 
        # Or better, just scan from end like before.
        num_rows = dataset.shape[0]
        for i in range(num_rows):
            if np.all(dataset[num_rows - 1 - i] == 0):
                continue
            return dataset, num_rows - i
        return dataset, 0

    dataset = h5py_file.create_dataset(name, shape=shape, dtype=dtype, compression=compression)
    return dataset, 0

def create_datasets_and_get_start_idx(
    hdf5_file: h5py.File,
    num_samples: int,
    channels: int,
    sample_rate: float,
    signal_duration_seconds: float,
    num_params: int,
):
    n_samples = int(sample_rate * signal_duration_seconds)
    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    if n_samples > n_fft:
        mel_frames = 1 + (n_samples - n_fft) // hop_length
    else:
        mel_frames = 1

    audio_dataset, audio_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, "audio", (num_samples, channels, n_samples), np.float16, hdf5plugin.Blosc2()
    )
    mel_dataset, mel_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, "mel_spec", (num_samples, 2, 128, mel_frames), np.float32, hdf5plugin.Blosc2()
    )
    param_dataset, param_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file, "param_array", (num_samples, num_params), np.float32, hdf5plugin.Blosc2()
    )

    return audio_dataset, mel_dataset, param_dataset, min(audio_start_idx, mel_start_idx, param_start_idx)

def worker_generate_samples(
    worker_id: int,
    sample_indices: list[int],
    plugin_path: str,
    preset_paths: list[str],
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    noise_level: float,
    worker_output_path: str,
    progress_queue: multiprocessing.Queue,
) -> None:
    plugin = load_plugin(plugin_path)
    
    with h5py.File(worker_output_path, "w") as worker_file:
        num_samples = len(sample_indices)
        # Create datasets (same logic as main)
        # ... (simplified for brevity, assuming helper functions available or copied)
        # For now, let's just copy the creation logic or import it if we refactor.
        # Since I can't easily refactor shared code in one go, I'll duplicate the creation logic here for the worker.
        
        n_samples = int(sample_rate * signal_duration_seconds)
        n_fft = int(0.025 * sample_rate)
        hop_length = int(sample_rate / 100.0)
        mel_frames = 1 + (n_samples - n_fft) // hop_length if n_samples > n_fft else 1

        audio_dataset = worker_file.create_dataset("audio", (num_samples, channels, n_samples), dtype=np.float16, compression=hdf5plugin.Blosc2())
        mel_dataset = worker_file.create_dataset("mel_spec", (num_samples, 2, 128, mel_frames), dtype=np.float32, compression=hdf5plugin.Blosc2())
        param_dataset = worker_file.create_dataset("param_array", (num_samples, len(param_spec)), dtype=np.float32, compression=hdf5plugin.Blosc2())

        audio_dataset.attrs["velocity"] = velocity
        audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
        audio_dataset.attrs["sample_rate"] = sample_rate
        audio_dataset.attrs["channels"] = channels
        audio_dataset.attrs["min_loudness"] = min_loudness

        for i, sample_idx in enumerate(sample_indices):
            # Pick random preset
            preset_path = random.choice(preset_paths)
            
            sample = generate_sample_near_preset(
                plugin,
                preset_path,
                velocity,
                signal_duration_seconds,
                sample_rate,
                channels,
                min_loudness,
                param_spec,
                noise_level,
            )

            audio_dataset[i, :, :] = sample.audio.T
            mel_dataset[i, :, :] = sample.mel_spec
            param_dataset[i, :] = sample.param_array

            progress_queue.put(("generated", worker_id, sample_idx))

    progress_queue.put(("worker_done", worker_id, len(sample_indices)))

def merge_worker_files(worker_files: list[str], output_file: h5py.File) -> None:
    # Same as original
    all_audio = []
    all_mel = []
    all_params = []

    for worker_file_path in worker_files:
        if not os.path.exists(worker_file_path): continue
        with h5py.File(worker_file_path, "r") as worker_file:
            all_audio.append(worker_file["audio"][:])
            all_mel.append(worker_file["mel_spec"][:])
            all_params.append(worker_file["param_array"][:])

    if all_audio:
        output_file["audio"][:] = np.concatenate(all_audio, axis=0)
        output_file["mel_spec"][:] = np.concatenate(all_mel, axis=0)
        output_file["param_array"][:] = np.concatenate(all_params, axis=0)

@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="vsts/Surge XT.vst3")
@click.option("--preset_dir", "-D", type=str, required=True, help="Directory containing presets")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-55.0)
@click.option("--param_spec", "-t", type=str, default="surge_xt")
@click.option("--noise_level", "-n", type=float, default=0.1, help="Standard deviation of Gaussian noise for parameters")
@click.option("--num_workers", "-w", type=int, default=1)
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
    noise_level: float,
    num_workers: int,
):
    spec = param_specs[param_spec]
    
    # Collect presets
    presets = []
    plugin_name = os.path.basename(plugin_path).lower()
    
    for root, _, files in os.walk(preset_dir):
        for file in files:
            if file.endswith(".vstpreset") or file.endswith(".vital"):
                # Simple heuristic filtering
                if "vital" in plugin_name and "vital" not in file.lower() and not file.endswith(".vital"):
                    continue
                if "surge" in plugin_name and "surge" not in file.lower():
                    continue
                    
                presets.append(os.path.join(root, file))
    
    if not presets:
        logger.error(f"No presets found in {preset_dir}")
        return

    logger.info(f"Found {len(presets)} presets")

    with h5py.File(data_file, "a") as f:
        audio_dataset, mel_dataset, param_dataset, start_idx = create_datasets_and_get_start_idx(
            f, num_samples, channels, sample_rate, signal_duration_seconds, len(spec)
        )
        
        audio_dataset.attrs["velocity"] = velocity
        audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
        audio_dataset.attrs["sample_rate"] = sample_rate
        audio_dataset.attrs["channels"] = channels
        audio_dataset.attrs["min_loudness"] = min_loudness

        if num_workers == 1:
            plugin = load_plugin(plugin_path)
            sample_batch = []
            sample_batch_start = start_idx
            
            for i in trange(start_idx, num_samples):
                preset_path = random.choice(presets)
                sample = generate_sample_near_preset(
                    plugin, preset_path, velocity, signal_duration_seconds, 
                    sample_rate, channels, min_loudness, spec, noise_level
                )
                
                sample_batch.append(sample)
                if len(sample_batch) == 32: # Batch size hardcoded for now
                    save_samples(sample_batch, audio_dataset, mel_dataset, param_dataset, sample_batch_start)
                    sample_batch = []
                    sample_batch_start += 32
            
            if sample_batch:
                save_samples(sample_batch, audio_dataset, mel_dataset, param_dataset, sample_batch_start)
        
        else:
            # Multiprocessing
            progress_queue = multiprocessing.Queue()
            sample_indices = list(range(start_idx, num_samples))
            indices_per_worker = len(sample_indices) // num_workers
            worker_files = []
            processes = []
            
            for i in range(num_workers):
                start = i * indices_per_worker
                end = (i + 1) * indices_per_worker if i < num_workers - 1 else len(sample_indices)
                w_indices = sample_indices[start:end]
                
                if not w_indices: continue
                
                w_file = f"{data_file}.worker_{i}"
                worker_files.append(w_file)
                
                p = multiprocessing.Process(
                    target=worker_generate_samples,
                    args=(i, w_indices, plugin_path, presets, sample_rate, channels, 
                          velocity, signal_duration_seconds, min_loudness, spec, 
                          noise_level, w_file, progress_queue)
                )
                p.start()
                processes.append(p)
                
            # Monitor (simplified)
            finished = 0
            with trange(len(sample_indices)) as pbar:
                while finished < len(processes):
                    try:
                        msg = progress_queue.get(timeout=1)
                        if msg[0] == "generated":
                            pbar.update(1)
                        elif msg[0] == "worker_done":
                            finished += 1
                    except:
                        pass
            
            for p in processes: p.join()
            
            merge_worker_files(worker_files, f)
            for wf in worker_files: os.remove(wf)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
