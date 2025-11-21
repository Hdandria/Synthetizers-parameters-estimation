import multiprocessing
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
from src.data.vst import load_plugin, param_specs, render_params  # noqa
from src.data.vst.param_spec import ParamSpec  # noqa


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
    """Values hardcoded to be roughly like those used by the audio spectrogram transformer.

    i.e. 100 frames per second, 128 mels, ~25ms window, hamming window.
    """

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
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def generate_sample(
    plugin: VST3Plugin,
    velocity: int,
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    min_loudness: float,
    param_spec: ParamSpec,
    preset_path: str | None,
) -> VSTDataSample:
    while True:
        logger.debug("sampling params")
        synth_params, note_params = param_spec.sample()

        logger.debug("sampling note")

        output = render_params(
            plugin,
            synth_params,
            note_params["pitch"],
            velocity,
            note_params["note_start_and_end"],
            signal_duration_seconds,
            sample_rate,
            channels,
            preset_path=preset_path,
        )

        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(output.T)
        logger.debug(f"loudness: {loudness}")
        if loudness < min_loudness:
            logger.debug("loudness too low, skipping")
            continue

        break

    logger.debug("making spectrogram")
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


def save_sample(
    sample: VSTDataSample,
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    idx: int,
) -> None:
    logger.info(f"Saving sample {idx}...")
    audio_dataset[idx, :, :] = sample.audio.T
    mel_dataset[idx, :, :] = sample.mel_spec
    param_dataset[idx, :] = sample.param_array
    logger.info(f"Sample {idx} written!")


def save_samples(
    samples: list[VSTDataSample],
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    start_idx: int,
) -> None:
    logger.info(f"Saving {len(samples)} samples...")
    audios = np.stack([s.audio.T for s in samples], axis=0)
    mel_specs = np.stack([s.mel_spec for s in samples], axis=0)
    param_arrays = np.stack([s.param_array for s in samples], axis=0)

    audio_dataset[start_idx : start_idx + len(samples), :, :] = audios
    mel_dataset[start_idx : start_idx + len(samples), :, :] = mel_specs
    param_dataset[start_idx : start_idx + len(samples), :] = param_arrays

    logger.info(f"{len(samples)} samples written!")


def get_first_unwritten_idx(dataset: h5py.Dataset) -> int:
    num_rows, *_ = dataset.shape
    for i in range(num_rows):
        row = dataset[num_rows - i - 1]
        if not np.all(row == 0):
            return num_rows - i
        logger.debug(f"Row {num_rows - i - 1} is empty...")

    return 0


def create_dataset_and_get_first_unwritten_idx(
    h5py_file: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    compression: Any,
) -> tuple[h5py.Dataset, int]:
    logger.info(f"Looking for dataset {name}...")
    if name in h5py_file:
        logger.info(f"Found dataset {name}, checking shape...")
        dataset = h5py_file[name]
        if dataset.shape != shape:
            logger.warning(f"Dataset {name} has shape {dataset.shape}, expected {shape}. Recreating dataset.")
            # remove and recreate with the expected shape
            del h5py_file[name]
            dataset = h5py_file.create_dataset(name, shape=shape, dtype=dtype, compression=compression)
            return dataset, 0
        logger.info(f"Dataset {name} shape OK, looking for first unwritten row.")
        return dataset, get_first_unwritten_idx(dataset)

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
    # compute integer-shaped dataset dimensions
    n_samples = int(sample_rate * signal_duration_seconds)
    # mel spectrogram framing should match make_spectrogram()
    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    if n_samples > n_fft:
        mel_frames = 1 + (n_samples - n_fft) // hop_length
    else:
        mel_frames = 1

    audio_dataset, audio_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "audio",
        (num_samples, channels, n_samples),
        dtype=np.float16,
        compression=hdf5plugin.Blosc2(),
    )
    mel_dataset, mel_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "mel_spec",
        (num_samples, 2, 128, mel_frames),
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )
    param_dataset, param_start_idx = create_dataset_and_get_first_unwritten_idx(
        hdf5_file,
        "param_array",
        (num_samples, num_params),  # +1 for MIDI note
        dtype=np.float32,
        compression=hdf5plugin.Blosc2(),
    )

    return (
        audio_dataset,
        mel_dataset,
        param_dataset,
        min(audio_start_idx, mel_start_idx, param_start_idx),
    )


def worker_generate_samples(
    worker_id: int,
    sample_indices: list[int],
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    worker_output_path: str,
    progress_queue: multiprocessing.Queue,
) -> None:
    """Worker function that generates samples in parallel and writes to its own file."""
    logger.info(f"Worker {worker_id} starting with {len(sample_indices)} samples")

    # Each worker loads its own plugin instance
    plugin = load_plugin(plugin_path)

    # Create worker's own HDF5 file
    logger.info(f"Worker {worker_id} creating file: {worker_output_path}")
    with h5py.File(worker_output_path, "w") as worker_file:
        # Create datasets in worker file
        num_samples = len(sample_indices)
        audio_dataset = worker_file.create_dataset(
            "audio",
            shape=(num_samples, channels, int(sample_rate * signal_duration_seconds)),
            dtype=np.float16,
            compression=hdf5plugin.Blosc2(),
        )
        mel_dataset = worker_file.create_dataset(
            "mel_spec",
            shape=(num_samples, 2, 128, (lambda: (1 + (int(sample_rate * signal_duration_seconds) - int(0.025 * sample_rate)) // int(sample_rate / 100.0)) if int(sample_rate * signal_duration_seconds) > int(0.025 * sample_rate) else 1)()),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )
        param_dataset = worker_file.create_dataset(
            "param_array",
            shape=(num_samples, len(param_spec)),
            dtype=np.float32,
            compression=hdf5plugin.Blosc2(),
        )

        # Set attributes
        audio_dataset.attrs["velocity"] = velocity
        audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
        audio_dataset.attrs["sample_rate"] = sample_rate
        audio_dataset.attrs["channels"] = channels
        audio_dataset.attrs["min_loudness"] = min_loudness

        # Generate and write samples directly
        for i, sample_idx in enumerate(sample_indices):
            logger.info(
                f"Worker {worker_id} making sample {sample_idx} ({i+1}/{len(sample_indices)})"
            )
            sample = generate_sample(
                plugin,
                velocity=velocity,
                signal_duration_seconds=signal_duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                min_loudness=min_loudness,
                param_spec=param_spec,
                preset_path=None,  # preset already loaded once below
            )

            # Write directly to worker's file
            audio_dataset[i, :, :] = sample.audio.T
            mel_dataset[i, :, :] = sample.mel_spec
            param_dataset[i, :] = sample.param_array

            # Send progress update
            progress_queue.put(("generated", worker_id, sample_idx))

    # Signal worker completion
    progress_queue.put(("worker_done", worker_id, len(sample_indices)))
    logger.info(f"Worker {worker_id} finished and wrote to {worker_output_path}")


def merge_worker_files(worker_files: list[str], output_file: h5py.File) -> None:
    """Merge multiple worker HDF5 files into the main output file."""
    logger.info(f"Merging {len(worker_files)} worker files into main dataset")

    all_audio = []
    all_mel = []
    all_params = []

    for worker_file_path in worker_files:
        logger.info(f"Reading worker file: {worker_file_path}")
        import os

        if not os.path.exists(worker_file_path):
            logger.error(f"Worker file does not exist: {worker_file_path}")
            continue

        with h5py.File(worker_file_path, "r") as worker_file:
            logger.info(f"Worker file contains {worker_file['audio'].shape[0]} samples")
            all_audio.append(worker_file["audio"][:])
            all_mel.append(worker_file["mel_spec"][:])
            all_params.append(worker_file["param_array"][:])

    # Concatenate and write to main file
    if all_audio:
        total_samples = sum(audio.shape[0] for audio in all_audio)
        logger.info(f"Writing {total_samples} total samples to main file")
        output_file["audio"][:] = np.concatenate(all_audio, axis=0)
        output_file["mel_spec"][:] = np.concatenate(all_mel, axis=0)
        output_file["param_array"][:] = np.concatenate(all_params, axis=0)
    else:
        logger.error("No worker files found to merge!")

    logger.info("Worker files merged successfully")


def make_dataset(
    hdf5_file: h5py.File,
    num_samples: int,
    plugin_path: str,
    preset_path: str,
    sample_rate: float,
    channels: int,
    velocity: int,
    signal_duration_seconds: float,
    min_loudness: float,
    param_spec: ParamSpec,
    sample_batch_size: int,
    num_workers: int = 1,
) -> None:
    audio_dataset, mel_dataset, param_dataset, start_idx = create_datasets_and_get_start_idx(
        hdf5_file=hdf5_file,
        num_samples=num_samples,
        channels=channels,
        sample_rate=sample_rate,
        signal_duration_seconds=signal_duration_seconds,
        num_params=len(param_spec),
    )

    audio_dataset.attrs["velocity"] = velocity
    audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
    audio_dataset.attrs["sample_rate"] = sample_rate
    audio_dataset.attrs["channels"] = channels
    audio_dataset.attrs["min_loudness"] = min_loudness

    if num_workers == 1:
        # Single-threaded fallback (original behavior)
        plugin = load_plugin(plugin_path)
        # Load preset once per dataset build for efficiency
        if preset_path:
            try:
                from src.data.vst.core import load_preset
                load_preset(plugin, preset_path)
            except Exception as e:
                logger.warning(f"Failed to pre-load preset '{preset_path}': {e}")
        sample_batch = []
        sample_batch_start = start_idx

        for i in trange(start_idx, num_samples):
            logger.info(f"Making sample {i}")
            sample = generate_sample(
                plugin,
                velocity=velocity,
                signal_duration_seconds=signal_duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                min_loudness=min_loudness,
                param_spec=param_spec,
                preset_path=preset_path,
            )

            sample_batch.append(sample)
            if len(sample_batch) == sample_batch_size:
                save_samples(
                    sample_batch,
                    audio_dataset,
                    mel_dataset,
                    param_dataset,
                    sample_batch_start,
                )
                sample_batch = []
                sample_batch_start += sample_batch_size

        if len(sample_batch) > 0:
            save_samples(
                sample_batch,
                audio_dataset,
                mel_dataset,
                param_dataset,
                sample_batch_start,
            )
    else:
        # Multiprocessed generation with separate files per worker
        logger.info(f"Starting multiprocessed generation with {num_workers} workers")
        logger.info(f"Main output file: {hdf5_file.filename}")

        # Create progress queue
        progress_queue = multiprocessing.Queue()

        # Distribute sample indices among workers
        sample_indices = list(range(start_idx, num_samples))
        indices_per_worker = len(sample_indices) // num_workers
        worker_tasks = []
        worker_files = []

        for i in range(num_workers):
            start_idx_worker = i * indices_per_worker
            if i == num_workers - 1:  # Last worker gets remaining samples
                end_idx_worker = len(sample_indices)
            else:
                end_idx_worker = (i + 1) * indices_per_worker

            worker_indices = sample_indices[start_idx_worker:end_idx_worker]
            if worker_indices:  # Only create worker if there are samples to process
                worker_tasks.append(worker_indices)
                # Create unique filename for each worker in the same directory
                import os

                main_file_dir = os.path.dirname(hdf5_file.filename)
                main_file_name = os.path.basename(hdf5_file.filename)
                worker_file_path = os.path.join(main_file_dir, f"worker_{i}_{main_file_name}")
                worker_files.append(worker_file_path)
                logger.info(f"Worker {i} will write to: {worker_file_path}")

        # Start worker processes
        processes = []
        for i, worker_indices in enumerate(worker_tasks):
            p = multiprocessing.Process(
                target=worker_generate_samples,
                args=(
                    i,
                    worker_indices,
                    plugin_path,
                    preset_path,
                    sample_rate,
                    channels,
                    velocity,
                    signal_duration_seconds,
                    min_loudness,
                    param_spec,
                    worker_files[i],
                    progress_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Monitor progress
        total_samples = len(sample_indices)
        samples_generated = 0
        workers_finished = 0

        with trange(total_samples, desc="Generating samples") as pbar:
            while workers_finished < len(processes):
                # Check for progress updates
                try:
                    while True:
                        progress_msg = progress_queue.get_nowait()
                        if progress_msg[0] == "generated":
                            samples_generated += 1
                            pbar.update(1)
                        elif progress_msg[0] == "worker_done":
                            workers_finished += 1
                except Exception:
                    pass  # No more progress messages

                # Update progress bar
                pbar.set_postfix(
                    {
                        "generated": samples_generated,
                        "workers_done": f"{workers_finished}/{len(processes)}",
                    }
                )

                # Small sleep to avoid busy waiting
                import time

                time.sleep(0.1)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        logger.info("All workers finished, merging files...")

        # Merge worker files into main file
        merge_worker_files(worker_files, hdf5_file)

        # Clean up worker files
        for worker_file in worker_files:
            try:
                import os

                os.remove(worker_file)
                logger.info(f"Cleaned up worker file: {worker_file}")
            except Exception as e:
                logger.warning(f"Could not clean up worker file {worker_file}: {e}")

        logger.info("Multiprocessed generation completed")


@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="vsts/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-55.0)
@click.option("--param_spec", "-t", type=str, default="surge_xt")
@click.option("--sample_batch_size", "-b", type=int, default=32)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=1,
    help="Number of worker processes for parallel generation",
)
def main(
    data_file: str,
    num_samples: int,
    plugin_path: str = "vsts/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -50.0,
    param_spec: str = "surge_xt",
    sample_batch_size: int = 32,
    num_workers: int = 1,
):
    param_spec = param_specs[param_spec]
    with h5py.File(data_file, "a") as f:
        make_dataset(
            f,
            num_samples,
            plugin_path,
            preset_path,
            sample_rate,
            channels,
            velocity,
            signal_duration_seconds,
            min_loudness,
            param_spec,
            sample_batch_size,
            num_workers,
        )


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    multiprocessing.set_start_method("spawn", force=True)
    main()
