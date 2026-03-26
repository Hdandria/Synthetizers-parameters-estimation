#!/usr/import/env python
import json
import logging
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import rootutils
import torch
import torchaudio
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NSYTH_TEST_URL = (
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
)


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Downloads and extracts the NSynth dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "nsynth-test.jsonwav.tar.gz"

    if not tar_path.exists():
        logger.info(f"Downloading {url} to {tar_path} ...")
        urllib.request.urlretrieve(url, tar_path)
    else:
        logger.info(f"Archive {tar_path} already exists. Skipping download.")

    extract_path = dest_dir / "nsynth-test"
    if not extract_path.exists():
        logger.info(f"Extracting {tar_path} to {dest_dir} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
    else:
        logger.info(f"Extracted folder {extract_path} already exists. Skipping extraction.")

    return extract_path


def create_nsynth_hdf5(
    extracted_dir: Path,
    output_h5: Path,
    sample_rate: int = 44100,
    duration_sec: float = 4.0,
    dummy_param_dim: int = 597,
):
    """Parses NSynth waveforms, resamples, pads to exactly length, and computes mel specs."""
    examples_file = extracted_dir / "examples.json"
    audio_dir = extracted_dir / "audio"

    logger.info("Loading NSynth metadata ...")
    with open(examples_file, "r") as f:
        metadata = json.load(f)

    # We only care about acoustic and electronic instruments (excluding vocals or highly synthetic unless desired)
    # NSynth test is only ~4096 samples total, we process a small subset for quick OOD evaluation
    keys = list(metadata.keys())[:100]
    num_samples = len(keys)
    logger.info(f"Using {num_samples} samples for rapid test split.")

    target_length = int(sample_rate * duration_sec)

    # Calculate mel frames as done in generate_preset_dataset.py
    n_samples = int(sample_rate * duration_sec)
    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    mel_frames = 1 + (n_samples - n_fft) // hop_length if n_samples > n_fft else 1

    logger.info(f"Creating HDF5 file: {output_h5}")
    with h5py.File(output_h5, "w") as f:
        ds_audio = f.create_dataset(
            "audio", (num_samples, 2, target_length), dtype=np.float32, **hdf5plugin.Blosc2()
        )
        ds_mel = f.create_dataset(
            "mel_spec", (num_samples, 2, 128, mel_frames), dtype=np.float32, **hdf5plugin.Blosc2()
        )
        # Dummy labels for Param Estimation
        ds_params = f.create_dataset(
            "param_array", (num_samples, dummy_param_dim), dtype=np.float32, **hdf5plugin.Blosc2()
        )

        for i, key in tqdm(enumerate(keys), total=num_samples, desc="Processing Audio"):
            audio_path = audio_dir / f"{key}.wav"

            import librosa
            import torch

            # Load and resample audio using librosa (returns mono or stereo as numpy array)
            # librosa.load normally returns shape (channels, time) if mono=False,
            # or just (time,) if mono audio.
            np_audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)

            # Ensure shape is [channels, time]
            if np_audio.ndim == 1:
                np_audio = np.expand_dims(np_audio, axis=0)

            # Convert Mono to Stereo by duplicating
            if np_audio.shape[0] == 1:
                np_audio = np.repeat(np_audio, 2, axis=0)

            # Pad / Trim to exactly target_length
            if np_audio.shape[1] > target_length:
                np_audio = np_audio[:, :target_length]
            elif np_audio.shape[1] < target_length:
                padding = target_length - np_audio.shape[1]
                np_audio = np.pad(np_audio, ((0, 0), (0, padding)), mode="constant")

            import librosa

            # Compute Mel via librosa matching generate_preset_dataset.py
            # torchaudio gives shape [channels, time]; librosa expects numpy [time] or [channels, time]
            mels = []
            for ch in range(np_audio.shape[0]):
                spec = librosa.feature.melspectrogram(
                    y=np_audio[ch],
                    sr=sample_rate,
                    n_mels=128,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window="hamming",
                    center=False,
                )
                spec_db = librosa.power_to_db(spec, ref=np.max)
                mels.append(spec_db)

            mel = np.stack(mels, axis=0)  # shape (2, 128, mel_frames)

            # Convert to numpy and save
            ds_audio[i] = np_audio
            ds_mel[i] = mel
            ds_params[i] = np.zeros(dummy_param_dim, dtype=np.float32)


def main():
    root_dir = Path("datasets/nsynth_eval")
    root_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = root_dir / "raw"

    extract_dir = download_and_extract(NSYTH_TEST_URL, temp_dir)

    output_h5 = root_dir / "test.h5"
    if not output_h5.exists():
        create_nsynth_hdf5(extract_dir, output_h5)
    else:
        logger.info(f"{output_h5} already exists. Skipping HDF5 generation.")

    stats_source = Path("datasets/vital_30k_eval/stats.npz")
    stats_dest = root_dir / "stats.npz"
    if stats_source.exists():
        logger.info(f"Copying {stats_source} to {stats_dest} for normalization...")
        shutil.copy(stats_source, stats_dest)
    else:
        logger.warning(
            f"Could not find {stats_source}. You MUST supply a stats.npz for the dataset to load correctly."
        )

    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()
