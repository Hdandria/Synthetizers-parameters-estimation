from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from pedalboard.io import AudioFile


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Values hardcoded to be roughly like those used by the audio spectrogram
    transformer. i.e. 100 frames per second, 128 mels, ~25ms window, hamming
    window."""

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


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, segment_length_seconds: float = 4.0):
        self.segment_length_seconds = segment_length_seconds

        self.root = Path(root)
        self.files = list(self.root.glob("*.wav"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]

        with AudioFile(str(file), "r") as f:
            sample_rate = f.sample_rate
            num_frames = int(sample_rate * self.segment_length_seconds)
            audio = f.read(num_frames)

        channels, _ = audio.shape
        if channels == 1:
            audio = np.stack([audio, audio], axis=0)
        elif channels > 2:
            raise ValueError(
                f"Audio must have two or fewer channels. Found {channels}."
            )

        spec = make_spectrogram(audio, sample_rate)

        return {
            "audio": audio,
            "mel_spec": spec,
        }


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        segment_length_seconds: float = 4.0,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True
    ):
        super().__init__()

        self.root = root
        self.segment_length_seconds = segment_length_seconds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, stage: Optional[str] = None):
        self.predict_dataset = AudioFolderDataset(
            self.root, self.segment_length_seconds
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
