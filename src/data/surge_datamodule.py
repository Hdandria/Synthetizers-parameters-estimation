from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from tqdm import trange

from src.data.ot import ot_collate_fn, regular_collate_fn


class SurgeXTDataset(torch.utils.data.Dataset):
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def __init__(
        self,
        dataset_file: Union[str, Path],
        read_audio: bool = False,
        use_saved_mean_and_variance: bool = True,
        rescale_params: bool = True,
    ):
        self.dataset_file = h5py.File(dataset_file, "r")
        self.read_audio = read_audio
        self.rescale_params = rescale_params

        if use_saved_mean_and_variance:
            self._load_dataset_statistics(dataset_file)

    def _load_dataset_statistics(self, dataset_file: Union[str, Path]):
        # for /path/to/train.h5 we would expect to find /path/to/stats.npz
        # if not, we throw an error
        stats_file = SurgeXTDataset.get_stats_file_path(dataset_file)
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Could not find statistics file {stats_file}. \n"
                "Make sure to first run `scripts/get_dataset_stats.py`."
            )

        with np.load(stats_file) as stats:
            self.mean = stats["mean"]
            self.std = stats["std"]

    @staticmethod
    def get_stats_file_path(dataset_file: Union[str, Path]) -> Path:
        dataset_file = Path(dataset_file)
        data_dir = dataset_file.parent
        return data_dir / "stats.npz"

    def __len__(self):
        return self.dataset_file["audio"].shape[0]

    def __getitem__(self, idx):
        if self.read_audio:
            audio = self.dataset_file["audio"][idx, :, :]
        else:
            audio = None

        mel_spec = self.dataset_file["mel_spec"][idx, :, :, :]
        if self.mean is not None and self.std is not None:
            mel_spec = (mel_spec - self.mean) / self.std

        param_array = self.dataset_file["param_array"][idx, :]
        if self.rescale_params:
            param_array = param_array * 2 - 1

        return dict(
            mel_spec=mel_spec,
            params=param_array,
            audio=audio,
        )


class SurgeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.use_saved_mean_and_variance = use_saved_mean_and_variance
        self.batch_size = batch_size
        self.ot = ot
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SurgeXTDataset(
            self.dataset_root / "train.h5",
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
        )
        self.val_dataset = SurgeXTDataset(
            self.dataset_root / "val.h5",
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
        )
        self.test_dataset = SurgeXTDataset(
            self.dataset_root / "test.h5",
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ot_collate_fn if self.ot else regular_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=regular_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=regular_collate_fn,
            pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        self.train_dataset.dataset_file.close()
        self.val_dataset.dataset_file.close()
        self.test_dataset.dataset_file.close()
