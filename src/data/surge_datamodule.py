import random
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
from lightning import LightningDataModule

from src.data.ot import _hungarian_match


class SurgeXTDataset(torch.utils.data.Dataset):
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def __init__(
        self,
        dataset_file: Union[str, Path],
        batch_size: int,
        ot: bool = True,
        read_audio: bool = False,
        use_saved_mean_and_variance: bool = True,
        rescale_params: bool = True,
        fake: bool = False,
    ):
        self.batch_size = batch_size
        self.ot = ot

        self.read_audio = read_audio
        self.rescale_params = rescale_params

        self.fake = fake
        if fake:
            self.dataset_file = None
            return

        self.dataset_file = h5py.File(dataset_file, "r")

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
        if self.fake:
            return 10000

        return self.dataset_file["audio"].shape[0] // self.batch_size

    def _get_fake_item(self):
        audio = (
            torch.randn(self.batch_size, 2, 44100 * 4) if not self.read_audio else None
        )
        mel_spec = torch.randn(self.batch_size, 2, 128, 401)
        param_array = torch.rand(self.batch_size, 189)

        if self.rescale_params:
            param_array = param_array * 2 - 1

        noise = torch.randn_like(param_array)

        return dict(
            mel_spec=mel_spec,
            params=param_array,
            noise=noise,
            audio=audio,
        )

    def _index_dataset(self, ds: h5py.Dataset, idx: Union[int, Sequence[int]]):
        if isinstance(idx, int):
            start_idx = idx * self.batch_size
            end_idx = start_idx + self.batch_size

            return ds[start_idx:end_idx]

        return ds[idx]

    def __getitem__(self, idx: Union[int, Sequence[int]]):
        if self.fake:
            return self._get_fake_item()

        if self.read_audio:
            audio = self._index_dataset(self.dataset_file["audio"], idx)
            audio = torch.from_numpy(audio).to(dtype=torch.float32)
        else:
            audio = None

        mel_spec = self._index_dataset(self.dataset_file["mel_spec"], idx)
        if self.mean is not None and self.std is not None:
            mel_spec = (mel_spec - self.mean) / self.std
        mel_spec = torch.from_numpy(mel_spec).to(dtype=torch.float32)

        param_array = self._index_dataset(self.dataset_file["param_array"], idx)
        if self.rescale_params:
            param_array = param_array * 2 - 1
        param_array = torch.from_numpy(param_array).to(dtype=torch.float32)
        noise = torch.randn_like(param_array)
        if self.ot:
            noise, param_array, mel_spec, audio = _hungarian_match(
                noise, param_array, mel_spec, audio
            )

        return dict(
            mel_spec=mel_spec.contiguous(),
            params=param_array.contiguous(),
            noise=noise.contiguous(),
            audio=audio.contiguous(),
        )


class WithinChunkShuffledSampler(torch.utils.data.Sampler):
    """When we have a hdf5 dataset on disk with layout:
        shard1.h5
        shard2.h5
        ...
        shardN.h5
    and each shard is 10,000 samples long, we want to sample items within each block of
    10,000 rather than randomly sampling across the entire dataset, to reduce the
    number of concurrent file handles that h5py has to deal with.
    This is not always exactly possible, but we can minimize the number of inter-shard
    reads to only the boundaries.
    """

    def __init__(self, batch_size: int, num_batches: int, batches_per_group: int):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batches_per_group = batches_per_group

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        num_groups = self.num_batches // self.batches_per_group
        samples_per_group = self.batches_per_group * self.batch_size
        group_sizes = [samples_per_group] * num_groups

        remaining = self.num_batches % self.batches_per_group
        if remaining > 0:
            num_groups += 1
            group_sizes.append(remaining * self.batch_size)

        indices = [
            np.random.permutation(group_size).reshape(-1, self.batch_size)
            + i * samples_per_group
            for i, group_size in enumerate(group_sizes)
        ]

        indices = np.concatenate(indices, axis=0)

        # shuffle by rows
        np.random.shuffle(indices)

        for row in indices:
            row.sort()
            yield row.tolist()


class ShiftedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches - 1

    def __iter__(self):
        offset = random.randint(0, self.num_batches - 1)
        perm = np.random.permutation(self.num_batches - 1)
        for i in perm:
            yield slice(
                i * self.batch_size + offset, (i + 1) * self.batch_size + offset
            )


class SurgeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_root: Union[str, Path],
        use_saved_mean_and_variance: bool = True,
        batch_size: int = 1024,
        ot: bool = True,
        num_workers: int = 0,
        fake: bool = False,
        predict_file: Optional[str] = None,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.use_saved_mean_and_variance = use_saved_mean_and_variance
        self.batch_size = batch_size
        self.ot = ot
        self.num_workers = num_workers
        self.fake = fake
        self.predict_file = predict_file

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = SurgeXTDataset(
            self.dataset_root / "train.h5",
            batch_size=self.batch_size,
            ot=self.ot,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
        )
        self.val_dataset = SurgeXTDataset(
            self.dataset_root / "val.h5",
            batch_size=self.batch_size,
            ot=False,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
        )
        self.test_dataset = SurgeXTDataset(
            self.dataset_root / "test.h5",
            batch_size=self.batch_size,
            ot=False,
            use_saved_mean_and_variance=self.use_saved_mean_and_variance,
            fake=self.fake,
        )
        if self.predict_file is not None:
            self.predict_dataset = SurgeXTDataset(
                self.predict_file,
                batch_size=self.batch_size,
                ot=False,
                use_saved_mean_and_variance=self.use_saved_mean_and_variance,
                fake=self.fake,
            )
        else:
            self.predict_dataset = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
            # sampler=WithinChunkShuffledSampler(
            #     self.batch_size, len(self.train_dataset), 4
            # ),
            sampler=ShiftedBatchSampler(self.batch_size, len(self.train_dataset)),
            # shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def teardown(self, stage: Optional[str] = None):
        self.train_dataset.dataset_file.close()
        self.val_dataset.dataset_file.close()
        self.test_dataset.dataset_file.close()
