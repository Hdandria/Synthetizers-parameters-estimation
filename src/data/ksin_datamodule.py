from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
from lightning import LightningDataModule
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits


def _sample_freqs(
    k: int,
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.empty(num_samples, k, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )


def _sample_amplitudes(
    k: int,
    num_samples: int,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.empty(num_samples, k, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )


def _sample_freqs_shifted(
    k: int,
    num_samples: int,
    is_test: bool,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample frequencies with different train and test distributions. These are
    slightly overlapping truncated normal distributions.
    """
    freqs = torch.empty(num_samples, k, device=device)
    mean = -1.0 / 3.0 if not is_test else 1.0 / 3.0

    torch.nn.init.trunc_normal_(freqs, mean, 1.0 / 3.0, -1.0, 1.0, generator=generator)

    return freqs


def make_sin(
    freqs: torch.Tensor, amps: torch.Tensor, length: int, break_symmetry: bool = False
):
    freqs = torch.pi * (freqs + 1.0) / 2.0

    if break_symmetry:
        k = freqs.shape[-1]
        shift = torch.arange(k, device=freqs.device) * torch.pi / k
        freqs = shift + freqs / k

    amps = (amps + 1.0) / 2.0

    n = torch.arange(length, device=freqs.device)
    phi = freqs[..., None] * n
    x = torch.sin(phi)
    x = x * amps[..., None]

    return x.sum(dim=-2)


class KSinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        k: int,
        signal_length: int,
        num_samples: int,
        sort_frequencies: bool,
        break_symmetry: bool,
        shift_test_distribution: bool,
        is_test: bool,
        seed: int,
    ):
        self.k = k
        self.signal_length = signal_length

        if shift_test_distribution and break_symmetry:
            raise ValueError(
                "Cannot use `shift_test_distribution` and `break_symmetry` at the same"
                "time."
            )

        self.sort_frequencies = sort_frequencies
        self.break_symmetry = break_symmetry
        self.shift_test_distribution = shift_test_distribution

        self.num_samples = num_samples

        self.seed = seed
        self.generator = torch.Generator(device=torch.device("cpu"))

        self.is_test = is_test

        self._init_dataset()

    def _init_dataset(self):
        self.generator.manual_seed(self.seed)
        freqs, amps = self._sample_parameters()
        freqs.share_memory_()
        amps.share_memory_()
        self.freqs = freqs
        self.amps = amps

    def _sample_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shift_test_distribution:
            freqs = _sample_freqs_shifted(
                self.k,
                self.num_samples,
                self.is_test,
                torch.device("cpu"),
                self.generator,
            )
        else:
            freqs = _sample_freqs(
                self.k, self.num_samples, torch.device("cpu"), self.generator
            )

        amplitudes = _sample_amplitudes(
            self.k, self.num_samples, torch.device("cpu"), self.generator
        )

        if self.sort_frequencies:
            freqs, _ = torch.sort(freqs, dim=-1)

        return freqs, amplitudes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        freq = self.freqs[idx][None]
        amp = self.amps[idx][None]
        sin_fn = partial(
            make_sin, length=self.signal_length, break_symmetry=self.break_symmetry
        )
        params = torch.cat((freq, amp), dim=-1)
        sins = sin_fn(freq, amp)
        return (sins, params, sin_fn)


def _hungarian_match(noise: torch.Tensor, params: torch.Tensor, sins: torch.Tensor):
    cost = torch.cdist(noise, params)
    cost = cost.numpy()

    with threadpool_limits(limits=1):
        row_ind, col_ind = linear_sum_assignment(cost)

    noise = noise[row_ind]
    params = params[col_ind]
    sins = sins[col_ind]

    return noise, params, sins


def regular_collate_fn(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)
    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


def ot_collate_fn(batch):
    sins, params, sin_fn = zip(*batch)
    sins = torch.cat(sins, dim=0)
    params = torch.cat(params, dim=0)
    noise = torch.randn_like(params)

    noise, params, sins = _hungarian_match(noise, params, sins)

    sin_fn = sin_fn[0]
    return (sins, params, noise, sin_fn)


class KSinDataModule(LightningDataModule):
    """k-Sin is a simple synthetic synthesiser parameter estimation task designed to
    elicit problematic behaviour in response to permutation invariant labels.

    Each item consists of a signal containing a mixture of sinusoids, and the amplitude
    and frequency parameters used to generate the sinusoids.
    """

    def __init__(
        self,
        k: int,
        signal_length: int = 1024,
        sort_frequencies: bool = False,
        break_symmetry: bool = False,
        shift_test_distribution: bool = False,
        train_val_test_sizes: Tuple[int, int, int] = (100_000, 10_000, 10_000),
        train_val_test_seeds: Tuple[int, int, int] = (123, 456, 789),
        batch_size: int = 1024,
        ot: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()

        # signal
        self.k = k
        self.signal_length = signal_length
        self.sort_frequencies = sort_frequencies
        self.break_symmetry = break_symmetry

        # dataset
        self.shift_test_distribution = shift_test_distribution
        self.train_size, self.val_size, self.test_size = train_val_test_sizes
        self.train_seed, self.val_seed, self.test_seed = train_val_test_seeds

        # dataloader
        self.batch_size = batch_size

        self.device = None
        self.num_workers = num_workers

        self.ot = ot

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.trainer is None:
            import warnings

            warnings.warn(
                "No trainer attached to datamodule, defaulting to device=None"
            )
            device = self.device
        else:
            device = self.trainer.strategy.root_device

        if stage == "fit":
            train_ds = KSinDataset(
                self.k,
                self.signal_length,
                self.train_size,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                False,
                self.train_seed,
            )
            val_ds = KSinDataset(
                self.k,
                self.signal_length,
                self.val_size,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                False,
                self.val_seed,
            )
            self.train = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=ot_collate_fn if self.ot else regular_collate_fn,
                num_workers=self.num_workers,
            )
            self.val = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=regular_collate_fn,
                num_workers=self.num_workers,
            )
        else:
            test_ds = KSinDataset(
                self.k,
                self.signal_length,
                self.test_size,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                True,
                self.test_seed,
            )
            self.test = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=regular_collate_fn,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        pass


if __name__ == "__main__":
    dm = KSinDataModule(k=4)
    dm.setup("fit")
    for x, y in dm.train:
        print(x.shape, y.shape)
        break
