import sys
from functools import partial
from typing import Optional, Tuple, Union

import torch
from lightning import LightningDataModule

from src.data.ot import ot_collate_fn, regular_collate_fn
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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


def make_sin(params: torch.Tensor, length: int, break_symmetry: bool = False) -> torch.Tensor:
    """Generate a sinusoidal signal from frequency and amplitude parameters.

    This function creates a mixture of sinusoids from parameter vectors containing
    frequencies and amplitudes. It's designed to test permutation-invariant parameter
    estimation models.

    Args:
        params: Parameter tensor of shape (..., 2*k) where k is the number of sinusoids.
                The first k parameters are frequencies, the last k are amplitudes.
        length: Length of the output signal in samples
        break_symmetry: If True, adds frequency shifts to break permutation symmetry

    Returns:
        Generated sinusoidal signal of shape (..., length)

    Example:
        ```python
        # Generate 4-sinusoid mixture
        params = torch.randn(1, 8)  # 4 freqs + 4 amps
        signal = make_sin(params, length=1024)
        ```
    """
    freqs, amps = params.chunk(2, dim=-1)
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
    """Dataset for k-sin parameter estimation task.

    This dataset generates synthetic signals consisting of k sinusoidal components
    with random frequencies and amplitudes. It's designed to test permutation-invariant
    parameter estimation models by providing ground truth parameters that can be
    permuted without changing the resulting audio signal.

    Args:
        k: Number of sinusoidal components in each signal
        signal_length: Length of generated signals in samples
        num_samples: Number of samples in the dataset
        sort_frequencies: If True, sort frequencies to break some permutation symmetry
        break_symmetry: If True, add frequency shifts to break permutation symmetry
        shift_test_distribution: If True, use different frequency distribution for test set
        is_test: Whether this is a test dataset (affects frequency distribution)
        seed: Random seed for reproducible generation

    Example:
        ```python
        # Create dataset with 4 sinusoids
        dataset = KSinDataset(
            k=4,
            signal_length=1024,
            num_samples=10000,
            sort_frequencies=True,
            break_symmetry=False,
            shift_test_distribution=False,
            is_test=False,
            seed=123
        )
        ```
    """

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
        # freqs, amps = self._sample_parameters()
        # # freqs.share_memory_()
        # # amps.share_memory_()
        # log.info("Done!")
        # self.freqs = freqs
        # self.amps = amps

    def _sample_parameters(self, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.generator.manual_seed(seed)

        if self.shift_test_distribution:
            freqs = _sample_freqs_shifted(
                self.k,
                1,
                self.is_test,
                torch.device("cpu"),
                self.generator,
            )
        else:
            freqs = _sample_freqs(self.k, 1, torch.device("cpu"), self.generator)

        amplitudes = _sample_amplitudes(self.k, 1, torch.device("cpu"), self.generator)

        if self.sort_frequencies:
            freqs, _ = torch.sort(freqs, dim=-1)

        return freqs, amplitudes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # modulo max int to avoid overflows
        seed = (self.seed * idx) % sys.maxsize
        freq, amp = self._sample_parameters(seed)
        sin_fn = partial(
            make_sin, length=self.signal_length, break_symmetry=self.break_symmetry
        )
        params = torch.cat((freq, amp), dim=-1)
        sins = sin_fn(params)
        return (sins, params, sin_fn)


class KSinDataModule(LightningDataModule):
    """PyTorch Lightning data module for the k-sin parameter estimation task.

    The k-sin task is a synthetic synthesizer parameter estimation problem designed
    to test models' ability to handle permutation-invariant parameter spaces. Each
    sample consists of a mixture of k sinusoids and the corresponding frequency and
    amplitude parameters used to generate them.

    The key challenge is that the parameters can be permuted without changing the
    resulting audio signal, making this a permutation-invariant learning problem.
    This data module provides various options to control the symmetry properties
    of the generated data.

    Args:
        k: Number of sinusoidal components in each signal
        signal_length: Length of generated signals in samples
        sort_frequencies: If True, sort frequencies to partially break permutation symmetry
        break_symmetry: If True, add frequency shifts to break permutation symmetry
        shift_test_distribution: If True, use different frequency distribution for test set
        train_val_test_sizes: Tuple of (train, val, test) dataset sizes
        train_val_test_seeds: Tuple of (train, val, test) random seeds
        batch_size: Batch size for data loaders
        ot: If True, use optimal transport for minibatch coupling
        num_workers: Number of worker processes for data loading

    Example:
        ```python
        # Create data module
        dm = KSinDataModule(
            k=4,
            signal_length=1024,
            sort_frequencies=True,
            batch_size=64,
            ot=True
        )

        # Setup and use
        dm.setup()
        train_loader = dm.train_dataloader()
        ```
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
