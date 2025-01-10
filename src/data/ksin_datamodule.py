from functools import partial
from typing import Optional, Tuple, Union

import torch
from lightning import LightningDataModule


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


class KSinDataLoader:
    def __init__(
        self,
        k: int,
        signal_length: int,
        sort_frequencies: bool,
        break_symmetry: bool,
        shift_test_distribution: bool,
        batch_size: int,
        batches_per_epoch: int,
        is_test: bool,
        seed: int,
        device: Union[str, torch.device],
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

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.seed = seed
        self.generator = torch.Generator(device=device)
        self.device = device

        self.is_test = is_test

    def __len__(self):
        return self.batches_per_epoch

    def _sample_parameters(
        self, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self.break_symmetry:
        #     freqs = _sample_freqs_symmetry_broken(
        #         self.k, self.batch_size, self.device, generator
        #     )
        if self.shift_test_distribution:
            freqs = _sample_freqs_shifted(
                self.k, self.batch_size, self.is_test, self.device, generator
            )
        else:
            freqs = _sample_freqs(self.k, self.batch_size, self.device, generator)

        amplitudes = _sample_amplitudes(self.k, self.batch_size, self.device, generator)

        if self.sort_frequencies:
            freqs, _ = torch.sort(freqs, dim=-1)

        return freqs, amplitudes

    def _make_batch(self, generator: Optional[torch.Generator] = None):
        sin_fn = partial(
            make_sin, length=self.signal_length, break_symmetry=self.break_symmetry
        )
        freqs, amps = self._sample_parameters(generator)
        sins = sin_fn(freqs, amps)
        params = torch.cat((freqs, amps), dim=-1)
        return (sins, params, sin_fn)

    def __iter__(self):
        self.generator.manual_seed(self.seed)
        for _ in range(self.batches_per_epoch):
            yield self._make_batch(self.generator)

        raise StopIteration


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
            self.train = KSinDataLoader(
                self.k,
                self.signal_length,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                self.batch_size,
                self.train_size // self.batch_size,
                False,
                self.train_seed,
                device,
            )
            self.val = KSinDataLoader(
                self.k,
                self.signal_length,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                self.batch_size,
                self.val_size // self.batch_size,
                False,
                self.val_seed,
                device,
            )
        else:
            self.test = KSinDataLoader(
                self.k,
                self.signal_length,
                self.sort_frequencies,
                self.break_symmetry,
                self.shift_test_distribution,
                self.batch_size,
                self.test_size // self.batch_size,
                True,
                self.test_seed,
                device,
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
