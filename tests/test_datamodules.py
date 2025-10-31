from pathlib import Path

import pytest
import torch

from src.data.surge_datamodule import SurgeDataModule


@pytest.mark.parametrize("batch_size", [8, 32])
def test_surge_datamodule_fake(batch_size: int) -> None:
    """Tests `SurgeDataModule` using `fake=True` so it doesn't require on-disk HDF5 files.

    The Surge datamodule supports a `fake` flag that creates synthetic data. This test
    verifies setup, dataloaders, and that batched tensors have expected dtypes and shapes.
    """
    dm = SurgeDataModule(dataset_root="/tmp/not_used", batch_size=batch_size, fake=True)

    # call setup to create fake datasets
    dm.setup()

    # dataloaders should be available
    td = dm.train_dataloader()
    vd = dm.val_dataloader()
    zd = dm.test_dataloader()

    assert td is not None
    assert vd is not None
    assert zd is not None

    # iterate a single batch from train loader
    batch = next(iter(td))

    # surge batch dict contains 'params' and 'noise' at minimum
    assert "params" in batch and "noise" in batch
    params = batch["params"]
    noise = batch["noise"]

    # params and noise should be tensors with batch_size rows
    assert params.shape[0] == batch_size
    assert noise.shape[0] == batch_size
    assert params.dtype == torch.float32
    assert noise.dtype == torch.float32
