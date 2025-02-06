from pathlib import Path

import click
import h5py
import numpy as np
from einops import rearrange
from loguru import logger
from music2latent import EncoderDecoder
from tqdm import tqdm, trange


@click.command()
@click.argument("data_dir", type=str)
@click.option("--batch-size", "-c", type=int, default=1024)
@click.option("--shard-range", "-r", type=int, nargs=2, default=None)
def main(data_dir: str, batch_size: int, shard_range: Optional[Tuple[int, int]] = None):
    data_dir = Path(data_dir)
    data_shards = data_dir.glob("shard-*.h5")

    logger.info("Initializing M2L")
    m2l = EncoderDecoder()

    outer_pbar = tqdm(data_shards, desc="Processing shards")
    for data_shard in outer_pbar:
        outer_pbar.set_description(f"Processing shard {data_shard.stem}")
        f = h5py.File(str(data_shard), "r+")
        num_samples, _ = f["audio"].shape

        outer_pbar.set_description(f"Creating dataset for shard {data_shard.stem}")
        f.create_dataset("music2latent", shape=(num_samples, 128, 42), dtype=np.float32)

        num_batches = num_samples // batch_size
        if num_samples % batch_size != 0:
            num_batches += 1

        outer_pbar.set_description(f"Processing batches for shard {data_shard.stem}")
        inner_pbar = trange(num_batches, desc="Processing batches", leave=False)
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            logger.info(f"Processing batch {i}")

            inner_pbar.set_description(f"Reading batch {i}")
            audio = f["audio"][start:end]

            audio = rearrange(audio, "b c t -> (b c) t")

            inner_pbar.set_description(f"Processing batch {i}")
            m2l_out = m2l(audio)
            m2l_out = rearrange(m2l_out, "(b c) d t -> b (c d) t")

            inner_pbar.set_description(f"Writing batch {i}")
            f["music2latent"][start:end] = m2l_out


if __name__ == "__main__":
    main()
