import os
import re
from pathlib import Path

import click
import h5py
import numpy as np


@click.command()
@click.argument("source_dataset_root", type=str)
@click.argument("target_dataset_root", type=str)
@click.option(
    "--train-shards",
    "-t",
    type=str,
    help="Comma-separated shard indices for training, e.g., '0,1,2,3,4'",
)
@click.option(
    "--val-shards", "-v", type=str, help="Comma-separated shard indices for validation, e.g., '5'"
)
@click.option(
    "--test-shards", "-e", type=str, help="Comma-separated shard indices for testing, e.g., '6'"
)
def main(
    source_dataset_root: str,
    target_dataset_root: str,
    train_shards: str,
    val_shards: str,
    test_shards: str,
):
    """Create a subset dataset by selecting specific shards from an existing dataset."""

    source_root = Path(source_dataset_root)
    target_root = Path(target_dataset_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # Parse shard indices
    def parse_shards(shard_str):
        if not shard_str:
            return []
        return [int(x.strip()) for x in shard_str.split(",")]

    train_indices = parse_shards(train_shards)
    val_indices = parse_shards(val_shards)
    test_indices = parse_shards(test_shards)

    # Get all shard files (try both naming conventions)
    all_shard_files = list(source_root.glob("shard_*.h5"))
    if not all_shard_files:
        all_shard_files = list(source_root.glob("shard-*.h5"))
    
    # Sort numerically by shard number (not alphabetically)
    def extract_shard_number(path):
        match = re.search(r'shard[_-](\d+)\.h5', path.name)
        return int(match.group(1)) if match else 0
    
    all_shard_files = sorted(all_shard_files, key=extract_shard_number)

    splits = {
        "train": [all_shard_files[i] for i in train_indices],
        "val": [all_shard_files[i] for i in val_indices],
        "test": [all_shard_files[i] for i in test_indices],
    }

    for split, files in splits.items():
        if not files:
            continue

        print(f"Creating {split} split with {len(files)} shards")
        split_len = len(files) * 10_000

        # Get shapes from first file
        with h5py.File(files[0], "r") as f:
            audio_shape = f["audio"].shape[1:]
            mel_shape = f["mel_spec"].shape[1:]
            param_shape = f["param_array"].shape[1:]

        # Create virtual layouts
        vl_audio = h5py.VirtualLayout(shape=(split_len, *audio_shape), dtype=np.float32)
        vl_mel = h5py.VirtualLayout(shape=(split_len, *mel_shape), dtype=np.float32)
        vl_param = h5py.VirtualLayout(shape=(split_len, *param_shape), dtype=np.float32)

        # Map shards to virtual layout
        for i, file in enumerate(files):
            # We'll open the shard with an absolute path to read shape/validate,
            # but store a RELATIVE POSIX path in the VDS so the whole dataset
            # tree can be moved/copied and still resolve correctly.
            shard_abs = os.path.abspath(str(file))
            # compute a path to store in the VDS relative to the target_root
            # Compute relative path to the VDS file location, and force POSIX separators
            rel_native = os.path.relpath(shard_abs, start=str(target_root))
            # On Windows, relpath uses backslashes; make sure to store POSIX paths in VDS
            vds_path = rel_native.replace("\\", "/")
            # Remove any redundant ./ or ../ where possible (keeps leading .. intact)
            while "//" in vds_path:
                vds_path = vds_path.replace("//", "/")

            # Use the VDS path (relative) when creating VirtualSource
            vs_audio = h5py.VirtualSource(
                vds_path, "audio", dtype=np.float32, shape=(10_000, *audio_shape)
            )
            vs_mel = h5py.VirtualSource(
                vds_path, "mel_spec", dtype=np.float32, shape=(10_000, *mel_shape)
            )
            vs_param = h5py.VirtualSource(
                vds_path, "param_array", dtype=np.float32, shape=(10_000, *param_shape)
            )

            range_start = i * 10_000
            range_end = (i + 1) * 10_000

            print(f"  Mapping {vds_path} (-> {shard_abs}) to indices {range_start}:{range_end}")
            vl_audio[range_start:range_end, :, :] = vs_audio
            vl_mel[range_start:range_end, :, :, :] = vs_mel
            vl_param[range_start:range_end, :] = vs_param

        # Create virtual dataset file
        split_file = target_root / f"{split}.h5"
        with h5py.File(split_file, "w") as f:
            f.create_virtual_dataset("audio", vl_audio)
            f.create_virtual_dataset("mel_spec", vl_mel)
            f.create_virtual_dataset("param_array", vl_param)

        print(f"Created {split_file} with {split_len} samples")


if __name__ == "__main__":
    main()
