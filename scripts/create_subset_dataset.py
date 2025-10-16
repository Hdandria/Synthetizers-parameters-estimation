from pathlib import Path

import click
import h5py
import numpy as np


@click.command()
@click.argument("source_dataset_root", type=str)
@click.argument("target_dataset_root", type=str)
@click.option("--train-shards", "-t", type=str, help="Comma-separated shard indices for training, e.g., '0,1,2,3,4'")
@click.option("--val-shards", "-v", type=str, help="Comma-separated shard indices for validation, e.g., '5'")
@click.option("--test-shards", "-e", type=str, help="Comma-separated shard indices for testing, e.g., '6'")
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
        return [int(x.strip()) for x in shard_str.split(',')]
    
    train_indices = parse_shards(train_shards)
    val_indices = parse_shards(val_shards)
    test_indices = parse_shards(test_shards)
    
    # Get all shard files
    all_shard_files = sorted(list(source_root.glob("shard-*.h5")))
    
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
            vs_audio = h5py.VirtualSource(
                file, "audio", dtype=np.float32, shape=(10_000, *audio_shape)
            )
            vs_mel = h5py.VirtualSource(
                file, "mel_spec", dtype=np.float32, shape=(10_000, *mel_shape)
            )
            vs_param = h5py.VirtualSource(
                file, "param_array", dtype=np.float32, shape=(10_000, *param_shape)
            )
            
            range_start = i * 10_000
            range_end = (i + 1) * 10_000
            
            print(f"  Mapping {file.name} to indices {range_start}:{range_end}")
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