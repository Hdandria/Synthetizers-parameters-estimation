import os
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import rootutils
from loguru import logger
import shutil

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Only needed for HDF5 files
from src.data.surge_datamodule import SurgeXTDataset


def is_virtual_dataset(filename):
    """Check if HDF5 file is a virtual dataset."""
    with h5py.File(filename, "r") as f:
        if "mel_spec" in f:
            return f["mel_spec"].is_virtual
    return False


def get_shard_files(dataset_dir):
    """Get all shard files from dataset directory, sorted numerically."""
    dataset_path = Path(dataset_dir)
    shard_files = list(dataset_path.glob("shard_*.h5"))
    if not shard_files:
        shard_files = list(dataset_path.glob("shard-*.h5"))
    
    # Sort numerically by shard number
    def extract_shard_number(path):
        match = re.search(r'shard[_-](\d+)\.h5', path.name)
        return int(match.group(1)) if match else 0
    
    return sorted(shard_files, key=extract_shard_number)


def get_stats_hdf5(filename):
    dataset_name = "mel_spec"

    print(f"Computing stats for {filename}...")
    
    # Check if this is a VDS file
    if is_virtual_dataset(filename):
        print("Detected virtual dataset - computing from source shards instead")
        dataset_dir = Path(filename).parent
        shard_files = get_shard_files(dataset_dir)
        
        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")
        
        print(f"Found {len(shard_files)} shard files")
        
        # Compute stats across all shards
        count = 0
        mean = None
        M2 = None
        
        for shard_file in shard_files:
            print(f"Processing {shard_file.name}...")

            # Copy shard locally to ensure full download from object storage
            with tempfile.NamedTemporaryFile(suffix=shard_file.suffix, delete=False) as tmp:
                local_path = Path(tmp.name)
            try:
                shutil.copyfile(shard_file, local_path)
                with h5py.File(local_path, "r") as f:
                    dataset = f[dataset_name]
                    num_samples = dataset.shape[0]

                    chunk_size = 1000
                    for i in range(0, num_samples, chunk_size):
                        end_idx = min(i + chunk_size, num_samples)
                        chunk = dataset[i:end_idx]

                        for j in range(chunk.shape[0]):
                            sample = chunk[j]
                            count += 1

                            if mean is None:
                                mean = sample.astype(np.float64).copy()
                                M2 = np.zeros_like(mean, dtype=np.float64)
                            else:
                                delta = sample.astype(np.float64) - mean
                                mean += delta / count
                                delta2 = sample.astype(np.float64) - mean
                                M2 += delta * delta2
            finally:
                try:
                    local_path.unlink(missing_ok=True)
                except Exception:
                    pass
        
        # Finalize stats
        variance = M2 / (count - 1) if count > 1 else np.zeros_like(M2)
        std = np.sqrt(variance)
        
    else:
        # Regular HDF5 file - process directly
        with h5py.File(filename, "r") as f:
            dataset = f[dataset_name]
            num_samples = dataset.shape[0]
            
            # Initialize Welford's algorithm state
            count = 0
            mean = None
            M2 = None
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            print(f"Processing {num_samples} samples in chunks of {chunk_size}...")
            
            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)
                chunk = dataset[i:end_idx]
                
                # Update stats for each sample in the chunk
                for j in range(chunk.shape[0]):
                    sample = chunk[j]
                    count += 1
                    
                    if mean is None:
                        mean = sample.astype(np.float64).copy()
                        M2 = np.zeros_like(mean, dtype=np.float64)
                    else:
                        delta = sample.astype(np.float64) - mean
                        mean += delta / count
                        delta2 = sample.astype(np.float64) - mean
                        M2 += delta * delta2
                
                if (i // chunk_size + 1) % 10 == 0:
                    print(f"  Processed {min(end_idx, num_samples)}/{num_samples} samples")
            
            # Finalize stats
            variance = M2 / (count - 1) if count > 1 else np.zeros_like(M2)
            std = np.sqrt(variance)
    
    print(f"Processed {count} total samples")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean value: {mean.mean():.6f}")
    print(f"Std value: {std.mean():.6f}")
    
    # Save stats
    print("Saving to file...")
    out_file = SurgeXTDataset.get_stats_file_path(filename)
    np.savez(out_file, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Saved to {out_file}")


def update(existing, new):
    count, mean, M2 = existing
    count += 1
    delta = new - mean
    mean += delta / count
    delta2 = new - mean
    M2 += delta * delta2
    return count, mean, M2


def finalize(existing):
    count, mean, M2 = existing
    variance = M2 / count if count > 1 else 0
    return mean, np.sqrt(variance)


# NOTE: Disabled - requires AudioFolderDataset which doesn't exist
# def get_stats_directory(directory):
#     dataset = AudioFolderDataset(directory)
#     out_file = AudioFolderDataset.get_stats_file_path(directory)
#
#     existing = (0, 0, 0)
#     # we run Welford's online algorithm
#     for i in range(len(dataset)):
#         x = dataset[i]["mel_spec"]
#         existing = update(existing, x)
#
#         if i % 10 == 0:
#             logger.info(f"Processed {i + 1} files...")
#
#     mean, std = finalize(existing)
#
#     logger.info(f"Saving to {str(out_file)}")
#
#     np.savez(out_file, mean=mean, std=std)


if __name__ == "__main__":
    # filename = "/data/scratch/acw585/surge/train.hdf5"
    filename = sys.argv[1]

    if os.path.splitext(filename)[-1] == ".h5":
        get_stats_hdf5(filename)
    else:
        raise NotImplementedError("Only .h5 files are supported. Directory-based datasets are not implemented.")
