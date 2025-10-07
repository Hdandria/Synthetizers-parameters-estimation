"""
Utility script to compute dataset statistics for normalization
Handles multiple HDF5 files efficiently
"""
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_stats_single_file(h5_path):
    """
    Compute mean and std for a single HDF5 file
    Returns sum, sum_sq, and count for incremental computation
    """
    print(f"Processing {h5_path.name}...")
    
    with h5py.File(h5_path, 'r') as f:
        mel_spec = f['mel_spec']
        n_samples = mel_spec.shape[0]
        shape = mel_spec.shape[1:]
        
        # Incremental computation to save memory
        running_sum = np.zeros(shape, dtype=np.float64)
        running_sum_sq = np.zeros(shape, dtype=np.float64)
        
        for i in tqdm(range(n_samples), desc=f"  {h5_path.name}"):
            sample = mel_spec[i].astype(np.float64)
            running_sum += sample
            running_sum_sq += sample ** 2
    
    return running_sum, running_sum_sq, n_samples


def compute_stats_directory(directory_path, output_path=None):
    """
    Compute mean and std of mel spectrograms across all HDF5 files in a directory
    
    Args:
        directory_path: Path to directory containing .h5 files
        output_path: Where to save stats (default: directory/stats.npz)
    """
    directory_path = Path(directory_path)
    
    # Find all .h5 files
    h5_files = sorted(directory_path.glob('*.h5'))
    if not h5_files:
        raise ValueError(f"No .h5 files found in {directory_path}")
    
    print(f"Found {len(h5_files)} HDF5 files in {directory_path}")
    print("Computing statistics across all files...\n")
    
    # Accumulate statistics across all files
    total_sum = None
    total_sum_sq = None
    total_count = 0
    
    for h5_path in h5_files:
        file_sum, file_sum_sq, file_count = compute_stats_single_file(h5_path)
        
        if total_sum is None:
            total_sum = file_sum
            total_sum_sq = file_sum_sq
        else:
            total_sum += file_sum
            total_sum_sq += file_sum_sq
        
        total_count += file_count
    
    print(f"\nTotal samples processed: {total_count:,}")
    
    # Compute final mean and std
    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 0))  # Avoid negative values due to numerical errors
    
    # Save statistics
    if output_path is None:
        output_path = directory_path / 'stats.npz'
    
    np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"\nSaved statistics to {output_path}")
    
    print(f"\nStatistics:")
    print(f"Mean shape: {mean.shape}")
    print(f"Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
    print(f"Std shape: {std.shape}")
    print(f"Std range: [{std.min():.2f}, {std.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Compute dataset statistics across multiple HDF5 files'
    )
    parser.add_argument(
        'path', 
        help='Path to directory containing .h5 files or single .h5 file'
    )
    parser.add_argument('--output', '-o', help='Output path for stats.npz')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_dir():
        # Directory mode: process all .h5 files
        compute_stats_directory(path, args.output)
    elif path.is_file() and path.suffix == '.h5':
        # Single file mode (legacy support)
        print("Single file mode (consider using directory for all files)")
        file_sum, file_sum_sq, file_count = compute_stats_single_file(path)
        mean = file_sum / file_count
        variance = (file_sum_sq / file_count) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 0))
        
        output_path = args.output or path.parent / 'stats.npz'
        np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
        print(f"Saved statistics to {output_path}")
    else:
        raise ValueError(f"Path must be a directory or .h5 file: {path}")


if __name__ == '__main__':
    main()

