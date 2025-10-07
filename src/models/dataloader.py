"""
Simple data loader for Surge VST parameter estimation
Handles large datasets split across multiple HDF5 files
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class MultiFileDataset(Dataset):
    """Dataset that loads from multiple HDF5 files with lazy loading"""
    
    def __init__(self, h5_files, indices, batch_size=256, normalize_mel=True, stats_path=None):
        """
        Args:
            h5_files: List of paths to HDF5 files
            indices: List of (file_idx, sample_idx) tuples indicating which samples to use
            batch_size: Batch size for pre-batching
            normalize_mel: Whether to normalize mel spectrograms
            stats_path: Path to stats.npz file for normalization
        """
        self.h5_files = h5_files
        self.indices = indices
        self.batch_size = batch_size
        self.normalize_mel = normalize_mel
        
        # Load normalization stats
        self.mean = None
        self.std = None
        if normalize_mel and stats_path and Path(stats_path).exists():
            stats = np.load(stats_path)
            self.mean = stats['mean']
            self.std = stats['std']
        
        # Calculate number of batches
        self.num_batches = len(indices) // batch_size
        
        # Cache for open file handles (opened lazily)
        self._file_handles = {}
    
    def _get_file_handle(self, file_idx):
        """Lazily open and cache file handles"""
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = h5py.File(self.h5_files[file_idx], 'r')
        return self._file_handles[file_idx]
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, batch_idx):
        """Return a pre-batched item by gathering samples from potentially multiple files"""
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        
        # Get indices for this batch
        batch_indices = self.indices[start:end]
        
        # Gather samples (may come from different files)
        mel_specs = []
        params_list = []
        
        for file_idx, sample_idx in batch_indices:
            h5_file = self._get_file_handle(file_idx)
            mel_specs.append(h5_file['mel_spec'][sample_idx])
            params_list.append(h5_file['param_array'][sample_idx])
        
        # Stack into arrays
        mel_spec = np.stack(mel_specs, axis=0)
        params = np.stack(params_list, axis=0)
        
        # Normalize mel spectrogram
        if self.mean is not None and self.std is not None:
            mel_spec = (mel_spec - self.mean) / self.std
        
        # Rescale params from [0, 1] to [-1, 1]
        params = params * 2 - 1
        
        # Create noise for flow matching
        noise = np.random.randn(*params.shape).astype(np.float32)
        
        return {
            'mel_spec': torch.from_numpy(mel_spec).float(),
            'params': torch.from_numpy(params).float(),
            'noise': torch.from_numpy(noise).float()
        }
    
    def close(self):
        """Close all open file handles"""
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()


def split_dataset_indices(data_path, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Create train/val/test splits from a directory of HDF5 files
    
    Args:
        data_path: Path to directory containing .h5 files
        split_ratios: Tuple of (train, val, test) ratios
        seed: Random seed for reproducibility
    
    Returns:
        (train_files, train_indices), (val_files, val_indices), (test_files, test_indices)
        where files are paths and indices are lists of (file_idx, sample_idx) tuples
    """
    data_path = Path(data_path)
    
    # Find all .h5 files
    h5_files = sorted(data_path.glob('*.h5'))
    if not h5_files:
        raise ValueError(f"No .h5 files found in {data_path}")
    
    print(f"Found {len(h5_files)} HDF5 files:")
    for f in h5_files:
        print(f"  - {f.name}")
    
    # Count total samples and create global index
    all_indices = []
    for file_idx, h5_path in enumerate(h5_files):
        with h5py.File(h5_path, 'r') as f:
            num_samples = f['mel_spec'].shape[0]
            print(f"  {h5_path.name}: {num_samples} samples")
            # Create indices as (file_idx, sample_idx) tuples
            for sample_idx in range(num_samples):
                all_indices.append((file_idx, sample_idx))
    
    total_samples = len(all_indices)
    print(f"\nTotal samples: {total_samples:,}")
    
    # Shuffle indices with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(all_indices)
    
    # Split into train/val/test
    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_indices):,} samples ({len(train_indices)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_indices):,} samples ({len(val_indices)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_indices):,} samples ({len(test_indices)/total_samples*100:.1f}%)")
    
    return (h5_files, train_indices), (h5_files, val_indices), (h5_files, test_indices)


def create_dataloaders(config):
    """Create train, val, and test dataloaders from a directory of HDF5 files"""
    data_path = Path(config['data']['dataset_path'])
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    
    # Get split ratios from config (default to 80/10/10)
    split_ratios = config['data'].get('split_ratios', [0.8, 0.1, 0.1])
    seed = config.get('seed', 42)
    
    # Check if stats file exists
    stats_path = data_path / 'stats.npz'
    
    # Split dataset into train/val/test
    print("Splitting dataset...")
    (h5_files, train_indices), (_, val_indices), (_, test_indices) = split_dataset_indices(
        data_path, tuple(split_ratios), seed
    )
    
    # Create datasets
    train_dataset = MultiFileDataset(
        h5_files, train_indices, batch_size, 
        normalize_mel=True, stats_path=stats_path
    )
    val_dataset = MultiFileDataset(
        h5_files, val_indices, batch_size,
        normalize_mel=True, stats_path=stats_path
    )
    test_dataset = MultiFileDataset(
        h5_files, test_indices, batch_size,
        normalize_mel=True, stats_path=stats_path
    )
    
    # Create dataloaders (batch_size=None because we pre-batch in dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

