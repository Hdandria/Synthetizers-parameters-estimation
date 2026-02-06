"""
UMAP visualization of synthesizer parameter space.

This script loads parameter arrays from HDF5 datasets and visualizes them
in 2D using UMAP dimensionality reduction. UMAP is faster and often better
than t-SNE for:
- Comparing random vs preset-based parameter distributions
- Understanding parameter space coverage
- Identifying clusters in the parameter space
- Preserving both local and global structure
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import hdf5plugin  # Must be imported before reading compressed HDF5 files
import matplotlib.pyplot as plt
import numpy as np
import umap
from loguru import logger
from tqdm import tqdm


def load_params_from_dataset(
    dataset_path: str,
    max_samples: Optional[int] = None,
    shards: Optional[List[str]] = None
) -> np.ndarray:
    """Load parameter arrays from HDF5 dataset.
    
    Args:
        dataset_path: Path to HDF5 file or directory containing shards
        max_samples: Maximum number of samples to load (None = all)
        shards: List of shard filenames to load (for multi-shard datasets)
        
    Returns:
        Array of shape (n_samples, n_params)
    """
    path = Path(dataset_path)

    if path.is_file():
        # Single HDF5 file (may be a VDS pointing to shards)
        logger.info(f"Loading parameters from {path}")
        with h5py.File(path, "r") as f:
            dataset = f["param_array"]

            # Check if it's a Virtual Dataset
            if dataset.is_virtual:
                logger.info("Detected Virtual Dataset, loading from virtual sources")

            # Load data (handles both regular and virtual datasets)
            if max_samples:
                params = dataset[:max_samples]
            else:
                params = dataset[:]
        return params

    elif path.is_dir():
        # Directory with shards
        if shards:
            shard_files = [path / shard for shard in shards]
        else:
            shard_files = sorted(path.glob("shard*.h5"))

        logger.info(f"Loading parameters from {len(shard_files)} shards")
        params_list = []
        total_loaded = 0

        for shard_file in tqdm(shard_files, desc="Loading shards"):
            with h5py.File(shard_file, "r") as f:
                shard_params = f["param_array"][:]

                if max_samples and total_loaded + len(shard_params) > max_samples:
                    # Load partial shard to reach max_samples
                    remaining = max_samples - total_loaded
                    shard_params = shard_params[:remaining]

                params_list.append(shard_params)
                total_loaded += len(shard_params)

                if max_samples and total_loaded >= max_samples:
                    break

        return np.vstack(params_list)

    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")


def compute_umap(
    params: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42
) -> np.ndarray:
    """Compute UMAP embedding.
    
    Args:
        params: Parameter array of shape (n_samples, n_params)
        n_neighbors: UMAP n_neighbors parameter (balance local/global structure)
        min_dist: Minimum distance between points in embedding
        metric: Distance metric to use
        random_state: Random seed
        
    Returns:
        2D embedding of shape (n_samples, 2)
    """
    logger.info(f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )
    embedding = reducer.fit_transform(params)
    logger.info("UMAP computation complete")
    return embedding


def plot_umap_single(
    embedding: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
    alpha: float = 0.3,
    s: float = 1
):
    """Plot a single UMAP embedding.
    
    Args:
        embedding: 2D embedding of shape (n_samples, 2)
        title: Plot title
        output_path: Path to save figure (None = display only)
        alpha: Point transparency
        s: Point size
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=s)
    plt.title(title)
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_umap_comparison(
    embeddings: Dict[str, np.ndarray],
    title: str,
    output_path: Optional[str] = None,
    alpha: float = 0.3,
    s: float = 1
):
    """Plot multiple UMAP embeddings for comparison.
    
    Args:
        embeddings: Dict of {label: embedding} pairs
        title: Plot title
        output_path: Path to save figure (None = display only)
        alpha: Point transparency
        s: Point size
    """
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5))

    if len(embeddings) == 1:
        axes = [axes]

    for ax, (label, embedding) in zip(axes, embeddings.items()):
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=alpha, s=s)
        ax.set_title(label)
        ax.set_xlabel("UMAP dimension 1")
        ax.set_ylabel("UMAP dimension 2")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_umap_overlay(
    embeddings: Dict[str, np.ndarray],
    title: str,
    output_path: Optional[str] = None,
    alpha: float = 0.3,
    s: float = 1
):
    """Plot multiple UMAP embeddings overlaid on the same axes.
    
    Args:
        embeddings: Dict of {label: embedding} pairs
        title: Plot title
        output_path: Path to save figure (None = display only)
        alpha: Point transparency
        s: Point size
    """
    plt.figure(figsize=(10, 8))

    for label, embedding in embeddings.items():
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=alpha,
            s=s,
            label=label
        )

    plt.title(title)
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved overlay plot to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize parameter space with UMAP"
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Paths to HDF5 datasets (files or directories)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each dataset (default: dataset names)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples per dataset (default: 10000, 0=all)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (default: 15, higher=more global)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (default: 0.1, lower=tighter clusters)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine", "correlation"],
        help="Distance metric (default: euclidean)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/umap",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create side-by-side comparison plot"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Create overlay plot with all datasets"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Point transparency (default: 0.3)"
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Point size (default: 1.0)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate labels if not provided
    if args.labels:
        if len(args.labels) != len(args.datasets):
            raise ValueError("Number of labels must match number of datasets")
        labels = args.labels
    else:
        labels = [Path(d).stem for d in args.datasets]

    # Load parameters from all datasets
    all_params = {}
    max_samples = args.max_samples if args.max_samples > 0 else None
    for dataset_path, label in zip(args.datasets, labels):
        params = load_params_from_dataset(dataset_path, max_samples=max_samples)
        all_params[label] = params
        logger.info(f"{label}: {params.shape[0]} samples, {params.shape[1]} parameters")

    # Compute UMAP for each dataset
    embeddings = {}
    for label, params in all_params.items():
        embedding = compute_umap(
            params,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric
        )
        embeddings[label] = embedding

        # Plot individual
        plot_umap_single(
            embedding,
            title=f"UMAP: {label}",
            output_path=output_dir / f"umap_{label}.png",
            alpha=args.alpha,
            s=args.point_size
        )

    # Create comparison plot if multiple datasets
    if len(embeddings) > 1:
        if args.comparison:
            plot_umap_comparison(
                embeddings,
                title="UMAP Comparison: Parameter Space Distribution",
                output_path=output_dir / "umap_comparison.png",
                alpha=args.alpha,
                s=args.point_size
            )

        if args.overlay:
            plot_umap_overlay(
                embeddings,
                title="UMAP Overlay: Parameter Space Distribution",
                output_path=output_dir / "umap_overlay.png",
                alpha=args.alpha,
                s=args.point_size
            )

    logger.info(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
