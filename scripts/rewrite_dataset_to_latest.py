#!/usr/bin/env python
import concurrent.futures
import os
from pathlib import Path

import click
import h5py


def rewrite_shard(shard: Path, backup: bool = True):
    """
    Rewrites a given HDF5 shard file so that it is created with libver="latest".

    Parameters:
        shard (Path): Path to the HDF5 shard file.
        backup (bool): If True, save a backup of the original file with a .bak.h5 extension.

    Returns:
        A tuple of (shard_path, success_flag, error_message).
    """
    temp_file = shard.with_suffix(".temp.h5")
    try:
        # Open the original file in read-only mode.
        with (
            h5py.File(shard, "r") as f_in,
            h5py.File(temp_file, "w", libver="latest") as f_out,
        ):
            # Copy each top-level object (dataset or group).
            for key in f_in:
                f_in.copy(key, f_out)

        # If backup is enabled, rename the original file.
        if backup:
            backup_file = shard.with_suffix(".bak.h5")
            os.replace(shard, backup_file)
        else:
            os.remove(shard)

        # Replace the original file with the new file.
        os.replace(temp_file, shard)
        return (shard, True, "")
    except Exception as e:
        # Clean up the temporary file if it exists.
        if temp_file.exists():
            temp_file.unlink()
        return (shard, False, str(e))


@click.command()
@click.argument("data_dir", type=str)
@click.option(
    "--pattern",
    "-p",
    type=str,
    default="shard-*.h5",
    help="Glob pattern to find shard files.",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    help="Whether to keep a backup of the original file (with a .bak.h5 extension).",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Number of worker processes to run in parallel.",
)
def main(data_dir, pattern, backup, workers):
    """
    Rewrite each HDF5 file matching the given pattern in DATA_DIR so that it is created
    with libver="latest" (i.e. with a superblock version >= 3). This is needed for SWMR mode.
    """
    data_dir = Path(data_dir)
    shard_files = list(data_dir.glob(pattern))
    if not shard_files:
        click.echo(f"No files found matching pattern '{pattern}' in {data_dir}.")
        return

    click.echo(
        f"Found {len(shard_files)} file(s). Rewriting with libver='latest' using {workers} workers..."
    )

    # Use a process pool to rewrite shards concurrently.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit each file for rewriting.
        future_to_shard = {
            executor.submit(rewrite_shard, shard, backup): shard
            for shard in shard_files
        }
        for future in concurrent.futures.as_completed(future_to_shard):
            shard = future_to_shard[future]
            try:
                shard_path, success, error = future.result()
                if success:
                    click.echo(f"Successfully rewritten: {shard_path}")
                else:
                    click.echo(f"Error rewriting {shard_path}: {error}")
            except Exception as exc:
                click.echo(f"Unexpected error rewriting {shard}: {exc}")


if __name__ == "__main__":
    main()
