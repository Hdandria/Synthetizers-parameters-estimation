import sys

import dask.array as da
import h5py
import numpy as np
import rootutils
from dask.distributed import Client, progress

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.surge_datamodule import SurgeXTDataset

if __name__ == "__main__":

    # filename = "/data/scratch/acw585/surge/train.hdf5"
    filename = sys.argv[1]
    dataset_name = "mel_spec"

    num_workers = 4

    print("Starting client...")
    client = Client(n_workers=num_workers, threads_per_worker=8)
    # Create a dask array that references the HDF5 dataset
    # "chunks=" controls the chunk size in memory
    print("Creating dask array...")
    darray = da.from_array(
        h5py.File(filename, "r")[dataset_name],
        chunks="auto",  # You can tune this chunk size
    )

    print("Computing mean and std...")
    mean_task = darray.mean(axis=0)
    std_task = darray.std(axis=0)

    print("Persisting tasks...")
    futures = [mean_task.persist(), std_task.persist()]

    print("Displaying progress...")
    progress(futures)

    print("Gathering results...")
    mean_val, std_val = client.gather(futures)

    print("Mean:", mean_val)
    print("std:", std_val)

    print("Saving to file...")
    out_file = SurgeXTDataset.get_stats_file_path(filename)
    mean = mean_val.compute()
    std = std_val.compute()
    np.savez(out_file, mean=mean, std=std)
