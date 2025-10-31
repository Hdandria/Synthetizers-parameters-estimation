import os
import sys

try:
    import hdf5plugin

    print("hdf5plugin OK:", getattr(hdf5plugin, "__version__", "unknown"))
except Exception as e:
    print("hdf5plugin import failed:", e, file=sys.stderr)

import h5py
import numpy as np

shard = "datasets/surge-100k/train.h5"  # change to an existing shard path
print("opening", shard)
with h5py.File(shard, "r") as f:
    print("top keys:", list(f.keys()))
    d = f["audio"]
    print("audio shape:", d.shape, "storage_size:", d.id.get_storage_size())
    # try a few samples
    for i in range(min(5, d.shape[0])):
        s = d[i]
        print(
            f"sample {i}: min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} any_nonzero={bool(np.any(s))}"
        )
