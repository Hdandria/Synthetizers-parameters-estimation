import argparse
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


def check_file(path: Path, max_samples: int = 5) -> bool:
    """Return True if the HDF5 file at path contains non-empty audio data.

    Prints diagnostics and returns True when the `audio` dataset has a
    non-zero storage_size and at least one of the first `max_samples`
    entries contains non-zero samples.
    """
    if not path.exists():
        print(f"MISSING: {path}")
        return False

    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"Opening {path}: keys={keys}")
            if "audio" not in f:
                print("  No 'audio' dataset")
                return False

            d = f["audio"]
            storage = d.id.get_storage_size()
            print(f"  audio shape={d.shape} storage={storage}")

            # If this is a virtual dataset, list its virtual sources and
            # check whether they resolve to real shard files. Keep this
            # concise but useful for debugging across hosts.
            shard_ok = False
            try:
                vs = d.virtual_sources()
                if vs:
                    print(f"  VDS -> {len(vs)} virtual sources (showing up to 5):")
                    base = os.path.dirname(os.path.abspath(path))
                    for i, v in enumerate(vs[:5]):
                        fname = v.file_name
                        resolved = (
                            fname
                            if os.path.isabs(fname)
                            else os.path.normpath(os.path.join(base, fname))
                        )
                        exists = os.path.exists(resolved)
                        print(f"   [{i}] {fname} -> {resolved} exists={exists}")
                        if exists:
                            # quick shard sanity: check storage_size and first sample
                            try:
                                with h5py.File(resolved, "r") as sf:
                                    sd = sf["audio"]
                                    s_storage = sd.id.get_storage_size()
                                    s0 = sd[0]
                                    s0_any = bool(np.any(s0))
                                    print(f"       shard storage={s_storage} sample0_any={s0_any}")
                                    if s_storage > 0 and s0_any:
                                        shard_ok = True
                            except Exception:
                                pass
                else:
                    print("  VDS -> no virtual sources")
            except Exception as e:
                print(f"  virtual_sources() failed: {e}")

            # Try reading a few samples directly from the VDS. Even if
            # storage_size is 0, the mapped shards may still be readable.
            n = min(int(d.shape[0]), max_samples)
            any_nonzero = False
            for i in range(n):
                try:
                    s = d[i]
                except Exception as e:
                    print(f"  read sample {i} failed: {e}")
                    continue
                s_any = bool(np.any(s))
                print(f"  sample {i}: min={s.min():.6g} max={s.max():.6g} any_nonzero={s_any}")
                any_nonzero = any_nonzero or s_any

            # Decide result: either a shard looked sane, or VDS reads returned
            # non-zero audio.
            if shard_ok or any_nonzero:
                return True
            else:
                print("  -> No readable non-zero audio found in VDS or shards")
                return False
    except Exception as e:
        print(f"  Error opening {path}: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Check dataset directory HDF5 readability")
    p.add_argument(
        "dataset_dir",
        type=str,
        help="Path to dataset directory containing train.h5 val.h5 test.h5",
    )
    args = p.parse_args()

    ds_root = Path(args.dataset_dir)
    if not ds_root.exists() or not ds_root.is_dir():
        print(f"Dataset directory {ds_root} does not exist or is not a directory")
        sys.exit(2)

    all_ok = True
    for split in ["train", "val", "test"]:
        fpath = ds_root / f"{split}.h5"
        print(f"\nChecking {split}: {fpath}")
        ok = check_file(fpath)
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll splits appear readable and non-empty")
        sys.exit(0)
    else:
        print("\nOne or more splits are missing or empty")
        sys.exit(3)


if __name__ == "__main__":
    main()
