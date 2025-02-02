import h5py
import numpy as np

splits = dict(
    train=(1, 200),
    val=(201, 210),
    test=(211, 220),
)

for split, (lo, hi) in splits.items():
    print(split)
    split_len = (hi - lo + 1) * 10_000

    vl_audio = h5py.VirtualLayout(shape=(split_len, 2, 44100 * 4), dtype=np.float32)
    vl_mel = h5py.VirtualLayout(shape=(split_len, 2, 128, 401), dtype=np.float32)
    vl_param = h5py.VirtualLayout(shape=(split_len, 189), dtype=np.float32)

    for i in range(lo, hi + 1):
        source_name = f"/data/home/acw585/data_scratch/surge-mini/shard-{i}.h5"
        vs_audio = h5py.VirtualSource(
            source_name, "audio", dtype=np.float32, shape=(10_000, 2, 44100 * 4)
        )
        vs_mel = h5py.VirtualSource(
            source_name, "mel_spec", dtype=np.float32, shape=(10_000, 2, 128, 401)
        )
        vs_param = h5py.VirtualSource(
            source_name, "param_array", dtype=np.float32, shape=(10_000, 189)
        )

        range_start = (i - lo) * 10_000
        range_end = (i + 1 - lo) * 10_000

        print(range_start, range_end)
        vl_audio[range_start:range_end, :, :] = vs_audio
        vl_mel[range_start:range_end, :, :, :] = vs_mel
        vl_param[range_start:range_end, :] = vs_param

    with h5py.File(f"/data/home/acw585/data_scratch/surge-mini/{split}.h5", "w") as f:
        f.create_virtual_dataset("audio", vl_audio)
        f.create_virtual_dataset("mel_spec", vl_mel)
        f.create_virtual_dataset("param_array", vl_param)
