import h5py
import hdf5plugin
import numpy as np
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

FILE_PATH = root / "datasets/vital_20k/shard_0.h5"
IDX = 5509

def check():
    with h5py.File(FILE_PATH, "r") as f:
        audio_ds = f["audio"]
        print(f"Audio shape: {audio_ds.shape}")
        
        audio = audio_ds[IDX]
        print(f"Sample {IDX} shape: {audio.shape}")
        print(f"Sample {IDX} min: {np.min(audio)}, max: {np.max(audio)}")
        print(f"Sample {IDX} mean: {np.mean(audio)}")
        
        if audio.size == 0:
            print("Audio array is empty!")
        
        # Check if it's all zeros
        if np.all(audio == 0):
            print("Audio is all zeros.")

        import soundfile as sf
        out_dir = root / "inspection_output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "sample_5509_debug.wav"
        sf.write(out_path, audio.T.astype(np.float32), 44100)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    check()
