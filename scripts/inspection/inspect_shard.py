import h5py
import hdf5plugin
import numpy as np
import soundfile as sf
import os
import random
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

FILE_PATH = root / "datasets/vital_20k/shard_0.h5"
OUTPUT_DIR = root / "inspection_output"

def inspect():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with h5py.File(FILE_PATH, "r") as f:
        print("Keys:", list(f.keys()))
        
        if "audio" not in f:
            print("No audio dataset found.")
            return

        audio_ds = f["audio"]
        print(f"Audio shape: {audio_ds.shape}")
        print(f"Audio dtype: {audio_ds.dtype}")
        
        # Check attributes
        attrs = dict(audio_ds.attrs)
        print("Attributes:", attrs)
        sample_rate = attrs.get("sample_rate", 44100)
        
        num_samples = audio_ds.shape[0]
        
        # Check for empty/nan
        print("Checking for silence/NaNs in a subset (first 100, middle 100, last 100)...")
        indices_to_check = list(range(0, min(num_samples, 100))) + \
                           list(range(num_samples//2, min(num_samples, num_samples//2 + 100))) + \
                           list(range(max(0, num_samples-100), num_samples))
        
        indices_to_check = sorted(list(set(indices_to_check)))
        
        issues_found = 0
        for idx in indices_to_check:
            audio = audio_ds[idx]
            if np.isnan(audio).any():
                print(f"NaN found at index {idx}")
                issues_found += 1
            if np.all(audio == 0):
                print(f"Silence found at index {idx}")
                issues_found += 1
            elif np.max(np.abs(audio)) < 1e-4:
                 print(f"Near silence found at index {idx} (max abs < 1e-4)")
                 issues_found += 1
        
        if issues_found == 0:
            print("No issues found in the checked subset.")

        # Generate 30 random wavs
        print("Generating 30 random WAV files...")
        random_indices = random.sample(range(num_samples), min(30, num_samples))
        
        for idx in random_indices:
            audio = audio_ds[idx]
            # Audio is (channels, n_samples) based on the generation script
            # soundfile expects (n_samples, channels)
            
            audio_to_save = audio.T
            
            out_path = os.path.join(OUTPUT_DIR, f"sample_{idx}.wav")
            sf.write(out_path, audio_to_save.astype(np.float32), int(sample_rate))
            print(f"Saved {out_path}")

if __name__ == "__main__":
    inspect()
