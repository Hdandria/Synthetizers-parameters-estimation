import os

import h5py
import hdf5plugin
import numpy as np
import scipy.io.wavfile as wavfile


def verify_and_export_audio(h5_path, output_dir, num_samples=10):
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'audio' not in f:
                print("Error: 'audio' dataset not found in HDF5 file.")
                return

            audio_ds = f['audio']
            print(f"Audio dataset shape: {audio_ds.shape}")

            num_to_read = min(num_samples, audio_ds.shape[0])

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for i in range(num_to_read):
                audio_data = audio_ds[i]
                print(f"Sample {i} shape: {audio_data.shape}")

                # Check for silence
                if np.all(audio_data == 0):
                    print(f"Sample {i}: WARNING - All zeros (Silent)")
                else:
                    max_val = np.max(np.abs(audio_data))
                    print(f"Sample {i}: Max amplitude = {max_val}")

                # Save as WAV
                if audio_data.shape[0] == 2:
                    audio_to_save = audio_data.T
                else:
                    audio_to_save = audio_data

                output_filename = os.path.join(output_dir, f"sample_{i}.wav")
                wavfile.write(output_filename, 44100, audio_to_save.astype(np.float32))
                print(f"Saved {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    verify_and_export_audio("datasets/vital-presets-test/shard-0.h5", "debug_audio")
