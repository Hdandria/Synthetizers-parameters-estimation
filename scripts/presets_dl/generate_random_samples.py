import os
import sys
import random
import shutil
import numpy as np
import soundfile as sf
import mido
from pedalboard import load_plugin
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.vst.core import load_preset

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESETS_DIR = "data/presets/vital"
OUTPUT_DIR = "generated_samples_30"

def generate_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading plugin from {PLUGIN_PATH}...")
    try:
        plugin = load_plugin(PLUGIN_PATH)
    except Exception as e:
        print(f"Failed to load plugin: {e}")
        return

    all_presets = list(Path(PRESETS_DIR).glob("*.vital"))
    random.shuffle(all_presets)
    
    print(f"Generating 30 samples from pool of {len(all_presets)} presets...")
    
    # Render Settings
    sample_rate = 44100
    duration = 4.0
    note_pitch = 60
    velocity = 100
    note_start = 0.5
    note_end = duration - 0.5
    
    note_on = mido.Message("note_on", note=note_pitch, velocity=velocity).bytes()
    note_off = mido.Message("note_off", note=note_pitch, velocity=velocity).bytes()
    
    midi_messages = [
        (note_on, note_start),
        (note_off, note_end)
    ]
    
    success_count = 0
    error_count = 0
    
    for p in tqdm(all_presets):
        try:
            # Load Preset
            load_preset(plugin, str(p))
            
            # Flush buffer (0.5s silence)
            plugin.process([], 0.5, sample_rate, num_channels=2, reset=True)
            
            # Render
            audio = plugin.process(
                midi_messages,
                duration,
                sample_rate,
                num_channels=2,
                buffer_size=2048,
                reset=True
            )
            
            # Check for Silence
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-4:
                print(f"Skipping {p.name}: Silent (RMS {rms:.6f})")
                continue

            # Save Audio
            output_filename = f"{p.stem}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            sf.write(output_path, audio.T, sample_rate)
            
            # Copy Vital File
            vital_dest = os.path.join(OUTPUT_DIR, p.name)
            shutil.copy(str(p), vital_dest)
            
            success_count += 1
            
            if success_count >= 30:
                break
            
        except Exception as e:
            error_count += 1
            print(f"Error processing {p.name}: {e}")
            
    print("\nGeneration Complete!")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Samples and Presets saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_samples()
