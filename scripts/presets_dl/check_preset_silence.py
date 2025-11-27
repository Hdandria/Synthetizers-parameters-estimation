import os
import sys
import numpy as np
import mido
from pedalboard import load_plugin
from pathlib import Path
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.vst.core import load_preset

# Configure logging to file to avoid cluttering stdout
logging.basicConfig(filename='preset_silence_check.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESETS_DIR = "data/presets/vital"
SILENCE_THRESHOLD = 1e-4  # RMS threshold for "silence"

def is_silent(audio, threshold=SILENCE_THRESHOLD):
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold, rms

def check_presets():
    print(f"Loading plugin from {PLUGIN_PATH}...")
    try:
        plugin = load_plugin(PLUGIN_PATH)
    except Exception as e:
        print(f"Failed to load plugin: {e}")
        return

    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    total = len(presets)
    silent_count = 0
    audible_count = 0
    errors = 0
    
    print(f"Checking {total} presets for silence...")
    
    # Pre-calculate MIDI messages
    sample_rate = 44100
    duration = 1.0 # Short duration for speed
    note_pitch = 60
    velocity = 100
    note_start = 0.1
    note_end = duration - 0.1
    
    note_on = mido.Message("note_on", note=note_pitch, velocity=velocity).bytes()
    note_off = mido.Message("note_off", note=note_pitch, velocity=velocity).bytes()
    
    midi_messages = [
        (note_on, note_start),
        (note_off, note_end)
    ]
    
    # Buffer for flush
    empty_midi = []

    for p in tqdm(presets):
        try:
            # Load Preset
            load_preset(plugin, str(p))
            
            # Flush buffer (Important!)
            plugin.process(empty_midi, 0.1, sample_rate, num_channels=2, reset=True)
            
            # Render
            audio = plugin.process(
                midi_messages,
                duration,
                sample_rate,
                num_channels=2,
                buffer_size=512, # Smaller buffer might be faster?
                reset=True
            )
            
            silent, rms = is_silent(audio)
            
            if silent:
                silent_count += 1
                logging.info(f"SILENT: {p.name} (RMS: {rms:.6f})")
            else:
                audible_count += 1
                logging.info(f"AUDIBLE: {p.name} (RMS: {rms:.6f})")
                
        except Exception as e:
            errors += 1
            logging.error(f"ERROR: {p.name} - {e}")
            
    print("\nResults:")
    print(f"Total Presets: {total}")
    print(f"Audible: {audible_count} ({audible_count/total*100:.1f}%)")
    print(f"Silent: {silent_count} ({silent_count/total*100:.1f}%)")
    print(f"Errors: {errors}")
    print("Detailed results saved to preset_silence_check.log")

if __name__ == "__main__":
    check_presets()
