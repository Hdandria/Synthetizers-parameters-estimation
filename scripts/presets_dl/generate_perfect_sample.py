"""Find and generate a sample from one perfect Vital preset."""
import json
import os
import sys
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
from pedalboard import load_plugin
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.vst.core import load_preset

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESETS_DIR = "data/presets/vital"
OUTPUT_DIR = "generated_perfect_sample"

SAFE_WT_NAMES = {"Init", "Basic Shapes"}


def is_osc_workable(wt_data, osc_level):
    """Check if oscillator uses safe wavetables."""
    if osc_level == 0:
        return True
    name = wt_data.get('name', '')
    return name in SAFE_WT_NAMES


def get_category(p):
    """Categorize preset as Perfect, Good, or Unusable."""
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        settings = data.get('settings', {})

        # Check Oscillators
        oscs_ok = True
        for i in range(1, 4):
            level_key = f"oscillator_{i}_level"
            level = settings.get(level_key, 0.0)
            wts = settings.get('wavetables', [])
            wt_data = wts[i-1] if i-1 < len(wts) else {}

            if not is_osc_workable(wt_data, level):
                oscs_ok = False
                break

        # Check Sample
        sample_level = settings.get('sample_level', 0.0)
        sample_data = settings.get('sample', {})
        sample_name = sample_data.get('name', '')

        has_active_sample = sample_level > 0 and sample_name

        if oscs_ok:
            if not has_active_sample:
                return "Perfect"
            elif sample_name == "White Noise":
                return "Good"
            else:
                return "Unusable"
        else:
            return "Unusable"

    except Exception:
        return None


def generate_sample():
    """Find one perfect preset that produces audio and generate output."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading plugin from {PLUGIN_PATH}...")
    try:
        plugin = load_plugin(PLUGIN_PATH)
    except Exception as e:
        print(f"Failed to load plugin: {e}")
        return

    # Find perfect presets and try until one produces audio
    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    perfect_presets = [p for p in presets if get_category(p) == "Perfect"]

    if not perfect_presets:
        print("No perfect presets found!")
        return

    print(f"Found {len(perfect_presets)} perfect presets. Testing for audio output...")

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

    for perfect_preset in tqdm(perfect_presets):
        try:
            # Load Preset
            load_preset(plugin, str(perfect_preset))

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
                continue  # Skip silent presets, try next

            # Save Audio
            output_filename = f"{perfect_preset.stem}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            sf.write(output_path, audio.T, sample_rate)

            print(f"\nSuccess! Audio generated from preset: {perfect_preset.name}")
            print(f"Audio saved to: {output_path}")
            print(f"RMS: {rms:.6f}")
            return

        except Exception as e:
            continue  # Skip on error, try next preset

    print("No perfect preset produced audio output.")


if __name__ == "__main__":
    generate_sample()
