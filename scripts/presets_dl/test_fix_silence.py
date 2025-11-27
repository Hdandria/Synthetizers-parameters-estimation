import os
import sys
import soundfile as sf
import mido
from pedalboard import load_plugin

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.vst.core import load_preset

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESET_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/data/presets/vital/Psy Top 2.vital"
OUTPUT_PATH = "test_fix_psy_top_2.wav"

def test_fix():
    plugin = load_plugin(PLUGIN_PATH)
    load_preset(plugin, PRESET_PATH)
    
    # Render Settings
    sample_rate = 44100
    duration = 4.0
    note_pitch = 60
    velocity = 100
    
    note_on = mido.Message("note_on", note=note_pitch, velocity=velocity).bytes()
    note_off = mido.Message("note_off", note=note_pitch, velocity=velocity).bytes()
    
    midi_messages = [
        (note_on, 0.5),
        (note_off, duration - 0.5)
    ]

    # Flush
    plugin.process([], 0.5, sample_rate, num_channels=2, reset=True)

    # APPLY FIX 1: Open Filter
    print("Applying fix: filter_1_switch = 1.0, filter_1_cutoff = 1.0")
    plugin.parameters["filter_1_switch"].raw_value = 1.0
    plugin.parameters["filter_1_cutoff"].raw_value = 1.0
    
    # Render 1
    audio1 = plugin.process(midi_messages, duration, sample_rate, num_channels=2, reset=True)
    sf.write("test_fix_open.wav", audio1.T, sample_rate)
    
    # APPLY FIX 3: Direct Out
    print("Applying fix: oscillator_1_destination = 0.0")
    plugin.parameters["oscillator_1_destination"].raw_value = 0.0
    
    # Render 3
    audio3 = plugin.process(midi_messages, duration, sample_rate, num_channels=2, reset=True)
    sf.write("test_fix_direct.wav", audio3.T, sample_rate)

if __name__ == "__main__":
    test_fix()
