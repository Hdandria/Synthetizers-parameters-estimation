import os
import sys
import numpy as np
import soundfile as sf
from pedalboard import load_plugin

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.vst.core import load_preset

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESET_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/data/presets/vital/Psy Top 2.vital"

def debug_preset():
    plugin = load_plugin(PLUGIN_PATH)
    print(f"Loading preset {PRESET_PATH}...")
    load_preset(plugin, PRESET_PATH)
    
    print("\n--- Parameter Inspection ---")
    params = {name: param.raw_value for name, param in plugin.parameters.items()}
    
    # Check Oscillators
    for i in range(1, 4):
        print(f"\nOscillator {i}:")
        print(f"  Level: {params.get(f'oscillator_{i}_level', 'N/A')}")
        print(f"  Destination: {params.get(f'oscillator_{i}_destination', 'N/A')}")
        print(f"  Filter Input: {params.get(f'filter_{i}_input', 'N/A')}") # Check if filter receives input
        
    # Check Filters (Generic search for keys)
    print("\nFilter Parameters:")
    for key in sorted(params.keys()):
        if "filter" in key or "switch" in key:
            print(f"  {key}: {params[key]}")
        
    # Check Master
    print("\nMaster:")
    print(f"  Volume: {params.get('volume', 'N/A')}")
    print(f"  Voice Amplitude: {params.get('voice_amplitude', 'N/A')}")
    
    # Check Envelopes
    print("\nEnvelope 1 (Amp):")
    print(f"  Attack: {params.get('envelope_1_attack', 'N/A')}")
    print(f"  Decay: {params.get('envelope_1_decay', 'N/A')}")
    print(f"  Sustain: {params.get('envelope_1_sustain', 'N/A')}")
    print(f"  Release: {params.get('envelope_1_release', 'N/A')}")

if __name__ == "__main__":
    debug_preset()
