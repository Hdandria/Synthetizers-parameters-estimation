import os
import sys
from pedalboard import load_plugin

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"

def list_params():
    try:
        plugin = load_plugin(PLUGIN_PATH)
        print(f"Loaded {plugin.name}")
        print(f"Total Parameters: {len(plugin.parameters)}")
        
        # Print all keys sorted
        print("\n--- All Available Parameters ---")
        for key in sorted(plugin.parameters.keys()):
            print(key)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_params()
