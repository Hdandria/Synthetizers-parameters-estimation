import os
import sys
from pedalboard import load_plugin

PLUGIN_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3"
PRESET_PATH = "/home/benjamin/Documents/work/Synthetizers-parameters-estimation/scripts/presets_dl/piano.vital"

try:
    plugin = load_plugin(PLUGIN_PATH)
    print("Plugin loaded.")
    plugin.load_preset(PRESET_PATH)
    print("Native load_preset succeeded!")
except Exception as e:
    print(f"Native load_preset failed: {e}")
