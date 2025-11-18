import json

import pickle
from pedalboard import Pedalboard, Reverb, load_plugin
from pedalboard.io import AudioFile
from mido import Message
from IPython.display import Audio
import json
import os
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the plugin
vital = load_plugin("/home/benjamin/Documents/work/Synthetizers-parameters-estimation/plugins/Vital.vst3")



preset_file = '/home/benjamin/Documents/work/Synthetizers-parameters-estimation/scripts/presets_dl/float.vital'
with open(preset_file, 'rb') as f:
    preset_data = json.load(f)

CORE_COMPONENT_MAP = {
    # Component Mapping (Prefixes)
    "osc": "oscillator",
    "env": "envelope",
    "random": "random_lfo",
    
    # Macro Control Mapping
    "macro_control": "macro",
    
    # Effect Mapping
    "reverb": "reverb",
    "chorus": "chorus",
    "phaser": "phaser",
    "flanger": "flanger",
    "eq": "eq",
    "delay": "delay",
    "compressor": "compressor", # Stays 'compressor' in both, though Set 2 uses specific band terms
    
    # Other Core Modules
    "sample": "sample",
    "wavetables": "wavetables",
    "lfos": "lfos",
    "modulations": "modulations"
}

COMMON_SUFFIX_MAP = {
    # On/Off
    "_on": "_switch",
    
    # Envelope Parameter Mapping
    "_attack": "_attack",
    "_decay": "_decay",
    "_sustain": "_sustain",
    "_release": "_release",
    "_hold": "_hold",
    
    # LFO Keytracking
    "_keytrack_tune": "_tune",
    "_keytrack_transpose": "_transpose",
    "_fade_time": "_fade_in",
    
    # Oscillator Specific
    "_spectral_morph": "_frequency_morph",
    "_random_phase": "_phase_randomization",
    "_frame_spread": "_frame_spread", # Note: Set 2 often uses "unison_frame_spread" for this context
    
    # Effect Mix
    "_dry_wet": "_mix",
    
    # Filter/Effect Keytracking
    "_keytrack": "_key_track",
}

preset_new_settings = {}
for key, value in tqdm(preset_data["settings"].items()):
    new_key = key
    for prefix, mapped_prefix in CORE_COMPONENT_MAP.items():
        if key.startswith(prefix):
            new_key = key.replace(prefix, mapped_prefix, 1)
            break
    for suffix, mapped_suffix in COMMON_SUFFIX_MAP.items():
        if new_key.endswith(suffix):
            new_key = new_key[:-len(suffix)] + mapped_suffix
            break
    preset_new_settings[new_key] = value

print(f"Length of vital parameters: {len(vital.parameters)}")
print(f"Length of preset settings before remapping: {len(preset_data['settings'])}")

# Print common length after remapping
common_keys_remapped = set(preset_new_settings.keys()).intersection(set(vital.parameters.keys()))
print(len(common_keys_remapped))

print("Generating sound with default settings...")
sound_raw = vital([Message("note_on", note=50), Message("note_off", note=50, time=2)],
        sample_rate=44100,
        duration=2,
        num_channels=2
)
print("Sound generated.")

print("Saving default sound to vital_raw.wav...")
print(f"Shape of sound_raw: {sound_raw.shape}") # (2, 88200) for stereo 2 seconds at 44100 Hz
# save to a WAV file
sample_rate = 44100
with AudioFile('vital_raw.wav', 'w', sample_rate, sound_raw.shape[0]) as f:
    f.write(sound_raw)
print("Loading preset settings into Vital...")

old_raw_values = {}
for key in preset_new_settings.keys():
    if key in vital.parameters:
        old_raw_values[key] = vital.parameters[key].raw_value

# load settings from preset_new_settings
success = 0
for key, value in tqdm(preset_new_settings.items()):
    if key in vital.parameters:
        if key in ['sample', 'wavetables']:
            # Skip sample and wavetables parameters
            continue
        # Cast to bool, string, int, or float as needed
        param_type = type(vital.parameters[key])
        initial_value = vital.parameters[key].raw_value
        try: # set vital.parameters[key].raw_value to value
            vital.parameters[key].raw_value = value
        except Exception as e:
            print(f"Failed to set {key} to {value}: {e}")
            continue
        # Read back to verify
        read_value = vital.parameters[key].raw_value
        if read_value == value:
            success += 1
        else:
            print(f"Verification failed for {key}: set to {value}, but read back {read_value}")
            print(f"Initial value was: {initial_value}, type: {param_type}")
print(f"Successfully set {success} parameters out of {len(preset_new_settings)}")

print("Verifying and saving parameter changes...")
# Save a csv file with the format: parameter_name, old_value, new_value
with open('vital_parameter_changes.csv', 'w') as f:
    f.write('parameter_name,old_value,new_value\n')
    for key in tqdm(preset_new_settings.keys()):
        if key in vital.parameters and key in old_raw_values:
            new_value = vital.parameters[key].raw_value
            old_value = old_raw_values[key]
            f.write(f"{key},{old_value},{new_value}\n")
print("Saved parameter changes to vital_parameter_changes.csv")


# Change a single parameter:
key = 'delay_feedback'
delay_feedback = getattr(vital, key)
# new value
delay_new = 0.1
vital.parameters[key].raw_value = delay_new
print(f"Delay Feedback after setting to {delay_new}: {vital.parameters[key].raw_value}")


print("Generating sound with loaded preset settings...")
sound = vital([Message("note_on", note=50), Message("note_off", note=50, time=2)],
        sample_rate=sample_rate,
        duration=2,
        num_channels=2
)

print("Sound generated with preset settings.")
# Save to a WAV file
with AudioFile('vital_loaded.wav', 'w', sample_rate, sound.shape[0]) as f:
    f.write(sound)

print("Saved loaded sound to vital_loaded.wav")