import json
import os

INPUT_PATH = "scripts/presets_dl/piano.vital"
OUTPUT_PATH = "scripts/presets_dl/piano_no_assets.vital"

def strip_preset():
    with open(INPUT_PATH, 'r') as f:
        data = json.load(f)
    
    if 'settings' in data:
        # Strip wavetables content but keep entries
        if 'wavetables' in data['settings']:
            print("Stripping wavetables content...")
            for wt in data['settings']['wavetables']:
                if 'groups' in wt:
                    wt['groups'] = []
            
        # Strip sample data but keep metadata
        if 'sample' in data['settings']:
            print("Stripping sample data...")
            if 'samples' in data['settings']['sample']:
                data['settings']['sample']['samples'] = ""
            if 'samples_stereo' in data['settings']['sample']:
                data['settings']['sample']['samples_stereo'] = ""
            
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Created stripped preset at {OUTPUT_PATH}")

if __name__ == "__main__":
    strip_preset()
