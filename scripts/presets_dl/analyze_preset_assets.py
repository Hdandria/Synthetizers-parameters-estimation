import os
import json
from pathlib import Path
from tqdm import tqdm

PRESETS_DIR = "data/presets/vital"

def analyze_presets():
    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    total = len(presets)
    
    uses_wavetables = 0
    uses_samples = 0
    uses_both = 0
    clean = 0
    errors = 0
    
    print(f"Analyzing {total} presets in {PRESETS_DIR}...")
    
    for p in tqdm(presets):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            settings = data.get('settings', {})
            
            has_wt = False
            if 'wavetables' in settings:
                # Check if any wavetable has actual group data
                for wt in settings['wavetables']:
                    if 'groups' in wt and wt['groups']:
                        has_wt = True
                        break
            
            has_sample = False
            if 'sample' in settings:
                # Check if sample has data
                s = settings['sample']
                if 'samples' in s and s['samples']:
                    has_sample = True
                elif 'samples_stereo' in s and s['samples_stereo']:
                    has_sample = True
            
            if has_wt and has_sample:
                uses_both += 1
            elif has_wt:
                uses_wavetables += 1
            elif has_sample:
                uses_samples += 1
            else:
                clean += 1
                
        except Exception as e:
            # print(f"Error reading {p.name}: {e}")
            errors += 1
            
    print("\nResults:")
    print(f"Total Presets: {total}")
    print(f"Clean (No custom assets): {clean} ({clean/total*100:.1f}%)")
    print(f"Uses Custom Wavetables: {uses_wavetables} ({uses_wavetables/total*100:.1f}%)")
    print(f"Uses Samples: {uses_samples} ({uses_samples/total*100:.1f}%)")
    print(f"Uses Both: {uses_both} ({uses_both/total*100:.1f}%)")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    analyze_presets()
