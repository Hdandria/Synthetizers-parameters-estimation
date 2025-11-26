import os
import json
from pathlib import Path
from tqdm import tqdm

PRESETS_DIR = "data/presets/vital"

SAFE_WT_NAMES = {"Init", "Basic Shapes"}
SAFE_SAMPLE_NAMES = set() # Currently none, as we can't load samples

def is_osc_workable(wt_data, osc_level):
    if osc_level == 0:
        return True
    name = wt_data.get('name', '')
    return name in SAFE_WT_NAMES

def is_sample_workable(sample_data, sample_level):
    if sample_level == 0:
        return True
    name = sample_data.get('name', '')
    return False # Strict: no samples supported

def analyze_workable():
    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    
    fully_workable = 0
    workable_no_noise = 0
    total = len(presets)
    
    print(f"Analyzing {total} presets...")
    
    for p in tqdm(presets):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            settings = data.get('settings', {})
            
            # Check Oscillators
            oscs_ok = True
            for i in range(1, 4):
                level_key = f"oscillator_{i}_level"
                level = settings.get(level_key, 0.0)
                
                # Get WT data (index i-1)
                wts = settings.get('wavetables', [])
                wt_data = wts[i-1] if i-1 < len(wts) else {}
                
                if not is_osc_workable(wt_data, level):
                    oscs_ok = False
                    break
            
            if not oscs_ok:
                continue
                
            # Check Sample
            sample_level = settings.get('sample_level', 0.0)
            sample_data = settings.get('sample', {})
            
            if is_sample_workable(sample_data, sample_level):
                fully_workable += 1
                workable_no_noise += 1 # Fully workable is also workable-no-noise
            else:
                # Check if it's just White Noise
                name = sample_data.get('name', '')
                if name == "White Noise":
                    workable_no_noise += 1
                    
        except Exception:
            pass
            
    print("\nResults:")
    print(f"Total Presets: {total}")
    print(f"Fully Workable (Init/Basic Shapes, No Active Samples): {fully_workable} ({fully_workable/total*100:.1f}%)")
    print(f"Workable if missing Noise (Init/Basic Shapes + White Noise): {workable_no_noise} ({workable_no_noise/total*100:.1f}%)")

if __name__ == "__main__":
    analyze_workable()
