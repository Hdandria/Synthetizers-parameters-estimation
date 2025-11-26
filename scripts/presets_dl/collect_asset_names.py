import os
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PRESETS_DIR = "data/presets/vital"

def collect_names():
    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    wt_names = Counter()
    sample_names = Counter()
    
    print(f"Scanning {len(presets)} presets for asset names...")
    
    for p in tqdm(presets):
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            settings = data.get('settings', {})
            
            # Collect WT names
            if 'wavetables' in settings:
                for wt in settings['wavetables']:
                    name = wt.get('name', 'Unknown')
                    wt_names[name] += 1
            
            # Collect Sample name
            if 'sample' in settings:
                name = settings['sample'].get('name', 'None')
                sample_names[name] += 1
                
        except Exception:
            pass
            
    print("\nTop 20 Wavetable Names:")
    for name, count in wt_names.most_common(20):
        print(f"{name}: {count}")
        
    print("\nTop 20 Sample Names:")
    for name, count in sample_names.most_common(20):
        print(f"{name}: {count}")

if __name__ == "__main__":
    collect_names()
