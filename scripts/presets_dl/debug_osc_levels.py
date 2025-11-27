import json
import random
from pathlib import Path

PRESETS_DIR = "data/presets/vital"

def debug_levels():
    presets = list(Path(PRESETS_DIR).glob("*.vital"))
    sample = random.sample(presets, 20)
    
    print(f"Inspecting 20 random presets from {PRESETS_DIR}...\n")
    
    for p in sample:
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            settings = data.get('settings', {})
            
            print(f"Preset: {p.name}")
            levels = []
            for i in range(1, 4):
                val = settings.get(f"oscillator_{i}_level", "MISSING")
                levels.append(f"Osc{i}={val} ({type(val).__name__})")
            
            print(f"  {', '.join(levels)}")
            
            # Test the logic
            all_silent = True
            for i in range(1, 4):
                val = settings.get(f"oscillator_{i}_level", 0.0)
                if val > 0:
                    all_silent = False
                    break
            print(f"  -> All Silent? {all_silent}")
            print("-" * 20)
            
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

if __name__ == "__main__":
    debug_levels()
