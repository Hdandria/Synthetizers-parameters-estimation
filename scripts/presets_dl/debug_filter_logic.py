import json
import os

PRESET_PATH = "data/presets/vital/Psy Top 2.vital"

SAFE_WT_NAMES = {
    "Init", "Basic Shapes", "Sine", "Triangle", "Saw", "Square", "Pulse", 
    "White Noise", "Pink Noise", "Brown Noise"
}

def is_osc_workable(wt_data, level):
    print(f"    Checking Osc: Level={level}, Name='{wt_data.get('name', '')}'")
    if level == 0:
        print("    -> Level is 0, OK.")
        return True
    name = wt_data.get('name', '')
    is_safe = name in SAFE_WT_NAMES
    print(f"    -> Is Safe Name? {is_safe}")
    return is_safe

def debug_logic():
    print(f"Debugging {PRESET_PATH}...")
    with open(PRESET_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    
    settings = data.get('settings', {})
    
    wts = settings.get('wavetables', [])
    print(f"Wavetables found: {len(wts)}")
    for idx, w in enumerate(wts):
        print(f"  [{idx}] {w.get('name', 'N/A')}")
        
    # Check Oscillators
    oscs_ok = True
    for i in range(1, 4):
        level_key = f"osc_{i}_level"
        level = settings.get(level_key, 0.0)
        wts = settings.get('wavetables', [])
        wt_data = wts[i-1] if i-1 < len(wts) else {}
        
        if not is_osc_workable(wt_data, level):
            print(f"  Osc {i} failed check!")
            oscs_ok = False
            break
        else:
            print(f"  Osc {i} passed check.")
            
    print(f"Result: oscs_ok={oscs_ok}")

if __name__ == "__main__":
    debug_logic()
