import json
import os
import shutil
from pathlib import Path

PRESETS_DIR = "data/presets/vital"
DEST_DIR = "scripts/presets_dl"

SAFE_WT_NAMES = {"Init", "Basic Shapes"}

def is_osc_workable(wt_data, osc_level):
    if osc_level == 0:
        return True
    name = wt_data.get('name', '')
    return name in SAFE_WT_NAMES

def get_category(p):
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        settings = data.get('settings', {})

        # Check Oscillators
        oscs_ok = True
        for i in range(1, 4):
            level_key = f"oscillator_{i}_level"
            level = settings.get(level_key, 0.0)
            wts = settings.get('wavetables', [])
            wt_data = wts[i-1] if i-1 < len(wts) else {}

            if not is_osc_workable(wt_data, level):
                oscs_ok = False
                break

        # Check Sample
        sample_level = settings.get('sample_level', 0.0)
        sample_data = settings.get('sample', {})
        sample_name = sample_data.get('name', '')

        has_active_sample = sample_level > 0 and sample_name

        if oscs_ok:
            if not has_active_sample:
                return "Perfect"
            elif sample_name == "White Noise":
                return "Good"
            else:
                return "Unusable" # Oscs are Init, but Sample is custom/active (rare but possible)
        else:
            return "Unusable" # Custom Wavetables

    except Exception:
        return None

def find_and_copy():
    presets = list(Path(PRESETS_DIR).glob("*.vital"))

    found = {"Perfect": None, "Good": None, "Unusable": None}

    print("Searching for examples...")
    for p in presets:
        cat = get_category(p)
        if cat and not found[cat]:
            found[cat] = p
            print(f"Found {cat}: {p.name}")

        if all(found.values()):
            break

    print("\nCopying files...")
    for cat, p in found.items():
        if p:
            dest = os.path.join(DEST_DIR, f"example_{cat.lower()}.vital")
            shutil.copy(p, dest)
            print(f"Copied {p.name} -> {dest}")

if __name__ == "__main__":
    find_and_copy()
