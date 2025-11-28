import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

PRESETS_DIR = "data/presets/vital"
REJECTED_DIR = "data/presets/vital_rejected"

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

        # Check if ALL oscillators are silent (Level 0)
        # If so, and we don't have the sample, the result is silence -> Unusable
        all_oscs_silent = True
        for i in range(1, 4):
            if settings.get(f"osc_{i}_level", 0.0) > 0:
                all_oscs_silent = False
                break

        if all_oscs_silent:
            return "Unusable" # Silent because no oscillators and sample is skipped

        if oscs_ok:
            if not has_active_sample:
                return "Perfect"
            elif sample_name == "White Noise":
                return "Good"
            else:
                return "Unusable" # Oscs are Init, but Sample is custom/active
        else:
            return "Unusable" # Custom Wavetables

    except Exception:
        return "Error"

def filter_presets():
    os.makedirs(REJECTED_DIR, exist_ok=True)
    presets = list(Path(PRESETS_DIR).glob("*.vital"))

    kept_count = 0
    rejected_count = 0
    error_count = 0

    print(f"Filtering {len(presets)} presets...")
    print(f"Moving rejected presets to {REJECTED_DIR}")

    for p in tqdm(presets):
        cat = get_category(p)

        if cat == "Unusable" or cat == "Error":
            # Move to rejected
            dest = os.path.join(REJECTED_DIR, p.name)
            shutil.move(str(p), dest)
            if cat == "Unusable":
                rejected_count += 1
            else:
                error_count += 1
        else:
            # Keep (Perfect or Good)
            kept_count += 1

    print("\nResults:")
    print(f"Kept (Perfect/Good): {kept_count}")
    print(f"Rejected (Unusable): {rejected_count}")
    print(f"Errors (Moved to Rejected): {error_count}")

if __name__ == "__main__":
    filter_presets()
