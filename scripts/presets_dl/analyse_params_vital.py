import json
import os
import rootutils
from collections import defaultdict
from tqdm import tqdm
import numpy as np

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")
plugin_dir = root.joinpath('data', 'presets', 'vital').as_posix() # Contains all the .vital files
scripts_dir = root.joinpath('scripts', 'presets_dl').as_posix()

all_vital_files = []
for root_dir, dirs, files in os.walk(plugin_dir):
    for file in files:
        if file.endswith('.vital'):
            all_vital_files.append(os.path.join(root_dir, file))

print(f"Found {len(all_vital_files)} .vital files.")

params = defaultdict(list)
for vital_file in tqdm(all_vital_files, desc="Processing .vital files"):
    # vital files are simply json files
    with open(vital_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if 'settings' in data:
                for param, value in data['settings'].items():
                    if type(value) in [int, float]:
                        params[param].append(value)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {vital_file}")

# Analyze parameters
print(f"Analyzing {len(params)} parameters...")
param_stats = {}
for param, values in tqdm(params.items(), desc="Analyzing parameters"):
    unique_values = set(values)
    param_stats[param] = {
        'count': len(values),
        'unique_count': len(unique_values),
        'mean': sum(values) / len(values),
        'std': np.std(values),
        'min': min(values),
        'max': max(values)
    }

# Save analysis to a JSON file
output_file = scripts_dir + '/vital_params_analysis.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(param_stats, f, indent=4)