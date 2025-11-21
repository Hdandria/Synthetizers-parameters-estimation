"""Run a small smoke generation by calling the dataset maker programmatically.
This avoids the CLI's preset handling and allows passing `preset_path=None`.
"""
from pathlib import Path
import h5py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vst.generate_vst_dataset import make_dataset
from src.data.vst import param_specs

out = Path('outputs/smoke_vital_programmatic.h5')
out.parent.mkdir(parents=True, exist_ok=True)

# parameters for smoke test
num_samples = 1
plugin_path = 'plugins/Vital.vst3'
preset_path = None
sample_rate = 22050.0
channels = 2
velocity = 100
signal_duration_seconds = 1.0
min_loudness = -80.0
param_spec = param_specs['vital']
sample_batch_size = 1
num_workers = 1

with h5py.File(str(out), 'a') as f:
    make_dataset(
        f,
        num_samples,
        plugin_path,
        preset_path,
        sample_rate,
        channels,
        velocity,
        signal_duration_seconds,
        min_loudness,
        param_spec,
        sample_batch_size,
        num_workers,
    )

print('Smoke run finished. Output file:', out)
