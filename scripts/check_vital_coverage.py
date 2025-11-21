"""Check authoritative min/max coverage for Vital ParamSpec against vital_details.

This script loads `vital_param_spec.py` and `vital_details.py` using importlib
so we avoid importing package-level dependencies. It then computes how many
parameter names from the specs have authoritative min/max ranges in
`vital_details.get_min_max_for_plugin_name`.

Run with: `uv run python scripts/check_vital_coverage.py`
"""
from importlib import util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VITAL_PARAM_SPEC_PATH = ROOT / "src" / "data" / "vst" / "vital_param_spec.py"
VITAL_DETAILS_PATH = ROOT / "src" / "data" / "vst" / "vital_details.py"

# load modules without package imports
spec_vps = util.spec_from_file_location("vital_param_spec", str(VITAL_PARAM_SPEC_PATH))
mod_vps = util.module_from_spec(spec_vps)
import sys
sys.path.insert(0, str(ROOT))
spec_vps.loader.exec_module(mod_vps)

spec_vd = util.spec_from_file_location("vital_details", str(VITAL_DETAILS_PATH))
mod_vd = util.module_from_spec(spec_vd)
spec_vd.loader.exec_module(mod_vd)

vps = mod_vps
vd = mod_vd

names = set()
for attr in ("VITAL_SIMPLE_PARAM_SPEC", "VITAL_PARAM_SPEC"):
    if hasattr(vps, attr):
        ps = getattr(vps, attr)
        # ParamSpec exposes `names` property; fall back to `synth_params` introspection
        if hasattr(ps, "names"):
            names.update(ps.names)
        else:
            if hasattr(ps, "synth_params"):
                names.update([p.name for p in ps.synth_params])
            if hasattr(ps, "note_params"):
                names.update([p.name for p in ps.note_params])

names = sorted(names)

have = 0
missing = []
for n in names:
    if vd.get_min_max_for_plugin_name(n) is not None:
        have += 1
    else:
        missing.append(n)

print(f"Total unique parameters in specs: {len(names)}")
print(f"Have authoritative min/max: {have}")
print(f"Missing authoritative min/max: {len(missing)}")
print("\nExamples of missing parameters (first 40):")
for m in missing[:40]:
    print(" -", m)

# Exit non-zero if coverage is low to help CI detection
percent = 100.0 * have / len(names) if names else 0.0
print(f"\nCoverage: {percent:.2f}%")
if percent < 80.0:
    sys.exit(2)
