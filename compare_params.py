import rootutils

from src.data.vst.vital_param_spec import VITAL_SIMPLE_PARAM_SPEC

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

with open("vital_params.txt", "r") as f:
    plugin_params = set(line.strip() for line in f)

spec_params = set(VITAL_SIMPLE_PARAM_SPEC.synth_param_names)

missing_in_plugin = spec_params - plugin_params
missing_in_spec = plugin_params - spec_params

print(f"Missing in plugin (Spec expects these, but plugin doesn't have them): {len(missing_in_plugin)}")
for p in sorted(missing_in_plugin):
    print(f"  {p}")

print(f"\nMissing in spec (Plugin has these, but spec doesn't): {len(missing_in_spec)}")
# for p in sorted(missing_in_spec):
#     print(f"  {p}")
