import rootutils

from src.data.vst import load_plugin

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

plugin = load_plugin("plugins/Vital.vst3")
with open("vital_params.txt", "w") as f:
    for name in sorted(plugin.parameters.keys()):
        f.write(f"{name}\n")
print("Parameters written to vital_params.txt")
