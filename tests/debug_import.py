import sys
import os
import rootutils

# Setup root like train.py
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

try:
    print("Attempting to import src.utils.callbacks...")
    import src.utils.callbacks

    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback

    traceback.print_exc()

try:
    print("\nAttempting to import src.models.surge_flow_matching_module...")
    import src.models.surge_flow_matching_module

    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()
