"""Download Surge presets from multiple sources."""
import os
import shutil

import patoolib
import rootutils
from sources_surge import direct_urls, kvraudio, presetshare

# Setup paths
root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = root.joinpath('data', 'presets', 'surge').as_posix()

# Clean start
RESET = True
if RESET:
    shutil.rmtree(save_path, ignore_errors=True)
os.makedirs(save_path, exist_ok=True)

# Download archives from all sources
downloaded_archives = []
downloaded_archives.extend(direct_urls.download(save_path))
downloaded_archives.extend(kvraudio.download(save_path, os.path.join(script_dir, 'kvr_cookies.txt')))

# Extract all archives
print("\nExtracting archives...")
total_fxp = 0
for archive in downloaded_archives:
    if archive.endswith(('.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz')):
        try:
            print(f'  Extracting {os.path.basename(archive)}...')
            patoolib.extract_archive(archive, outdir=save_path)

            # Count .fxp files
            count = sum(1 for root, dirs, files in os.walk(save_path)
                       for file in files if file.endswith('.fxp'))
            extracted = count - total_fxp
            total_fxp = count
            print(f'  Found {extracted} .fxp files')
        except Exception as e:
            print(f'  Failed: {e}')

# Download .fxp files directly from PresetShare
# Create a presetshare folder if it doesn't exist
presetshare_path = os.path.join(save_path, 'presetshare')
os.makedirs(presetshare_path, exist_ok=True)
total_fxp += presetshare.download(presetshare_path, os.path.join(script_dir, 'presetshare_cookies.txt'))

print(f'\n=== Total: {total_fxp} .fxp files ===')
