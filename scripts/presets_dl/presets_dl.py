import requests
from bs4 import BeautifulSoup as bs
import urllib.parse as ur
import os
import patoolib
import shutil
import rootutils

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")

def extract_from_archive(archive_path, extract_to):
    # Extract all .fxp files from the archive (all subdirectories)
    patoolib.extract_archive(archive_path, outdir=extract_to)
    i = 0
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.fxp'):
                i += 1
    return i

urls = [
    'https://demos.newloops.com/New_Loops-Surge_Presets.zip',
    'https://damon-armani.com/wp-content/uploads/Damon-Armani-Surge-Presets-Vol-2.zip',
    'https://rekkerd.org/bin/presets/inigo_kennedy_03.zip',
    'https://rekkerd.org/bin/presets/inigo_kennedy_02.zip',
    'https://rekkerd.org/bin/presets/inigo_kennedy_01.zip',
    'https://rekkerd.org/bin/presets/NICK_MORITZ_Surge_Bank_v.1.rar',
    'https://rekkerd.org/bin/presets/Bronto_Scorpio_Surge_2.zip',
    'https://rekkerd.org/bin/presets/Bronto_Scorpio_Surge.zip',
    'https://raw.githubusercontent.com/surge-synthesizer/surge-extra-content/main/Website/wiki/Additional%20Content/Philippe%20Favre%20Patches%202024.zip',
    'https://raw.githubusercontent.com/surge-synthesizer/surge-extra-content/main/Website/wiki/Additional%20Content/Psiome-Album.7z'
]

save_path = root.joinpath('data', 'presets', 'surge').as_posix()

RESET = True # delete existing files
if RESET:
    shutil.rmtree(save_path, ignore_errors=True)
os.makedirs(save_path, exist_ok=True)

agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
headers = {'User-Agent': agent}

total_files = 0
for url in urls:
    local_filename = os.path.join(save_path, url.split('/')[-1])
    try:
        print(f'Downloading {url} to {local_filename}...')
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # All archive extensions:
        extensions = ('.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz')
        if local_filename.endswith(extensions):
            num = extract_from_archive(local_filename, save_path)
            total_files += num
            print(f'Extracted {num} .fxp files from {local_filename}.')
    except Exception as e:
        print(f'Failed to download {url}. Error: {e}')
        continue

print(f'Total .fxp files extracted: {total_files}')