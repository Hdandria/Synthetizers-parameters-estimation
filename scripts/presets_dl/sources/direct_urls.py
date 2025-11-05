"""Direct download URLs for Surge presets."""
import requests
import os

URLS = [
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

def download(save_path):
    """Download all archives from direct URLs."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36'
    }

    downloaded_files = []
    
    print("Downloading from direct URLs...")
    for url in URLS:
        filename = os.path.join(save_path, url.split('/')[-1])
        try:
            print(f'  Downloading {url.split("/")[-1]}...')
            with requests.get(url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            downloaded_files.append(filename)
        except Exception as e:
            print(f'  Failed: {e}')
    
    return downloaded_files
