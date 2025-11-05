import requests
from bs4 import BeautifulSoup as bs
import urllib.parse as ur
import os
import patoolib
import shutil
import rootutils
import http.cookiejar
import cloudscraper

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")
script_dir = os.path.dirname(os.path.abspath(__file__))

def get_kvr_download_links():
    """Get download links from KVR if cookies are available."""
    cookies_file = os.path.join(script_dir, 'kvr_cookies.txt')
    if not os.path.exists(cookies_file):
        print("No KVR cookies found, skipping KVR downloads.")
        return []
    
    print("Found KVR cookies, attempting to get download links...")
    
    session = cloudscraper.create_scraper()
    session.cookies = http.cookiejar.MozillaCookieJar(cookies_file)
    
    try:
        session.cookies.load(ignore_discard=True)
    except Exception as e:
        print(f"Failed to load KVR cookies: {e}")
        return []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    base_url = 'https://www.kvraudio.com'
    download_page = base_url + '/product/surge-xt-by-surge-synth-team/downloads'
    
    try:
        response = session.get(download_page, headers=headers)
        if response.status_code != 200:
            print(f"Failed to access KVR downloads page: {response.status_code}")
            return []
        
        soup = bs(response.text, 'html.parser')
        download_links = [a['href'] for a in soup.find_all('a', class_='kvronoffright flex flexcenter')]
        full_links = [ur.urljoin(base_url, link) for link in download_links]
        
        print(f"Found {len(full_links)} KVR download links.")
        return [(link, session, headers) for link in full_links]
    except Exception as e:
        print(f"Error getting KVR links: {e}")
        return []

# Regular download URLs
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

RESET = True
if RESET:
    shutil.rmtree(save_path, ignore_errors=True)
os.makedirs(save_path, exist_ok=True)

# Download all archives first
agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
headers = {'User-Agent': agent}
downloaded_files = []

print("Downloading archives...")
for url in urls:
    filename = os.path.join(save_path, url.split('/')[-1])
    try:
        print(f'Downloading {url}...')
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        downloaded_files.append(filename)
        print(f'Saved to {filename}')
    except Exception as e:
        print(f'Failed to download {url}: {e}')

# Download KVR files if cookies exist
kvr_links = get_kvr_download_links()
for i, (link, session, kvr_headers) in enumerate(kvr_links):
    filename = os.path.join(save_path, f'kvr_preset_{i}.zip')
    try:
        print(f'Downloading from KVR...')
        response = session.get(link, headers=kvr_headers)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            downloaded_files.append(filename)
            print(f'Saved to {filename}')
        else:
            print(f'Failed with status {response.status_code}')
    except Exception as e:
        print(f'Failed to download KVR file: {e}')

# Extract all archives
print("\nExtracting archives...")
total_files = 0
archive_extensions = ('.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz')

for archive_path in downloaded_files:
    if archive_path.endswith(archive_extensions):
        try:
            print(f'Extracting {os.path.basename(archive_path)}...')
            patoolib.extract_archive(archive_path, outdir=save_path)
            
            # Count extracted .fxp files
            count = 0
            for root_dir, dirs, files in os.walk(save_path):
                for file in files:
                    if file.endswith('.fxp'):
                        count += 1
            
            extracted = count - total_files
            total_files = count
            print(f'Extracted {extracted} .fxp files.')
        except Exception as e:
            print(f'Failed to extract {archive_path}: {e}')

print(f'\nTotal .fxp files: {total_files}')