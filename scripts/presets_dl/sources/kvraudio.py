import requests
from bs4 import BeautifulSoup as bs
import urllib.parse as ur
import os
import patoolib
import http.cookiejar
import cloudscraper
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
cookies_file = os.path.join(script_dir, 'kvr_cookies.txt')

if not os.path.exists(cookies_file):
    print(f"Cookie file '{cookies_file}' not found. Please add your cookies first.")
    sys.exit(1)

# Setup session with cookies
session = cloudscraper.create_scraper()
session.cookies = http.cookiejar.MozillaCookieJar(cookies_file)

try:
    session.cookies.load(ignore_discard=True)
    print(f"Loaded cookies from {cookies_file}")
except Exception as e:
    print(f"Error loading cookies: {e}")
    sys.exit(1)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

# Get download links
base_url = 'https://www.kvraudio.com'
download_page = base_url + '/product/surge-xt-by-surge-synth-team/downloads'

print(f"Accessing {download_page}...")
response = session.get(download_page, headers=headers)

if response.status_code != 200:
    print(f"Failed to access page: {response.status_code}")
    sys.exit(1)

soup = bs(response.text, 'html.parser')
download_links = [a['href'] for a in soup.find_all('a', class_='kvrloginrequired kvronoffleft flexflexed')]
full_links = [ur.urljoin(base_url, link) for link in download_links]

print(f"Found {len(full_links)} download links.")

# Download all files
save_path = '.'
downloaded_files = []

for i, link in enumerate(full_links):
    filename = os.path.join(save_path, f'kvr_preset_{i}.zip')
    print(f"Downloading {link}...")
    
    try:
        response = session.get(link, headers=headers)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            downloaded_files.append(filename)
            print(f"Saved to {filename}")
        else:
            print(f"Failed with status {response.status_code}")
    except Exception as e:
        print(f"Error downloading: {e}")

# Extract all archives
print("\nExtracting archives...")
total_files = 0

for archive_path in downloaded_files:
    try:
        print(f"Extracting {archive_path}...")
        patoolib.extract_archive(archive_path, outdir=save_path)
        
        # Count .fxp files
        count = 0
        for root, dirs, files in os.walk(save_path):
            for file in files:
                if file.endswith('.fxp'):
                    count += 1
        
        extracted = count - total_files
        total_files = count
        print(f"Extracted {extracted} .fxp files.")
    except Exception as e:
        print(f"Failed to extract {archive_path}: {e}")

print(f"\nTotal .fxp files: {total_files}")