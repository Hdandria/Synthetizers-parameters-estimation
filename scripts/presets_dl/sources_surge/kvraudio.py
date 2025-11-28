"""KVR Audio preset downloads (requires cookies)."""
import http.cookiejar
import os
import urllib.parse as ur

import cloudscraper
from bs4 import BeautifulSoup as bs


def download(save_path, cookies_file):
    """Download archives from KVR if cookies are available."""
    if not os.path.exists(cookies_file):
        print("No KVR cookies found, skipping.")
        return []

    print("Downloading from KVR...")

    session = cloudscraper.create_scraper()
    session.cookies = http.cookiejar.MozillaCookieJar(cookies_file)

    try:
        session.cookies.load(ignore_discard=True)
    except Exception as e:
        print(f"  Failed to load cookies: {e}")
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
            print(f"  Failed to access page: {response.status_code}")
            return []

        soup = bs(response.text, 'html.parser')
        download_links = [a['href'] for a in soup.find_all('a', class_='kvrloginrequired kvronoffleft flexflexed')]
        full_links = [ur.urljoin(base_url, link) for link in download_links]

        downloaded_files = []
        for i, link in enumerate(full_links):
            filename = os.path.join(save_path, f'kvr_preset_{i}.zip')
            print(f'  Downloading archive {i+1}/{len(full_links)}...')

            response = session.get(link, headers=headers)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(filename)
            else:
                print(f'  Failed with status {response.status_code}')

        return downloaded_files
    except Exception as e:
        print(f"  Error: {e}")
        return []
