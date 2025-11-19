"""PresetShare preset downloads (requires cookies)."""
from bs4 import BeautifulSoup as bs
import urllib.parse as ur
import os
import http.cookiejar
import cloudscraper
import time

def download(save_path, cookies_file):
    """Download .fxp files directly from PresetShare if cookies are available."""
    if not os.path.exists(cookies_file):
        print("No PresetShare cookies found, skipping.")
        return 0

    print("Downloading from PresetShare...")

    session = cloudscraper.create_scraper()
    session.cookies = http.cookiejar.MozillaCookieJar(cookies_file)

    try:
        session.cookies.load(ignore_discard=True)
    except Exception as e:
        print(f"  Failed to load cookies: {e}")
        return 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    base_url = 'https://presetshare.com'
    first_page = ur.urljoin(base_url, 'presets?query=&instrument=2&page=1')

    try:
        response = session.get(first_page, headers=headers)
        if response.status_code != 200:
            print(f"  Failed to access page: {response.status_code}")
            return 0

        soup = bs(response.text, 'html.parser')

        # Find last page number
        last_page_li = soup.find('li', class_='last')
        if last_page_li:
            last_page_link = last_page_li.find('a')['href']
            parsed_url = ur.urlparse(last_page_link)
            query_params = ur.parse_qs(parsed_url.query)
            last_page_number = int(query_params.get('page', [1])[0])
        else:
            last_page_number = 1

        print(f"Scanning {last_page_number} pages...")

        # Download from all pages
        downloaded = 0
        for page_num in range(1, last_page_number + 1):
            page_url = ur.urljoin(base_url, f'presets?query=&instrument=2&page={page_num}')
            response = session.get(page_url, headers=headers)

            if response.status_code != 200:
                continue

            soup = bs(response.text, 'html.parser')
            download_links = [a['href'] for a in soup.find_all('a', class_='download-button')]
            full_links = [ur.urljoin(base_url, link) for link in download_links]

            for link in full_links:
                filename = os.path.join(save_path, f'presetshare_{downloaded}.fxp')

                try:
                    response = session.get(link, headers=headers)
                    if response.status_code == 200:
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
                        if downloaded % 10 == 0:
                            print(f'  Downloaded {downloaded} presets...')
                    time.sleep(0.1)  # Be polite
                except Exception as e:
                    print(f'  Error: {e}')

        print(f'  Total: {downloaded} presets')
        return downloaded
    except Exception as e:
        print(f"  Error: {e}")
        return 0
