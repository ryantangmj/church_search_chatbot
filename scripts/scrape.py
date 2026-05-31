import os
import requests
from bs4 import BeautifulSoup
import time
import random
import re

FULL_COOKIE_STRING = os.environ.get("SMF_COOKIE", "")

# Board 6 = Wednesday service, Board 7 = Sunday service
BOARD_IDS = [6, 7]
BASE_URL = "https://heavensbride.org/index.php"
DOWNLOAD_DIR = "sermon_documents"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Cookie": FULL_COOKIE_STRING,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://heavensbride.org/index.php"
})

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def download_file_with_retry(url, filename, retries=3, backoff_factor=30):
    for i in range(retries):
        try:
            res = session.get(url, stream=True)
            if res.status_code == 429:
                wait_time = backoff_factor * (i + 1)
                print(f"    [!] Rate limited (429). Sleeping for {wait_time}s...")
                time.sleep(wait_time)
                continue
            res.raise_for_status()
            path = os.path.join(DOWNLOAD_DIR, filename)
            with open(path, 'wb') as f:
                for chunk in res.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"    [OK] Downloaded: {filename}")
            return filename
        except Exception as e:
            print(f"    [X] Attempt {i+1} failed for {filename}: {e}")
            time.sleep(5)
    return None


def scrape_latest_from_board(board_id, already_embedded: set[str]) -> str | None:
    """
    Fetches the first (most recent) topic on board_id's first page,
    downloads its attachment only if not already in already_embedded.
    Returns the downloaded filename, or None if skipped/failed.
    """
    url = f"{BASE_URL}?board={board_id}.0"
    print(f"\n--- Board {board_id}: {url} ---")
    try:
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        if "Login" in (soup.title.string or ""):
            print("FAILED: Session invalid. Cookie may have expired.")
            return None

        links = soup.select('td.subject div span a')
        topic_urls = [a['href'].split('#')[0] for a in links if 'topic=' in a.get('href', '')]

        if not topic_urls:
            print("    [?] No topics found on this page.")
            return None

        topic_url = topic_urls[0]
        print(f"    Latest topic: {topic_url}")

        response = session.get(topic_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        attachment_div = soup.select_one('div.attachments')

        if not attachment_div:
            print("    [?] No attachments found in topic.")
            return None

        for link in attachment_div.find_all('a', href=True):
            link_text = link.get_text(strip=True).upper()
            if not link_text.startswith(('SM', 'WM')):
                continue

            raw_name = link.get_text(strip=True).split('(')[0].strip()
            file_name = re.sub(r'[\\/*?:"<>|]', "", raw_name)
            if not file_name.lower().endswith(('.docx', '.doc')):
                file_name += ".docx"

            stem = os.path.splitext(file_name)[0]
            if any(stem in s for s in already_embedded):
                print(f"    [SKIP] Already embedded: {file_name}")
                return None

            print(f"    [!] Found: {file_name}. Downloading...")
            time.sleep(random.uniform(1.0, 3.0))
            return download_file_with_retry(link['href'], file_name)

        print("    [?] No SM/WM attachment found in topic.")
        return None

    except Exception as e:
        print(f"    [!] Error on board {board_id}: {e}")
        return None


if __name__ == "__main__":
    for board_id in BOARD_IDS:
        scrape_latest_from_board(board_id, already_embedded=set())
        time.sleep(random.uniform(3.0, 6.0))
