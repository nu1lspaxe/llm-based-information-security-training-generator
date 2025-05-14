import os
import requests
import zipfile
import tarfile
import logging
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from typing import List, Tuple
from functools import lru_cache
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN", "")
API_URL = "https://api.github.com/repos/OTRF/Security-Datasets/contents/datasets"
BASE_SAVE_PATH = "./security_datasets"
MAX_THREADS = min(8, os.cpu_count() * 4)
MAX_RETRIES = 3
TIMEOUT = 15 
RATE_LIMIT_WAIT = 0.5 
BACKOFF_FACTOR = 2  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_session() -> requests.Session:
    """Create a requests session with retry configuration and authentication"""
    session = requests.Session()
    retries = Retry(total=MAX_RETRIES, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    if GITHUB_TOKEN:
        session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
    session.headers.update({"Accept": "application/vnd.github.v3+json"})
    return session

@sleep_and_retry
@limits(calls=5000, period=3600)  
@lru_cache(maxsize=1000)
def fetch_api_contents(url: str, session: requests.Session) -> List[dict]:
    """Fetch API contents with rate limit handling and caching"""
    wait_time = RATE_LIMIT_WAIT
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=TIMEOUT)
            if response.status_code == 403 and "rate limit" in response.text.lower():
                logger.warning(f"Rate limit hit. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                wait_time *= BACKOFF_FACTOR 
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch contents from {url}: {e}")
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(wait_time)
            wait_time *= BACKOFF_FACTOR
    return []

def fetch_all_archives(url: str, current_path: str = "", session: requests.Session = None) -> List[Tuple[str, str]]:
    """Recursively fetch all archive download tasks"""
    tasks = []
    contents = fetch_api_contents(url, session)

    for item in contents:
        item_name = item.get("name", "")
        item_path = os.path.join(current_path, item_name)

        if item.get("type") == "dir":
            tasks.extend(fetch_all_archives(item.get("url", ""), item_path, session))
        elif item.get("type") == "file" and item_name.lower().endswith((".zip", ".tar.gz")):
            download_url = item.get("download_url", "")
            if download_url:
                tasks.append((download_url, item_path))

    return tasks

def download_and_extract(file_url: str, save_path: str, session: requests.Session) -> bool:
    """Download and extract a single archive file"""
    try:
        response = session.get(file_url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()

        file_data = BytesIO()
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path), disable=total_size < 1024*1024) as pbar:
            for chunk in response.iter_content(chunk_size=16384): 
                file_data.write(chunk)
                pbar.update(len(chunk))

        file_data.seek(0)

        extract_path = os.path.join(BASE_SAVE_PATH, os.path.splitext(save_path)[0])
        os.makedirs(extract_path, exist_ok=True)

        if file_url.endswith(".zip"):
            try:
                with zipfile.ZipFile(file_data) as zf:
                    if any(f.flag_bits & 0x1 for f in zf.infolist()): 
                        logger.warning(f"Skipping {file_url}: ZIP is encrypted and no password provided")
                        return False
                    else:
                        zf.extractall(extract_path)
            except RuntimeError as e:
                if "encrypted" in str(e).lower():
                    logger.warning(f"Skipping {file_url}: ZIP is encrypted and no password provided")
                    return False
                raise
        elif file_url.endswith(".tar.gz"):
            with tarfile.open(fileobj=file_data, mode="r:gz") as tf:
                tf.extractall(extract_path, filter='data')

        return True

    except (requests.RequestException, zipfile.BadZipFile, tarfile.TarError) as e:
        logger.error(f"Failed to process {file_url}: {e}")
        return False

def spider():
    """Main function to orchestrate the download process"""
    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN provided. Rate limits may apply. See https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting")

    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    session = create_session()

    download_tasks = fetch_all_archives(API_URL, session=session)

    logger.info(f"Total archives found: {len(download_tasks)}")

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {
            executor.submit(download_and_extract, url, path, session): url
            for url, path in download_tasks
        }

        for future in as_completed(future_to_url):
            if future.result():
                successful += 1
            else:
                failed += 1

    logger.info(f"Download complete. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    spider()