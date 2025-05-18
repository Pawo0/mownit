import os
import json
import aiohttp
import asyncio
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from langdetect import detect, LangDetectException

START_URL = "https://edition.cnn.com/politics/president-donald-trump-47"
OUTPUT_DIR = "crawled_docs_async_2"
MAX_DOCS = 10000
CONCURRENCY = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

visited = set()
queue = asyncio.Queue()
IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip",
    ".exe", ".tar", ".gz", ".mp3", ".mp4", ".avi", ".css", ".js"
)


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
            parsed.scheme in {"http", "https"} and
            not any(parsed.path.endswith(ext) for ext in IGNORED_EXTENSIONS)
    )



def extract_data(html: str, url: str):
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
        tag.decompose()

    content_div = soup.find('body')
    if not content_div:
        return None

    text = content_div.get_text(separator='\n', strip=True)
    if len(text) < 200:
        return None

    try:
        lang = detect(text)
        if lang != 'en':
            return None
    except LangDetectException:
        return None

    title_tag = soup.find('title')
    title = title_tag.text.strip() if title_tag else "No title"

    return {
        "url": url,
        "title": title,
        "text": text
    }


async def fetch_and_process(session, url, sem, counter_lock, counter_obj, pbar):
    async with sem:
        try:
            print(f"Pobieram: {url}")
            async with session.get(url, timeout=10) as response:
                if response.status != 200 :
                    print(f"Pomijam: {url} - niewłaściwy status lub typ zawartości")
                    return
                html = await response.text()
                data = extract_data(html, url)
                if data is None:
                    print(f"Pomijam: {url} - brak przydatnej zawartości")
                    return

                async with counter_lock:
                    if counter_obj["count"] >= MAX_DOCS:
                        return
                    index = counter_obj["count"]
                    counter_obj["count"] += 1
                    pbar.update(1)

                filename = f"doc_{index:05d}.json"
                print(f"Zapisuję: {url} -> {filename} (Tytuł: {data['title']})")

                async with aiofiles.open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(data, ensure_ascii=False, indent=2))

                visited.add(url)

                soup = BeautifulSoup(html, 'html.parser')
                new_links = 0
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(url, link['href'])
                    if is_valid_url(abs_url) and abs_url not in visited:
                        visited.add(abs_url)
                        await queue.put(abs_url)
                        new_links += 1

                print(f"Znaleziono {new_links} nowych linków na stronie {url}")
        except Exception as e:
            print(f"[Błąd] {url}: {e}")


async def crawler():
    await queue.put(START_URL)
    sem = asyncio.Semaphore(CONCURRENCY)
    counter_lock = asyncio.Lock()
    counter_obj = {"count": 0}

    print(f"Rozpoczynam crawling od URL: {START_URL}")
    print(f"Planuję pobrać maksymalnie {MAX_DOCS} dokumentów")

    pbar = tqdm(total=MAX_DOCS)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        while counter_obj["count"] < MAX_DOCS:
            if queue.empty():
                print("Kolejka pusta, czekam na nowe URL...")
                await asyncio.sleep(0.1)
                continue
            url = await queue.get()
            task = asyncio.create_task(fetch_and_process(session, url, sem, counter_lock, counter_obj, pbar))
            tasks.append(task)
            # usuwaj stare zakończone zadania z listy
            tasks = [t for t in tasks if not t.done()]

        print("Osiągnięto limit dokumentów, kończę...")
        await asyncio.gather(*tasks)

    pbar.close()
    print(f"Crawling zakończony. Pobrano {counter_obj['count']} dokumentów.")


if __name__ == "__main__":
    asyncio.run(crawler())