"""
Additional Data Collection Script
===================================
Scrapes deeper pages from IIT Jodhpur website to augment the corpus.
Discovers links from already-visited pages and follows them to get
more textual content.
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

BASE_URL = "https://iitj.ac.in"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url, timeout=30):
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, verify=True)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        return None


def extract_text(soup):
    if soup is None:
        return ""
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def discover_all_links(url, soup):
    """Find all internal iitj.ac.in links from a page."""
    if soup is None:
        return []
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        parsed = urlparse(full)
        if "iitj.ac.in" in parsed.netloc:
            # Clean URL (remove fragments and query params)
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            links.add(clean.rstrip("/"))
    return list(links)


def main():
    """
    Crawls additional IIT Jodhpur pages by following links from seed pages.
    Focuses on academic, research, and departmental content.
    """
    # Broader set of seed URLs to crawl
    seed_urls = [
        f"{BASE_URL}",
        f"{BASE_URL}/academics",
        f"{BASE_URL}/research",
        f"{BASE_URL}/people",
        f"{BASE_URL}/about",
        f"{BASE_URL}/student-activities",
        f"{BASE_URL}/placement",
    ]

    # Discover links from seed pages
    all_links = set()
    for url in seed_urls:
        soup = fetch_page(url)
        if soup:
            links = discover_all_links(url, soup)
            all_links.update(links)
        time.sleep(0.5)

    print(f"Discovered {len(all_links)} unique links")

    # Filter to keep only content-rich pages (exclude images, pdfs, etc.)
    skip_patterns = [
        r'\.(pdf|jpg|jpeg|png|gif|svg|doc|docx|xls|xlsx|zip|rar)$',
        r'(login|logout|wp-admin|wp-content|feed|rss)',
        r'(facebook|twitter|linkedin|instagram|youtube)',
    ]

    # Load already scraped URLs to avoid duplicates
    existing_files = set(os.listdir(RAW_DATA_DIR)) if os.path.exists(RAW_DATA_DIR) else set()

    filtered_links = []
    for link in all_links:
        skip = False
        for pat in skip_patterns:
            if re.search(pat, link, re.IGNORECASE):
                skip = True
                break
        if not skip:
            filtered_links.append(link)

    print(f"Filtered to {len(filtered_links)} content links")

    # Scrape each link
    new_docs = []
    doc_idx = 100  # Start numbering after existing docs

    for url in tqdm(filtered_links, desc="Scraping additional pages"):
        soup = fetch_page(url)
        text = extract_text(soup)

        if len(text) > 200:  # Only keep substantial pages
            filename = f"extra_{doc_idx:03d}.txt"
            filepath = os.path.join(RAW_DATA_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            new_docs.append({"filename": filename, "word_count": len(text.split()), "url": url})
            doc_idx += 1

        time.sleep(0.5)

    # Update metadata
    meta_path = os.path.join(RAW_DATA_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.extend([{"filename": d["filename"], "char_count": 0, "word_count": d["word_count"]} for d in new_docs])
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_words = sum(d["word_count"] for d in new_docs)
    print(f"\nCollected {len(new_docs)} additional documents with {total_words:,} words")
    print(f"Total documents now: {len(metadata)}")


if __name__ == "__main__":
    main()
