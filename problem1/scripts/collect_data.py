"""
Data Collection Script for IIT Jodhpur Textual Data
=====================================================
This script collects textual data from multiple IIT Jodhpur sources:
1. IIT Jodhpur official website (departments, academic programs, research pages)
2. Academic regulation documents (mandatory)
3. Faculty profile pages
4. Course-related pages

The collected raw text is saved per-source as individual text files in the data/raw/ directory.
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
# Base URLs for various IIT Jodhpur web sources
BASE_URL = "https://iitj.ac.in"

# Output directory for raw scraped text
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# Headers to mimic a real browser request (avoids being blocked by the server)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Delay between requests to be polite to the server (in seconds)
REQUEST_DELAY = 1.0


def fetch_page(url, timeout=30):
    """
    Fetches the HTML content of a given URL.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds

    Returns:
        BeautifulSoup object of the page, or None if the request fails
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, verify=True)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        print(f"  [WARNING] Failed to fetch {url}: {e}")
        return None


def extract_text_from_soup(soup):
    """
    Extracts clean text from a BeautifulSoup object by:
    1. Removing script, style, nav, footer, and header tags (boilerplate)
    2. Getting the remaining visible text
    3. Cleaning up whitespace and non-ASCII characters

    Args:
        soup: BeautifulSoup object

    Returns:
        Cleaned text string
    """
    if soup is None:
        return ""

    # Remove non-content elements (scripts, styles, navigation, etc.)
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript", "iframe"]):
        tag.decompose()

    # Extract visible text from the page
    text = soup.get_text(separator=" ", strip=True)

    # Remove non-ASCII characters (handles Hindi/other language text removal)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def collect_from_urls(urls, source_name):
    """
    Collects text from a list of URLs and saves them as individual documents.

    Args:
        urls: List of URLs to scrape
        source_name: Name of the source category (used for file naming)

    Returns:
        List of (filename, text) tuples for successfully scraped pages
    """
    documents = []

    for i, url in enumerate(tqdm(urls, desc=f"Scraping {source_name}")):
        soup = fetch_page(url)
        text = extract_text_from_soup(soup)

        # Only keep documents with meaningful content (at least 100 chars)
        if len(text) > 100:
            filename = f"{source_name}_{i+1:03d}.txt"
            documents.append((filename, text))

        # Polite delay between requests to avoid overwhelming the server
        time.sleep(REQUEST_DELAY)

    return documents


def discover_links(base_url, soup, pattern=None, max_links=50):
    """
    Discovers internal links on a page that match an optional URL pattern.
    This is used to automatically find department pages, faculty pages, etc.

    Args:
        base_url: The base URL for resolving relative links
        soup: BeautifulSoup object of the page to search
        pattern: Optional regex pattern to filter URLs
        max_links: Maximum number of links to return

    Returns:
        List of absolute URLs found on the page
    """
    if soup is None:
        return []

    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)

        # Only keep links within the iitj.ac.in domain
        if "iitj.ac.in" in urlparse(full_url).netloc:
            if pattern is None or re.search(pattern, full_url):
                links.add(full_url)

        if len(links) >= max_links:
            break

    return list(links)


def collect_department_pages():
    """
    Source 1: Collects text from IIT Jodhpur department and academic program pages.
    Starts from the main academics page and follows links to individual departments.

    Returns:
        List of (filename, text) tuples
    """
    print("\n[SOURCE 1] Collecting department and academic program pages...")

    # Seed URLs covering departments, schools, and academic programs
    seed_urls = [
        f"{BASE_URL}/academics",
        f"{BASE_URL}/schools-and-departments",
        f"{BASE_URL}/department-of-computer-science-and-engineering",
        f"{BASE_URL}/department-of-electrical-engineering",
        f"{BASE_URL}/department-of-mechanical-engineering",
        f"{BASE_URL}/department-of-mathematics",
        f"{BASE_URL}/department-of-physics",
        f"{BASE_URL}/department-of-chemistry",
        f"{BASE_URL}/department-of-humanities-and-social-sciences",
        f"{BASE_URL}/department-of-bioscience-and-bioengineering",
        f"{BASE_URL}/department-of-metallurgical-and-materials-engineering",
        f"{BASE_URL}/department-of-civil-and-infrastructure-engineering",
        f"{BASE_URL}/school-of-artificial-intelligence-and-data-science",
        f"{BASE_URL}/school-of-management-and-entrepreneurship",
        f"{BASE_URL}/programs-offered",
        f"{BASE_URL}/btech-programs",
        f"{BASE_URL}/mtech-programs",
        f"{BASE_URL}/msc-programs",
        f"{BASE_URL}/phd-programs",
        f"{BASE_URL}/research",
        f"{BASE_URL}/research-and-development",
        f"{BASE_URL}/centres-of-excellence",
        f"{BASE_URL}/sponsored-research",
        f"{BASE_URL}/announcements",
    ]

    # Also try to discover more department/academic links from the main page
    main_soup = fetch_page(BASE_URL)
    if main_soup:
        extra_links = discover_links(
            BASE_URL, main_soup,
            pattern=r"(department|school|academ|program|research|centre)",
            max_links=30
        )
        seed_urls.extend(extra_links)

    # Deduplicate URLs while preserving order
    seen = set()
    unique_urls = []
    for url in seed_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return collect_from_urls(unique_urls, "department")


def collect_academic_regulations():
    """
    Source 2 (MANDATORY): Collects academic regulation documents.
    Tries to find and scrape academic regulation pages and any linked PDF content pages.

    Returns:
        List of (filename, text) tuples
    """
    print("\n[SOURCE 2] Collecting academic regulation documents...")

    # URLs where academic regulations are typically found
    regulation_urls = [
        f"{BASE_URL}/academic-regulations",
        f"{BASE_URL}/academic-regulations-for-btech",
        f"{BASE_URL}/academic-regulations-for-mtech",
        f"{BASE_URL}/academic-regulations-for-phd",
        f"{BASE_URL}/academic-regulations-for-msc",
        f"{BASE_URL}/rules-and-regulations",
        f"{BASE_URL}/ordinances-and-regulations",
        f"{BASE_URL}/examination",
        f"{BASE_URL}/examination-guidelines",
        f"{BASE_URL}/grading-system",
        f"{BASE_URL}/academic-calendar",
        f"{BASE_URL}/convocation",
        f"{BASE_URL}/semester-registration",
    ]

    # Try to discover regulation links from the academics page
    acad_soup = fetch_page(f"{BASE_URL}/academics")
    if acad_soup:
        extra = discover_links(
            f"{BASE_URL}/academics", acad_soup,
            pattern=r"(regulat|ordinance|rule|exam|grade|calendar)",
            max_links=20
        )
        regulation_urls.extend(extra)

    seen = set()
    unique_urls = []
    for url in regulation_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return collect_from_urls(unique_urls, "regulation")


def collect_faculty_profiles():
    """
    Source 3: Collects text from faculty profile pages.
    Discovers faculty listing pages and then scrapes individual profiles.

    Returns:
        List of (filename, text) tuples
    """
    print("\n[SOURCE 3] Collecting faculty profile pages...")

    # Faculty listing pages for various departments
    faculty_urls = [
        f"{BASE_URL}/faculty",
        f"{BASE_URL}/people",
        f"{BASE_URL}/cse/faculty",
        f"{BASE_URL}/ee/faculty",
        f"{BASE_URL}/me/faculty",
        f"{BASE_URL}/maths/faculty",
        f"{BASE_URL}/physics/faculty",
        f"{BASE_URL}/chemistry/faculty",
        f"{BASE_URL}/hss/faculty",
        f"{BASE_URL}/bb/faculty",
        f"{BASE_URL}/mme/faculty",
        f"{BASE_URL}/cie/faculty",
        f"{BASE_URL}/saide/faculty",
    ]

    # Try to discover individual faculty profile links
    all_faculty_pages = list(faculty_urls)
    for fac_url in faculty_urls[:5]:  # Only check first few to save time
        soup = fetch_page(fac_url)
        if soup:
            profiles = discover_links(
                fac_url, soup,
                pattern=r"(faculty|people|profile|~)",
                max_links=20
            )
            all_faculty_pages.extend(profiles)
        time.sleep(REQUEST_DELAY)

    seen = set()
    unique_urls = []
    for url in all_faculty_pages:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return collect_from_urls(unique_urls, "faculty")


def collect_course_and_misc():
    """
    Source 4: Collects text from course syllabi, newsletters, and other miscellaneous pages.

    Returns:
        List of (filename, text) tuples
    """
    print("\n[SOURCE 4] Collecting course syllabi, newsletters, and misc pages...")

    misc_urls = [
        f"{BASE_URL}/courses-offered",
        f"{BASE_URL}/course-catalogue",
        f"{BASE_URL}/curriculum",
        f"{BASE_URL}/about",
        f"{BASE_URL}/about-iit-jodhpur",
        f"{BASE_URL}/vision-and-mission",
        f"{BASE_URL}/director-message",
        f"{BASE_URL}/history",
        f"{BASE_URL}/campus",
        f"{BASE_URL}/infrastructure",
        f"{BASE_URL}/library",
        f"{BASE_URL}/placement",
        f"{BASE_URL}/placement-statistics",
        f"{BASE_URL}/student-activities",
        f"{BASE_URL}/student-clubs",
        f"{BASE_URL}/sports",
        f"{BASE_URL}/hostel",
        f"{BASE_URL}/admission",
        f"{BASE_URL}/international-relations",
        f"{BASE_URL}/alumni",
        f"{BASE_URL}/newsletter",
        f"{BASE_URL}/circulars",
        f"{BASE_URL}/notices",
        f"{BASE_URL}/achievements",
        f"{BASE_URL}/rankings",
    ]

    return collect_from_urls(misc_urls, "misc")


def save_documents(documents):
    """
    Saves all collected documents to the raw data directory.
    Each document is saved as a separate .txt file.
    Also saves a metadata JSON file with document info.

    Args:
        documents: List of (filename, text) tuples
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    metadata = []
    for filename, text in documents:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        metadata.append({
            "filename": filename,
            "char_count": len(text),
            "word_count": len(text.split())
        })

    # Save metadata for later reference
    meta_path = os.path.join(RAW_DATA_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[INFO] Saved {len(documents)} documents to {RAW_DATA_DIR}")
    total_words = sum(m["word_count"] for m in metadata)
    print(f"[INFO] Total words across all documents: {total_words:,}")


def main():
    """
    Main function that orchestrates the entire data collection pipeline.
    Collects data from all four sources and saves to disk.
    """
    print("=" * 60)
    print("IIT Jodhpur Data Collection for Word2Vec Training")
    print("=" * 60)

    all_documents = []

    # Collect from all four sources
    # Source 1: Department and academic pages
    all_documents.extend(collect_department_pages())

    # Source 2: Academic regulations (mandatory as per assignment)
    all_documents.extend(collect_academic_regulations())

    # Source 3: Faculty profiles
    all_documents.extend(collect_faculty_profiles())

    # Source 4: Course syllabi, newsletters, and misc
    all_documents.extend(collect_course_and_misc())

    # Save all collected data
    save_documents(all_documents)

    print("\n[DONE] Data collection complete!")
    print(f"Total documents collected: {len(all_documents)}")


if __name__ == "__main__":
    main()
