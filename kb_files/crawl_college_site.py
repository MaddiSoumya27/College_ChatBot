"""
BVRITH website crawler -> clean text corpus for RAG.

Run this on your own machine/server (not in a restricted sandbox), since it
needs open internet access to bvrithyderabad.edu.in.

Install deps:
    pip install requests beautifulsoup4 tqdm

Usage:
    python crawl_college_site.py

Output:
    ./bvrith_corpus/<slug>.json   -- one file per page: {"url", "title", "text"}
    ./bvrith_corpus/all_pages.jsonl -- combined, one JSON object per line

This script:
  1. Starts at the homepage and BFS-crawls every internal link
     (same domain, bvrithyderabad.edu.in), skipping PDFs/images by default
     (PDFs are listed separately so you can OCR/parse them with a PDF lib).
  2. Strips header/nav/footer boilerplate that repeats on every page (so your
     RAG chunks are signal, not menu links).
  3. Saves clean page text + a jsonl file ready for chunking/embedding.
"""

import json
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# NOTE: the site has individual pages per faculty member (e.g. /electronics-and-
# communication-engineering/mr-mahesh-babu-katta/) and PDFs with per-student placement
# records (name, roll number, package). Faculty profile pages are fine to ingest (public
# professional bios). Per-student placement PDFs contain PII of individuals who did not
# consent to a public chatbot repeating it -- filter these out or aggregate them
# (company-wise / branch-wise totals only) before adding to your vector store.
PII_FILENAME_HINTS = ["placements-data", "2020-24-placements", "2020-2024-batch"]

BASE = "https://bvrithyderabad.edu.in/"
DOMAIN = urlparse(BASE).netloc
OUT_DIR = "bvrith_corpus"
MAX_PAGES = 400          # safety cap
REQUEST_DELAY = 0.5      # be polite to their server
TIMEOUT = 15

import os
os.makedirs(OUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; BVRITH-RAG-bot/1.0)"})

visited = set()
pdf_links = set()
queue = deque([BASE])
results = []


def is_internal(url: str) -> bool:
    p = urlparse(url)
    return (p.netloc == "" or p.netloc == DOMAIN)


def clean_url(base, href):
    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    url = urljoin(base, href)
    url = url.split("#")[0]
    if not is_internal(url):
        return None
    return url


def extract_main_text(soup: BeautifulSoup) -> str:
    # Remove obvious boilerplate / non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
        tag.decompose()

    # Elementor sites often wrap nav in elements with these classes/ids; drop repeated menu blocks
    for sel in [".elementor-nav-menu", ".site-header", ".site-footer", "#masthead",
                "#colophon", ".menu-item", ".widget_nav_menu"]:
        for el in soup.select(sel):
            el.decompose()

    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    # Deduplicate consecutive repeated lines (menus often repeat twice in the DOM)
    deduped = []
    for l in lines:
        if not deduped or deduped[-1] != l:
            deduped.append(l)
    return "\n".join(deduped)


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    return re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-")[:150]


pbar = tqdm(total=MAX_PAGES)
while queue and len(visited) < MAX_PAGES:
    url = queue.popleft()
    if url in visited:
        continue
    visited.add(url)

    if url.lower().endswith(".pdf"):
        pdf_links.add(url)
        continue

    try:
        resp = session.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            continue
    except Exception as e:
        print(f"Failed: {url} ({e})")
        continue

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    text = extract_main_text(soup)

    page = {"url": url, "title": title, "text": text}
    results.append(page)

    slug = slugify(url)
    with open(os.path.join(OUT_DIR, f"{slug}.json"), "w", encoding="utf-8") as f:
        json.dump(page, f, ensure_ascii=False, indent=2)

    # enqueue links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full = urljoin(url, href)
            if is_internal(full):
                pdf_links.add(full)
            continue
        nxt = clean_url(url, href)
        if nxt and nxt not in visited:
            queue.append(nxt)

    pbar.update(1)
    time.sleep(REQUEST_DELAY)

pbar.close()

with open(os.path.join(OUT_DIR, "all_pages.jsonl"), "w", encoding="utf-8") as f:
    for page in results:
        f.write(json.dumps(page, ensure_ascii=False) + "\n")

with open(os.path.join(OUT_DIR, "pdf_links.txt"), "w", encoding="utf-8") as f:
    for link in sorted(pdf_links):
        f.write(link + "\n")

print(f"\nDone. Crawled {len(results)} HTML pages.")
print(f"Found {len(pdf_links)} PDF documents (fee structures, syllabi, NAAC/NBA reports, etc.)")
print(f"Output written to: {OUT_DIR}/")
print("Next: parse the PDFs too (see pdf_links.txt) with a library like pypdf or pdfplumber,")
print("since colleges usually put fee structure, syllabus, and academic calendar in PDFs.")
