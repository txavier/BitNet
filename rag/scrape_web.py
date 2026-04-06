"""
Web scraper for the RAG pipeline.
Crawls a website, extracts clean text via trafilatura, and saves as MD files
ready for indexing with index_data.py.

Usage:
    # Scrape a single page
    python rag/scrape_web.py https://example.com/page -o rag/sampling_data

    # Crawl from a start URL, following links up to depth 2
    python rag/scrape_web.py https://example.com --crawl --max-depth 2 --max-pages 100

    # Multiple sites in parallel (one thread per domain)
    python rag/scrape_web.py https://site1.com/faq/ https://site2.com/docs/ --crawl --workers 2

    # Restrict crawl to a URL prefix
    python rag/scrape_web.py https://www.saabnet.com/tsn/faq/ --crawl --max-pages 50

    # Then index the scraped data
    python rag/index_data.py rag/sampling_data
"""

import argparse
import hashlib
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from trafilatura.settings import use_config

# Respect robots.txt
from urllib.robotparser import RobotFileParser


USER_AGENT = "BitNet-RAG-Scraper/1.0 (educational project)"


def get_robot_parser(base_url):
    """Load and parse robots.txt for the given base URL."""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass  # If robots.txt is unavailable, allow everything
    return rp


def is_allowed(rp, url):
    """Check if our user agent is allowed to fetch this URL."""
    try:
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def url_to_filename(url):
    """Convert a URL to a safe filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "index"
    # Truncate and add hash suffix for uniqueness
    path = re.sub(r"[^a-zA-Z0-9_\-.]", "_", path)
    if len(path) > 120:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        path = path[:120] + "_" + url_hash
    return path + ".md"


def fetch_page(url, session, timeout=15):
    """Fetch a page and return the HTML content."""
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None
        return resp.text
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def download_pdf(url, output_dir, session, timeout=30, force=False):
    """Download a PDF from a direct URL and convert to markdown."""
    try:
        from rag.pdf_handler import pdf_to_markdown
    except ImportError:
        from pdf_handler import pdf_to_markdown

    # Check if already downloaded
    parsed = urlparse(url)
    pdf_name = os.path.basename(parsed.path) or "download.pdf"
    pdf_name = re.sub(r"[^a-zA-Z0-9_\-.]" , "_", pdf_name)
    md_path = os.path.join(output_dir, os.path.splitext(pdf_name)[0] + ".md")
    if not force and os.path.exists(md_path):
        print(f"  Skipped (exists): {os.path.basename(md_path)}")
        return True

    try:
        resp = session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to download PDF {url}: {e}")
        return False

    # Save PDF to temp location, process, then clean up
    pdf_path = os.path.join(output_dir, pdf_name)

    with open(pdf_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    md_content = pdf_to_markdown(pdf_path)
    if not md_content:
        os.remove(pdf_path)
        print(f"  No text extracted from PDF: {url}")
        return False

    # Save as MD and remove the raw PDF
    md_path = os.path.splitext(pdf_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- Source: {url} -->\n\n")
        f.write(md_content)

    os.remove(pdf_path)
    char_count = len(md_content)
    print(f"  Saved PDF: {os.path.basename(md_path)} ({char_count} chars)")
    return True


def extract_text(html, url):
    """Extract clean text from HTML using trafilatura."""
    config = use_config()
    config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
    config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "50")

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_recall=True,
        config=config,
    )
    return text


def extract_links(html, base_url, prefix):
    """Extract same-site links from HTML that match the URL prefix."""
    links = set()
    try:
        from trafilatura import extract_metadata
        # Fall back to basic link extraction via regex for reliability
        for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = match.group(1)
            absolute = urljoin(base_url, href)
            # Strip fragment
            absolute = absolute.split("#")[0]
            # Only follow links under the prefix
            if absolute.startswith(prefix) and absolute != base_url:
                # Skip non-HTML resources (but keep PDFs)
                path = urlparse(absolute).path.lower()
                skip_ext = (".jpg", ".jpeg", ".png", ".gif", ".zip",
                            ".css", ".js", ".xml", ".ico", ".svg", ".mp3",
                            ".mp4", ".avi", ".mov", ".webp", ".woff", ".ttf")
                if not any(path.endswith(ext) for ext in skip_ext):
                    links.add(absolute)
    except Exception:
        pass
    return links


def extract_gdrive_ids(html):
    """Extract Google Drive file IDs from links in HTML.

    Handles these URL patterns:
      - drive.google.com/file/d/FILE_ID/view
      - drive.google.com/open?id=FILE_ID
      - docs.google.com/document/d/FILE_ID/...
    """
    ids = set()
    for match in re.finditer(
        r'https?://(?:drive|docs)\.google\.com/'
        r'(?:file/d/|open\?id=|document/d/)'
        r'([a-zA-Z0-9_-]+)',
        html,
    ):
        ids.add(match.group(1))
    return ids


def download_gdrive_pdf(file_id, output_dir, session, timeout=120, force=False):
    """Download a file from Google Drive by ID and process as PDF."""
    try:
        from rag.pdf_handler import pdf_to_markdown
    except ImportError:
        from pdf_handler import pdf_to_markdown

    # Check if already downloaded
    md_path = os.path.join(output_dir, f"gdrive_{file_id}.md")
    if not force and os.path.exists(md_path):
        print(f"    Skipped (exists): gdrive_{file_id}.md")
        return True

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        resp = session.get(download_url, timeout=timeout, stream=True, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    Failed to download GDrive {file_id}: {e}")
        return False

    # Google Drive may show a virus scan confirmation page for large files
    content_type = resp.headers.get("content-type", "")
    if "text/html" in content_type:
        confirm_match = re.search(
            r'href="(/uc\?export=download&amp;confirm=[^"]+&amp;id='
            + re.escape(file_id) + r'[^"]*)"',
            resp.text,
        )
        if not confirm_match:
            confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)&', resp.text)
            if confirm_match:
                confirm_url = (
                    f"https://drive.google.com/uc?export=download"
                    f"&confirm={confirm_match.group(1)}&id={file_id}"
                )
            else:
                print(f"    GDrive {file_id}: got HTML instead of file (may need manual download)")
                return False
        else:
            confirm_url = "https://drive.google.com" + confirm_match.group(1).replace("&amp;", "&")

        try:
            resp = session.get(confirm_url, timeout=timeout, stream=True, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    Failed to confirm GDrive download {file_id}: {e}")
            return False

    # Save to temp file
    pdf_name = f"gdrive_{file_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)

    bytes_written = 0
    with open(pdf_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bytes_written += len(chunk)

    if bytes_written == 0:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        print(f"    GDrive {file_id}: empty response (rate-limited or unavailable)")
        return False

    # Verify it's actually a PDF
    with open(pdf_path, "rb") as f:
        header = f.read(8)

    if not header.startswith(b"%PDF"):
        os.remove(pdf_path)
        print(f"    GDrive {file_id}: not a PDF (could be a doc, spreadsheet, or blocked)")
        return False

    md_content = pdf_to_markdown(pdf_path)
    if not md_content:
        os.remove(pdf_path)
        print(f"    GDrive {file_id}: no text extracted")
        return False

    md_path = os.path.splitext(pdf_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- Source: Google Drive file {file_id} -->\n\n")
        f.write(md_content)

    os.remove(pdf_path)
    char_count = len(md_content)
    print(f"    Saved GDrive PDF: {os.path.basename(md_path)} ({char_count} chars)")
    return True


def scrape_single(url, output_dir, session, force=False):
    """Scrape a single URL and save as markdown."""
    filename = url_to_filename(url)
    filepath = os.path.join(output_dir, filename)

    if not force and os.path.exists(filepath):
        print(f"  Skipped (exists): {filename}")
        return True

    html = fetch_page(url, session)
    if not html:
        return False

    text = extract_text(html, url)
    if not text or len(text.strip()) < 50:
        print(f"  No meaningful content extracted from {url}")
        return False

    # Write as markdown with source URL header
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {url}\n\n")
        f.write(text)
        f.write("\n")

    print(f"  Saved: {filename} ({len(text)} chars)")
    return True


def crawl(start_url, output_dir, max_depth=2, max_pages=50, delay=1.0, force=False):
    """Crawl from a start URL, extracting and saving content."""
    parsed = urlparse(start_url)
    prefix = start_url.rstrip("/")
    # If the start URL points to a specific page, use its directory as prefix
    if "." in parsed.path.split("/")[-1]:
        prefix = start_url.rsplit("/", 1)[0]

    print(f"Crawl prefix: {prefix}")
    print(f"Max depth: {max_depth}, Max pages: {max_pages}, Delay: {delay}s")

    rp = get_robot_parser(start_url)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    visited = set()
    queue = deque([(start_url, 0)])  # (url, depth)
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        if not is_allowed(rp, url):
            print(f"  Blocked by robots.txt: {url}")
            continue

        print(f"[{saved + 1}/{max_pages}] Depth {depth}: {url}")

        # Handle PDF links
        if urlparse(url).path.lower().endswith(".pdf"):
            if download_pdf(url, output_dir, session, force=force):
                saved += 1
            time.sleep(delay)
            continue

        html = fetch_page(url, session)
        if not html:
            continue

        # Extract and save text
        text = extract_text(html, url)
        if text and len(text.strip()) >= 50:
            filename = url_to_filename(url)
            filepath = os.path.join(output_dir, filename)
            if not force and os.path.exists(filepath):
                print(f"  Skipped (exists): {filename}")
                saved += 1
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {url}\n\n")
                    f.write(text)
                    f.write("\n")
                print(f"  Saved: {filename} ({len(text)} chars)")
                saved += 1

        # Discover new links if within depth limit
        if depth < max_depth:
            new_links = extract_links(html, url, prefix)
            for link in sorted(new_links):
                if link not in visited:
                    queue.append((link, depth + 1))

        # Download any Google Drive PDFs linked on this page
        gdrive_ids = extract_gdrive_ids(html)
        if gdrive_ids:
            print(f"  Found {len(gdrive_ids)} Google Drive link(s)")
            for gid in sorted(gdrive_ids):
                if gid in visited:
                    continue
                visited.add(gid)
                if saved >= max_pages:
                    break
                if download_gdrive_pdf(gid, output_dir, session, force=force):
                    saved += 1
                time.sleep(delay)

        # Polite delay between requests
        time.sleep(delay)

    print(f"\nDone. Saved {saved} pages to {output_dir}/")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Scrape website content for RAG indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single page
  python rag/scrape_web.py https://www.saabnet.com/tsn/faq/c900.html

  # Crawl FAQ section
  python rag/scrape_web.py https://www.saabnet.com/tsn/faq/ --crawl --max-pages 50

  # Multiple sites in parallel (one thread per domain)
  python rag/scrape_web.py https://www.saabnet.com/tsn/faq/ https://www.saabplanet.com/tech/ --crawl --max-pages 50 --workers 2

  # Then index
  python rag/index_data.py rag/sampling_data
        """,
    )
    parser.add_argument("urls", nargs="+", help="URL(s) to scrape (or starting URLs for crawl)")
    parser.add_argument("-o", "--output", default="rag/sampling_data",
                        help="Output directory for scraped MD files (default: rag/sampling_data)")
    parser.add_argument("--crawl", action="store_true",
                        help="Crawl and follow links from each start URL")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Max link-following depth when crawling (default: 2)")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages to scrape per URL (default: 50)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files even if they already exist")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for multi-URL scraping (default: 1). "
                             "Each domain gets its own thread with its own rate limiting.")
    args = parser.parse_args()

    # Validate URLs
    for url in args.urls:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            print(f"Error: Invalid URL: {url}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    if len(args.urls) == 1:
        # Single URL — original behavior
        url = args.urls[0]
        if args.crawl:
            crawl(url, args.output, args.max_depth, args.max_pages, args.delay, args.force)
        else:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            rp = get_robot_parser(url)
            if not is_allowed(rp, url):
                print(f"Blocked by robots.txt: {url}")
                sys.exit(1)
            if scrape_single(url, args.output, session, args.force):
                print(f"\nDone. Now index with: python rag/index_data.py {args.output}")
            else:
                print("No content extracted.")
                sys.exit(1)
    else:
        # Multiple URLs — parallel by domain
        workers = min(args.workers or len(args.urls), len(args.urls))
        print(f"Scraping {len(args.urls)} URL(s) with {workers} worker(s)")

        def _scrape_one(url):
            domain = urlparse(url).netloc
            if args.crawl:
                count = crawl(url, args.output, args.max_depth, args.max_pages, args.delay, args.force)
                return domain, count
            else:
                session = requests.Session()
                session.headers.update({"User-Agent": USER_AGENT})
                rp = get_robot_parser(url)
                if not is_allowed(rp, url):
                    print(f"Blocked by robots.txt: {url}")
                    return domain, 0
                ok = scrape_single(url, args.output, session, args.force)
                return domain, 1 if ok else 0

        total = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_scrape_one, url): url for url in args.urls}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    domain, count = future.result()
                    total += count
                    print(f"\n[{domain}] Finished: {count} pages saved")
                except Exception as e:
                    print(f"\n[{url}] Error: {e}")

        print(f"\nAll done. {total} total pages saved to {args.output}/")
        print(f"Index with: python rag/index_data.py {args.output}")


if __name__ == "__main__":
    main()
