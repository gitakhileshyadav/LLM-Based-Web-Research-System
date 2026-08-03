"""Fallback fetcher layer.

When crawl4ai returns empty / errored content for a URL (which happens often
with JS-heavy news sites, paywalled pages, or sites that block headless
chromium's automation fingerprint), this module attempts a much cheaper
extraction:

  1. httpx GET with a real-browser User-Agent (no JS execution, ~1-3s).
  2. Pass the HTML to trafilatura, which is purpose-built for extracting
     the main article text from news/blog/research pages and discarding
     nav/ads/sidebars/boilerplate. This recovers a clean text blob even
     from pages crawl4ai's BM25ContentFilter rejected.
  3. If trafilatura returns nothing (rare; usually happens on
     list-of-links pages with no main article), fall back to a naive
     BeautifulSoup <article>/<main>/<p> scrape. We keep only paragraphs
     longer than a small threshold to drop menu items.

The returned text is plain text (no markdown), which is fine for RAG
chunking - ChromaDB chunks on whitespace anyway and the LLM context is
built from raw excerpts in app.py.
"""

import asyncio

from bs4 import BeautifulSoup

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

from config import CRAWLER_USER_AGENT

# Hard cap on fetch time. We’re a fallback, not the primary path; if the
# server hasn't even sent the body in 10s we give up and return None so the
# caller can drop the URL from the corpus.
_FETCH_TIMEOUT = 10.0

# httpx follows up to 5 redirects (enough for shorteners + http->https
# upgrades) but won't follow infinite redirect loops.
_MAX_REDIRECTS = 5

# Minimum paragraph length kept by the BeautifulSoup scrape step.
# 80 chars weeds out "Skip to content", "Cookie settings", menu labels, etc.
_MIN_PARA_LEN = 80


async def fallback_fetch(url: str, query: str | None = None) -> str | None:
    """
    Try hard to extract main article text from `url` without a browser.
    Returns the extracted text, or None if we couldn't get anything usable.

    `query` is accepted for API symmetry but currently unused - trafilatura's
    own extraction already focuses on the main content; we don't need to
    bias it with the query here (the BM25 step in crawler.py already had its
    shot upstream).
    """
    if not _HAS_HTTPX:
        print("[Fallback] httpx not installed - skipping")
        return None

    try:
        html = await _fetch_html(url)
    except Exception as e:
        print(f"[Fallback] httpx failed {url} - {e}")
        return None

    if not html:
        return None

    text = _extract_with_trafilatura(html)
    if text and len(text) >= _MIN_PARA_LEN:
        return text

    text = _extract_with_bs4(html)
    if text and len(text) >= _MIN_PARA_LEN:
        return text

    return None


async def _fetch_html(url: str) -> str:
    """
    Synchronous httpx wrapped to run in a thread (httpx.AsyncClient is also
    fine, but we want a single shared client config and easy cancellation).

    Runs via asyncio.to_thread when the loop is healthy; if the event loop is
    shutting down (happens when Streamlit tears down a run mid-crawl),
    asyncio.to_thread raises "cannot schedule new futures after shutdown" and
    the crawl silently loses every page. In that case we fall back to a
    direct blocking call - this is the last-resort fallback path, so a few
    seconds of blocking is acceptable.
    """
    headers = {
        "User-Agent": CRAWLER_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
    }
    try:
        return await asyncio.to_thread(_fetch_html_sync, url, headers)
    except RuntimeError:
        return _fetch_html_sync(url, headers)


def _fetch_html_sync(url: str, headers: dict) -> str:
    # verify=True (default) - we never bypass TLS for SSRF-safe fetches.
    # follow_redirects=True so shorteners resolve; httpx caps at _MAX_REDIRECTS.
    with httpx.Client(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        max_redirects=_MAX_REDIRECTS,
        verify=True,
    ) as client:
        resp = client.get(url, headers=headers)
        # Don't raise_for_status - some sites return 403/429 with usable
        # body text, and we'd rather try to extract from what we got.
        if resp.status_code >= 500:
            return ""
        # Decode defensively; servers sometimes lie about charset.
        return resp.text


def _extract_with_trafilatura(html: str) -> str | None:
    if not _HAS_TRAFILATURA:
        return None
    try:
        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,    # err on side of clean text, not recall
            no_fallback=False,        # let trafilatura's own rules run
        )
    except Exception as e:
        print(f"[Fallback] trafilatura error - {e}")
        return None


def _extract_with_bs4(html: str) -> str | None:
    """
    Naive but robust: grab text from <article>, <main>, or all <p> tags,
    keep only paragraphs >= _MIN_PARA_LEN, join with double newlines so the
    downstream chunker has natural boundaries.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in ("script", "style", "nav", "footer", "header", "aside", "form"):
            for el in soup.find_all(tag):
                el.decompose()

        container = soup.find("article") or soup.find("main") or soup.body or soup
        if container is None:
            return None
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= _MIN_PARA_LEN
        ]
        if not paragraphs:
            return None
        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[Fallback] BeautifulSoup error - {e}")
        return None
