"""Crawling layer - domain intelligence, source prioritization, and dual-mode
(standard / JS-heavy) crawling with crawl4ai + aggressive content cleaning.

Key changes vs the previous implementation:
- CacheMode.ENABLED (was BYPASS): repeated crawls of the same URL within the
  cache TTL are now instant. This matters because Streamlit re-runs the whole
  script on every interaction - previously every re-run re-crawled every URL.
- Priority URLs are crawled first (still serially - see the note at Step 8
  on why we avoid `arun_many` in this crawl4ai version).
- A single retry is attempted when crawl4ai returns empty / errored content,
  before falling through to the fetcher.py fallback layer.
- After crawl4ai finishes, any URL whose extracted text is shorter than
  _FALLBACK_MIN_CHARS is re-fetched via fetcher.fallback_fetch() (httpx +
  trafilatura + BeautifulSoup raw scrape). This is what rescues
  "unscrapable" JS sites that returned empty despite a 200.
"""

from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult

from app.config import (
    BM25_THRESHOLD,
    CRAWLER_USER_AGENT,
    CURATION_LENIENT_MAX_URLS,
    JS_HEAVY_DOMAINS,
    MAX_CRAWL_URLS,
    MAX_URLS_PER_DOMAIN,
    MIN_URL_TOKEN_OVERLAP,
    NEWS_DOMAINS,
    NEWS_KEYWORDS,
    SCIENCE_KEYWORDS,
    SCIENTIFIC_DOMAINS,
)
from app.fetcher import fallback_fetch
from app.search import tokenize

# Below this character count, the crawl4ai result is considered extractively
# failed and we hand the URL to the httpx+trafilatura fallback layer. 200
# chars is roughly one paragraph - the minimum useful unit for RAG context.
_FALLBACK_MIN_CHARS = 200

# Pure content-ready CHECK used as wait_for (no DOM mutation). SPAs like
# msn.com render their article inside OPEN shadow roots invisible to
# page.content(); the light DOM alone holds ~60-80 chars there. This polls
# for real text across BOTH the light DOM and all open shadow roots.
# csp_compliant_wait polls it every 100ms; returns False on timeout, so a
# hard bot-wall just falls through to the retry/fallback chain.
# The actual FLATTENING happens later, in the before_retrieve_html hook
# (_flatten_shadow_dom_hook) - the last possible moment before capture.
_CONTENT_READY_JS = (
    "js:() => {"
    " let textLen = document.body ? (document.body.innerText || '').length : 0;"
    " document.querySelectorAll('*').forEach(el => {"
    "   if (el.shadowRoot && el.shadowRoot.textContent)"
    "     textLen += el.shadowRoot.textContent.length;"
    " });"
    " return textLen > 500;"
    "}"
)

# Shadow-DOM flattening snippet, run via the before_retrieve_html hook right
# before page.content() capture. SPAs like msn.com render their entire article
# inside OPEN shadow roots - the light DOM holds ~60 chars while
# page.content() (which does NOT pierce shadow roots) sees an empty shell.
# This snippet walks every open shadow root depth-first and copies each
# root's innerHTML into a marked div appended to its host, so nested shadows
# flatten before their parents are serialized. crawl4ai 0.4.248 has no
# built-in flatten_shadow_dom (that landed in 0.9.x) - this replicates it.
# Runs at the LAST possible moment so an SPA's React re-render has no time
# to wipe the flattened content between flatten and capture.
#
# MUST be plain statements, NOT a bare "() => {...}" function expression:
# crawl4ai 0.4.248 wraps every js_code snippet as
#   (async () => { try { {script} return {success:true}; } ... })()
# so a function expression would only be evaluated, never invoked.
_SHADOW_FLATTEN_JS = """
const flattenShadowRoots = (root) => {
    for (const el of Array.from(root.querySelectorAll('*'))) {
        if (el.shadowRoot) {
            flattenShadowRoots(el.shadowRoot);
            const div = document.createElement('div');
            div.setAttribute('data-shadow-flattened', '1');
            div.innerHTML = el.shadowRoot.innerHTML;
            el.appendChild(div);
        }
    }
};
flattenShadowRoots(document);
"""


async def _flatten_shadow_dom_hook(page, context=None, config=None, **kwargs):
    """
    before_retrieve_html hook: flattens open shadow roots into the light DOM
    immediately before page.content() capture. Runs AFTER the wait condition
    (content is confirmed rendered) and after overlay/iframe handling - the
    last possible moment, so a re-rendering SPA has no time to wipe the
    flattened content before the HTML is captured.
    """
    try:
        await page.evaluate(_SHADOW_FLATTEN_JS)
    except Exception as e:
        print(f"[Crawl] Shadow flatten failed: {e}")


# ── Surgical patch: crawl4ai 0.4.248 unconditionally sets a
# Content-Security-Policy REQUEST header on every page (async_crawler_strategy.
# py _crawl_web, line ~1345). Real browsers never send CSP on requests - MSN's
# edge treats it as a bot signal and serves an empty JS shell. Verified: an
# identical Playwright context WITH the header polls textLen=0 forever;
# WITHOUT it the full article renders in ~3s. crawl4ai's ONLY
# set_extra_http_headers call is this CSP header, so filtering the key out
# makes it a no-op. Harmless for static sites (they worked with it anyway).
from playwright.async_api import Page

_orig_set_extra_http_headers = Page.set_extra_http_headers


async def _set_extra_http_headers_no_csp(self, headers):
    filtered = {
        k: v for k, v in (headers or {}).items()
        if k.lower() != "content-security-policy"
    }
    return await _orig_set_extra_http_headers(self, filtered)


Page.set_extra_http_headers = _set_extra_http_headers_no_csp


def detect_query_type(prompt: str) -> str:
    """
    Detect whether query is scientific, news-based, or general.
    Returns: 'science' | 'news' | 'general'
    """
    prompt_lower = prompt.lower()

    science_score = sum(1 for kw in SCIENCE_KEYWORDS if kw in prompt_lower)
    news_score    = sum(1 for kw in NEWS_KEYWORDS    if kw in prompt_lower)

    if science_score > news_score and science_score >= 1:
        return "science"
    elif news_score > science_score and news_score >= 1:
        return "news"
    else:
        return "general"


def prioritize_urls(urls: list[str], query_type: str) -> list[str]:
    """
    Reorder URLs so reputed sources come first based on query type.
    Scientific queries: arxiv, huggingface, openai blogs first.
    News queries: reuters, bbc, apnews first.
    General: Wikipedia and encyclopedic sources first.
    """
    priority_domains = {
        "science": SCIENTIFIC_DOMAINS,
        "news":    NEWS_DOMAINS,
        "general": ["wikipedia.org", "britannica.com", "worldhistory.org"],
    }.get(query_type, [])

    priority = []
    standard = []

    for url in urls:
        if any(domain in url for domain in priority_domains):
            priority.append(url)
        else:
            standard.append(url)

    reordered = priority + standard
    print(f"[Crawl] Query type: {query_type} | Priority URLs: {priority} | Standard: {standard}")
    return reordered


def is_js_heavy(url: str) -> bool:
    return any(domain in url for domain in JS_HEAVY_DOMAINS)


_TRUSTED_DOMAINS = set(
    NEWS_DOMAINS
    + SCIENTIFIC_DOMAINS
    + JS_HEAVY_DOMAINS
    + [
        "wikipedia.org", "britannica.com", "worldhistory.org",
        "newworldencyclopedia.org", "hrw.org",
    ]
)


def _is_trusted_host(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return any(d in host for d in _TRUSTED_DOMAINS)


def curate_urls(urls: list[str], prompt: str, query_type: str) -> list[str]:
    """
    Pre-crawl curation, adaptive to pool size so we never over-filter.

    MODERATE mode (pool > CURATION_LENIENT_MAX_URLS):
      - drops non-trusted root homepages (path empty or '/'),
      - drops non-trusted URLs sharing fewer than MIN_URL_TOKEN_OVERLAP
        substantive prompt tokens (e.g. impact.com matching only "impact"),
      - dedupes to MAX_URLS_PER_DOMAIN per host,
      - caps the total at MAX_CRAWL_URLS.

    LENIENT mode (pool <= CURATION_LENIENT_MAX_URLS): when engines are
    suspended and search only returned a handful of links, every URL counts.
    We keep everything except obvious non-trusted root homepages and still
    apply the per-host cap - but skip the token-overlap filter and the total
    cap. This prevents a degraded search from collapsing into a single
    useless portal URL.

    Trusted news/scientific/priority domains always pass through. Each drop
    is logged with its reason for visibility.
    """
    prompt_tokens = set(tokenize(prompt))
    lenient = len(urls) <= CURATION_LENIENT_MAX_URLS

    kept: list[str] = []
    host_counts: dict[str, int] = {}

    for url in urls:
        reason = None
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        is_root = parsed.path.rstrip("/") in ("", "/")

        if not _is_trusted_host(url) and is_root:
            reason = "root homepage"

        if reason is None and not lenient and not _is_trusted_host(url):
            url_tokens = set(tokenize(f"{host} {parsed.path}"))
            min_overlap = min(MIN_URL_TOKEN_OVERLAP, len(prompt_tokens) or 1)
            if len(prompt_tokens.intersection(url_tokens)) < min_overlap:
                reason = "low relevance"

        if reason is None and host_counts.get(host, 0) >= MAX_URLS_PER_DOMAIN:
            reason = "domain cap"

        if reason is None:
            kept.append(url)
            host_counts[host] = host_counts.get(host, 0) + 1
        else:
            print(f"[Crawl] Filtered: {url} ({reason})")

    if not lenient and len(kept) > MAX_CRAWL_URLS:
        kept = kept[:MAX_CRAWL_URLS]
        print(f"[Crawl] Capped to {MAX_CRAWL_URLS} URLs")

    print(f"[Crawl] Curation ({'lenient' if lenient else 'moderate'}): "
          f"{len(kept)} of {len(urls)} URLs kept")
    return kept


async def crawl_webpages(urls: list[str], prompt: str) -> list[CrawlResult]:
    """
    Crawl pages with aggressive cleaning for RAG-quality content.

    Features:
      - Detects query type (science/news/general) and prioritizes reputed sources
      - Separate crawler configs for standard vs JS-heavy sites
      - BM25 filtering for query-relevant sentences only
      - Removes ads, navigation, scripts, styles, and all HTML noise
    """

    # ── Step 1: Detect query type and reorder URLs ────────────────────────────
    query_type    = detect_query_type(prompt)
    urls          = prioritize_urls(urls, query_type)

    # ── Step 2: BM25 filter ───────────────────────────────────────────────────
    # An empty prompt (custom-URLs-only mode) would score everything zero and
    # drop all content — skip the filter in that case.
    md_options = {
        "ignore_links":        True,
        "ignore_images":       True,
        "body_width":          0,
        "skip_internal_links": True,
        "include_sup_sub":     False,
        "mark_code":           True,    # preserve code blocks for science
    }
    if prompt and prompt.strip():
        bm25_filter = BM25ContentFilter(
            user_query=prompt,
            bm25_threshold=BM25_THRESHOLD,
            language="english",
        )
        md_generator = DefaultMarkdownGenerator(
            content_filter=bm25_filter,
            options=md_options,
        )
    else:
        md_generator = DefaultMarkdownGenerator(options=md_options)

    # ── Shared excluded tags and selectors ────────────────────────────────────
    EXCLUDED_TAGS = [
        "nav", "footer", "header", "form",
        "img", "script", "style", "aside",
        "iframe", "noscript", "figure", "figcaption",
        "button", "input", "select",
    ]

    # ⚠️ The ad selectors are BOUNDARY-AWARE on purpose: a bare "[class*='ad']"
    # matches any class CONTAINING "ad" - which kills real content containers
    # whose names happen to include it: "header", "headline", "lead", "reader"
    # (article-reader-container!), "badge", "loaded", "advanced". Verified on
    # msn.com: "[class*='ad']" alone removed 113K chars INCLUDING the entire
    # article. Dash/token-bounded variants only match genuine ad containers.
    EXCLUDED_SELECTORS = ",".join([
        "[class~='ad']",           "[class$='-ad']",
        "[class*='-ad-']",         "[class*='ad-']",
        "[class*='advert']",       "[id~='ad']",
        "[id$='-ad']",             "[id*='-ad-']",
        "[class*='banner']",       "[class*='popup']",
        "[class*='modal']",        "[class*='overlay']",
        "[class*='cookie']",       "[class*='consent']",
        "[class*='subscribe']",    "[class*='newsletter']",
        "[class*='social']",       "[class*='share']",
        "[class*='related']",      "[class*='recommended']",
        "[class*='sidebar']",      "[class*='widget']",
        "[class*='promo']",        "[class*='sponsor']",
        "[class*='tracking']",     "[class*='analytics']",
        "[id*='google_ads']",      "[id*='taboola']",
        "[id*='outbrain']",        "[id*='disqus']",
    ])

    # ── Step 4: Standard config - fast, for normal sites ─────────────────────
    standard_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=EXCLUDED_TAGS,
        excluded_selector=EXCLUDED_SELECTORS,
        only_text=True,
        keep_data_attributes=False,
        remove_overlay_elements=True,
        # Was BYPASS - that re-crawled every URL on every Streamlit re-run.
        # ENABLED reuses the in-process cache when the same URL is requested
        # again within the session, which is the common case for this app.
        cache_mode=CacheMode.ENABLED,
        word_count_threshold=50,
        page_timeout=20000,            # was 30000 - 20s is plenty for static
        delay_before_return_html=2.0, # was 3.0 - shave 1s per page
        user_agent=CRAWLER_USER_AGENT,
    )

    # ── Step 5: JS-heavy config - slower, full browser for news/media ─────────
    js_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=EXCLUDED_TAGS,
        excluded_selector=EXCLUDED_SELECTORS,
        only_text=False,               # ✅ must be False - JS needs full render
        keep_data_attributes=False,
        # ❌ remove_overlay_elements deliberately OFF: its bundled JS removes
        # every block element with empty innerText - which kills shadow-DOM
        # flattened content (unslotted light children have empty innerText even
        # when full of text). Popup/cookie/ad classes are already stripped by
        # EXCLUDED_SELECTORS in the markdown generator, so this step is
        # redundant - and destructive for SPAs like msn.com.
        remove_overlay_elements=False,
        cache_mode=CacheMode.ENABLED,  # was BYPASS
        word_count_threshold=30,      # lower threshold - news articles are shorter

        # ✅ Key JS-site settings
        page_timeout=45000,            # 45s for heavy JS frameworks
        # 0.0 (was 5.0) - the wait condition already polls until real content
        # exists, and the shadow flatten runs in the before_retrieve_html hook
        # AFTER this delay, immediately before page.content(). Any delay here
        # only widens the window in which an SPA's React re-render can wipe
        # the flattened content before capture.
        delay_before_return_html=0.0,
        wait_for=_CONTENT_READY_JS,    # flatten shadow roots + wait for text
        # ❌ process_iframes deliberately OFF: cross-origin iframes can hang
        # 30s each in wait_for_load_state, and MSN's article lives in shadow
        # DOM, not iframes.

        # ✅ Scroll simulation BEFORE wait_for (0.4.248 runs js_code first) -
        # triggers lazy-loaded content. Shadow flattening lives INSIDE the
        # wait condition (_CONTENT_READY_JS) because js_code here fires too
        # early: SPAs haven't rendered their shadow content yet at ~1s.
        # NOTE: deliberately NO wait_until="networkidle" - SPAs keep polling
        # connections open and never go idle (times out at 45s on msn.com).
        js_code=[
            "window.scrollTo(0, document.body.scrollHeight / 2);",
            "await new Promise(r => setTimeout(r, 1500));",
            "window.scrollTo(0, document.body.scrollHeight);",
            "await new Promise(r => setTimeout(r, 1000));",
        ],
        user_agent=CRAWLER_USER_AGENT,
    )

    # ── Step 6: Browser configs ───────────────────────────────────────────────

    # Standard browser — lightweight
    standard_browser = BrowserConfig(
        headless=True,
        browser_type="chromium",
        verbose=False,
        text_mode=True,                # ✅ fast — disables images/fonts
        light_mode=True,               # ✅ fast — disables extra features
        user_agent=CRAWLER_USER_AGENT,
    )

    # JS browser — full featured for JS rendering
    js_browser = BrowserConfig(
        headless=True,
        browser_type="chromium",
        verbose=False,
        text_mode=False,               # ✅ full render needed for JS sites
        light_mode=False,              # ✅ full render needed for JS sites
        user_agent=CRAWLER_USER_AGENT,
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "DNT":             "1",
        }
    )

    # ── Step 7: Split URLs into standard and JS-heavy ─────────────────────────
    standard_urls = [u for u in urls if not is_js_heavy(u)]
    js_urls       = [u for u in urls if is_js_heavy(u)]

    all_results = []


    # ── Step 8: Crawl standard sites - priority first ─────────────────────────
    if standard_urls:
        priority_standard = [u for u in standard_urls if any(
            d in u for d in {
                "science":  SCIENTIFIC_DOMAINS,
                "news":     NEWS_DOMAINS,
                "general":  ["wikipedia.org", "worldhistory.org", "newworldencyclopedia.org"],
            }.get(query_type, [])
        )]
        other_standard = [u for u in standard_urls if u not in priority_standard]

        all_standard = priority_standard + other_standard
        print(f"[Crawl] Standard sites (ordered): {all_standard}")

        # NOTE: crawls are serial (one `arun` per URL) rather than via
        # `arun_many`. In crawl4ai 0.4.248 the dispatcher path used by
        # `arun_many` does NOT run the markdown generator's BM25 content
        # filter - every result comes back with empty `fit_markdown` while
        # the unfiltered `markdown` is intact. That silently disables the
        # query-relevance step this app depends on, so we stay on serial
        # `arun`, which applies the filter correctly.
        async with AsyncWebCrawler(config=standard_browser) as crawler:
            # Flatten shadow roots right before capture (no-op for static
            # sites without shadow roots).
            crawler.crawler_strategy.set_hook(
                "before_retrieve_html", _flatten_shadow_dom_hook
            )
            for url in all_standard:
                try:
                    result = await crawler.arun(url, config=standard_config)
                except Exception as e:
                    print(f"[Crawl] Standard failed: {url} - {e}")
                    result = None
                fixed = await _maybe_retry_and_fallback(
                    crawler, result, standard_config, prompt
                )
                all_results.append(fixed)
                _log_crawl("Standard", fixed)

    # ── Step 9: Crawl JS-heavy sites one at a time ────────────────────────────
    # JS-heavy sites are kept serial: they each spin up a full browser and
    # running them in parallel would balloon memory (each chromium is ~150MB).
    if js_urls:
        print(f"[Crawl] JS-heavy sites: {js_urls}")
        async with AsyncWebCrawler(config=js_browser) as crawler:
            # Flatten shadow roots right before capture - the last possible
            # moment, so an SPA's React re-render can't wipe the content.
            crawler.crawler_strategy.set_hook(
                "before_retrieve_html", _flatten_shadow_dom_hook
            )
            for url in js_urls:
                try:
                    result = await crawler.arun(url, config=js_config)
                except Exception as e:
                    print(f"[Crawl] JS arun failed: {url} - {e}")
                    result = None
                fixed = await _maybe_retry_and_fallback(
                    crawler, result, js_config, prompt
                )
                all_results.append(fixed)
                _log_crawl("JS", fixed)

    return all_results


# ---------------------------------------------------------------------------
# Helpers: retry, fallback bridging, logging
# ---------------------------------------------------------------------------

async def _maybe_retry_and_fallback(
    crawler: AsyncWebCrawler,
    result: CrawlResult,
    config: CrawlerRunConfig,
    prompt: str,
) -> CrawlResult:
    """
    If crawl4ai returned empty/errored content for `result.url`, retry once
    with the same config. If still below _FALLBACK_MIN_CHARS, hand off to
    fetcher.fallback_fetch() (httpx + trafilatura + BS4) and patch the
    result's markdown fields in place, so the downstream pipeline still sees
    a CrawlResult-shaped object.
    """
    current = result
    for attempt in (1, 2):
        if current is None:
            content_len = 0
        else:
            content_len = _content_len(current)
        if content_len >= _FALLBACK_MIN_CHARS:
            return current
        if attempt == 2:
            break
        print(f"[Crawl] Retry {result.url if result else '?'} - only {content_len} chars")
        try:
            current = await crawler.arun(result.url, config=config)
        except Exception as e:
            print(f"[Crawl] Retry failed {result.url} - {e}")
            current = None

    # Still empty after retry -> invoke fetcher fallback.
    url = result.url if result else "?"
    fb = await fallback_fetch(url, prompt)
    if not fb:
        return current
    print(f"[Crawl] Fallback OK {url} - {len(fb)} chars")
    return _make_fallback_result(url, fb)


def _content_len(result: CrawlResult) -> int:
    if hasattr(result, "fit_markdown") and result.fit_markdown:
        return len(result.fit_markdown)
    if getattr(result, "markdown", None):
        return len(result.markdown)
    return 0


def _log_crawl(label: str, result: CrawlResult) -> None:
    print(f"[Crawl] {label} done: {result.url} - {_content_len(result)} chars")


def _make_fallback_result(url: str, text: str) -> CrawlResult:
    """
    Build a minimal CrawlResult wrapper around plain text fetched by the
    fallback layer. We set both `markdown` and `fit_markdown` so the
    downstream code in app.py / vectordb.py that reads `result.fit_markdown`
    keeps working without a special case.
    """
    # CrawlResult is a pydantic model in crawl4ai; `html` and `success` are
    # required fields - omitting them raises a ValidationError that crashes
    # the whole crawl once a fallback fires. Construct with safe defaults.
    try:
        return CrawlResult(
            url=url,
            html="",
            success=True,
            markdown=text,
            fit_markdown=text,
            metadata={},
        )
    except Exception:
        # Older crawl4ai versions may not accept fit_markdown at construction.
        r = CrawlResult(url=url, html="", success=True, markdown=text, metadata={})
        try:
            r.fit_markdown = text
        except Exception:
            pass
        return r
