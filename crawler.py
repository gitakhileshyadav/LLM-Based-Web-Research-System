"""Crawling layer — domain intelligence, source prioritization, and dual-mode
(standard / JS-heavy) crawling with crawl4ai + aggressive content cleaning."""

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult

from config import (
    BM25_THRESHOLD,
    CRAWLER_USER_AGENT,
    JS_HEAVY_DOMAINS,
    NEWS_DOMAINS,
    NEWS_KEYWORDS,
    SCIENCE_KEYWORDS,
    SCIENTIFIC_DOMAINS,
)


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
    bm25_filter = BM25ContentFilter(
        user_query=prompt,
        bm25_threshold=BM25_THRESHOLD,
        language="english",
    )

    # ── Step 3: Markdown generator ────────────────────────────────────────────
    md_generator = DefaultMarkdownGenerator(
        content_filter=bm25_filter,
        options={
            "ignore_links":        True,
            "ignore_images":       True,
            "body_width":          0,
            "skip_internal_links": True,
            "include_sup_sub":     False,
            "mark_code":           True,    # preserve code blocks for science
        }
    )

    # ── Shared excluded tags and selectors ────────────────────────────────────
    EXCLUDED_TAGS = [
        "nav", "footer", "header", "form",
        "img", "script", "style", "aside",
        "iframe", "noscript", "figure", "figcaption",
        "button", "input", "select",
    ]

    EXCLUDED_SELECTORS = ",".join([
        "[class*='ad']",           "[class*='ads']",
        "[class*='advert']",       "[id*='ad']",
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

    # ── Step 4: Standard config — fast, for normal sites ─────────────────────
    standard_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=EXCLUDED_TAGS,
        excluded_selector=EXCLUDED_SELECTORS,
        only_text=True,
        keep_data_attributes=False,
        remove_overlay_elements=True,
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=50,
        page_timeout=30000,
        delay_before_return_html=3.0,
        user_agent=CRAWLER_USER_AGENT,
    )

    # ── Step 5: JS-heavy config — slower, full browser for news/media ─────────
    js_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=EXCLUDED_TAGS,
        excluded_selector=EXCLUDED_SELECTORS,
        only_text=False,               # ✅ must be False — JS needs full render
        keep_data_attributes=False,
        remove_overlay_elements=True,
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=30,       # lower threshold — news articles are shorter

        # ✅ Key JS-site settings
        page_timeout=45000,            # 45s for heavy JS frameworks
        delay_before_return_html=5.0,  # 5s for JS to fully render
        wait_for="css:article,main,p", # wait for content elements in DOM

        # ✅ Scroll simulation — triggers lazy-loaded content
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


    # ── Step 8: Crawl standard sites — priority first ─────────────────────────
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

        async with AsyncWebCrawler(config=standard_browser) as crawler:
            # ✅ Crawl priority URLs first one by one to preserve order
            for url in priority_standard:
                try:
                    result = await crawler.arun(url, config=standard_config)
                    all_results.append(result)
                    print(f"[Crawl] Priority done: {url} — "
                          f"{len(result.fit_markdown) if hasattr(result, 'fit_markdown') and result.fit_markdown else 0} chars")
                except Exception as e:
                    print(f"[Crawl] Priority failed: {url} — {e}")

            # ✅ Crawl remaining in batch
            if other_standard:
                results = await crawler.arun_many(other_standard, config=standard_config)
                all_results.extend(results)
                for r in results:
                    print(f"[Crawl] Standard done: {r.url} — "
                          f"{len(r.fit_markdown) if hasattr(r, 'fit_markdown') and r.fit_markdown else 0} chars")

    # ── Step 9: Crawl JS-heavy sites one at a time ────────────────────────────
    if js_urls:
        print(f"[Crawl] JS-heavy sites: {js_urls}")
        async with AsyncWebCrawler(config=js_browser) as crawler:
            for url in js_urls:
                try:
                    result = await crawler.arun(url, config=js_config)
                    all_results.append(result)
                    content_len = (
                        len(result.fit_markdown)
                        if hasattr(result, "fit_markdown") and result.fit_markdown
                        else len(result.markdown) if result.markdown else 0
                    )
                    print(f"[Crawl] JS done: {url} — {content_len} chars")

                except Exception as e:
                    print(f"[Crawl] JS failed: {url} — {e}")

    return all_results
