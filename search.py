"""Web search layer — SearXNG querying with sanitization, SSRF protection,
rate-limit backoff, and IP-safe async multi-search across research dimensions."""

import asyncio
import random
import re
import time
from urllib.parse import urlparse

import requests

from config import (
    ALLOWED_SCHEMES,
    BLOCKED_HOSTS,
    DISCARD_DOMAINS,
    HEADERS,
    MAX_QUERY_LENGTH,
    MAX_SEARCH_RESULTS,
    SEARXNG_INSTANCES,
)


def sanitize_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Search query must be a string")
    query = query.replace("\x00", "")
    query = re.sub(r"[;&|`$<>{}]", "", query)
    query = re.sub(r"<[^>]*>", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
    if not query:
        raise ValueError("Search query is empty after sanitization")
    return query


def is_safe_url(url: str) -> bool:
    try:
        if len(url) > 2048:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            return False
        host = parsed.hostname
        if not host:
            return False
        if any(host.startswith(b) or host == b for b in BLOCKED_HOSTS):
            return False
        if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host):
            return False
        return True
    except Exception:
        return False


def safe_parse_response(response: requests.Response) -> list[dict]:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise ValueError(f"Unexpected Content-Type: {content_type}")
    try:
        data = response.json()
    except Exception:
        raise ValueError("Response is not valid JSON")
    if not isinstance(data, dict):
        raise ValueError("JSON response is not a dictionary")
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Results field is not a list")
    unresponsive = data.get("unresponsive_engines", [])
    if unresponsive:
        print(f"[SearXNG] Unresponsive engines: {unresponsive}")
    return results


def get_web_urls(search_term: str, numresults: int = 5) -> list[str]:
    try:
        search_term = sanitize_query(search_term)
    except ValueError as e:
        print(f"[Security] Query rejected: {e}")
        return []

    for domain in DISCARD_DOMAINS:
        search_term += f" -site:{domain}"

    ENGINE_SETS = [
        "bing,brave",
        "bing,duckduckgo",
        "duckduckgo",
    ]

    params_base = {
        "format":   "json",
        "language": "en",
        "pageno":   1,
        "q":        search_term,
    }

    local     = SEARXNG_INSTANCES[:1]
    fallbacks = random.sample(SEARXNG_INSTANCES[1:], len(SEARXNG_INSTANCES[1:]))
    instances = local + fallbacks

    for instance in instances:
        search_url = f"{instance}/search"

        for engine_set in ENGINE_SETS:
            params = {**params_base, "engines": engine_set}

            for attempt in range(3):
                try:
                    response = requests.get(
                        search_url, params=params, headers=HEADERS,
                        timeout=10, allow_redirects=True, verify=True
                    )

                    if response.status_code == 429:
                        wait = 2 ** attempt
                        print(f"[SearXNG] Rate limited — waiting {wait}s")
                        time.sleep(wait)
                        continue

                    if response.status_code != 200:
                        print(f"[SearXNG] HTTP {response.status_code} — skipping")
                        break

                    try:
                        raw_results = safe_parse_response(response)
                    except ValueError as e:
                        print(f"[Security] {e}")
                        break

                    if not raw_results:
                        break

                    safe_urls = []
                    for r in raw_results:
                        candidate = r.get("url", "")
                        if not isinstance(candidate, str):
                            continue
                        if not is_safe_url(candidate):
                            continue
                        if any(d in candidate for d in DISCARD_DOMAINS):
                            continue
                        safe_urls.append(candidate)
                        if len(safe_urls) == MAX_SEARCH_RESULTS:
                            break

                    if not safe_urls:
                        break

                    time.sleep(random.uniform(1.5, 3.0))
                    return safe_urls

                except requests.exceptions.SSLError:
                    print("[Security] SSL error — blocked")
                    return []
                except requests.exceptions.Timeout:
                    time.sleep(2 ** attempt)
                except requests.exceptions.ConnectionError:
                    break
                except Exception as e:
                    print(f"[SearXNG] Unexpected error: {e}")
                    break

    print("[SearXNG] All instances exhausted.")
    return []


async def async_multi_search(
    sub_queries: list[str],
    delay_between: float = 3.0,
) -> dict[str, list[str]]:
    """
    Search all sub-queries with delay between each to avoid IP blocking.
    Returns dict mapping sub-query → list of URLs found.
    """
    results = {}

    for i, query in enumerate(sub_queries):
        print(f"[MultiSearch] Searching ({i+1}/{len(sub_queries)}): {query}")

        # ✅ Run search in thread pool — requests is not async-native
        loop = asyncio.get_event_loop()
        urls = await loop.run_in_executor(
            None,
            lambda q=query: get_web_urls(search_term=q)
        )

        results[query] = urls
        print(f"[MultiSearch] Found {len(urls)} URLs for: {query}")

        # ✅ Delay between searches — prevents IP blocking
        if i < len(sub_queries) - 1:
            wait = delay_between + random.uniform(0.5, 1.5)
            print(f"[MultiSearch] Waiting {wait:.1f}s before next search...")
            await asyncio.sleep(wait)

    return results
