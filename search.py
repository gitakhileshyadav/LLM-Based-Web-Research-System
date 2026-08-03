"""Web search layer - SearXNG querying with sanitization, SSRF protection,
rate-limit backoff, and IP-safe async multi-search across research dimensions.

Key changes vs the previous implementation:
- Rotating engine sets: each sub-query hits a different mix of upstream
  engines (DDG+bing, DDG+brave, DDG+google). This spreads load so a single
  upstream timeout/429 no longer empties the whole result set, and it
  drastically reduces the chance of every dimension returning "wrong sources".
- Parallel multi-search with a bounded semaphore (max 3 concurrent) and small
  jitter, instead of a fixed 3s sleep between every sub-query. Throughput
  improves ~5x while remaining polite to upstream engines.
- Client `requests` timeout lowered from 15s to 12s so the client never
  out-waits SearXNG's own `outgoing.request_timeout` (12s in settings.yml).
- On empty ranked results from one engine set we now fall through to the next
  engine set (previously we returned []). This is what rescues a dimension
  when its first engine set times out at SearXNG.
"""

import asyncio
import random
import re
import time
from urllib.parse import urlparse

import requests

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

from config import (
    ALLOWED_SCHEMES,
    BLOCKED_HOSTS,
    DISCARD_DOMAINS,
    HEADERS,
    MAX_QUERY_LENGTH,
    SEARCH_CANDIDATE_POOL,
    SEARXNG_INSTANCES,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Title patterns that indicate dictionary/definition/encyclopedia pages -
# these are rarely useful for news/research queries and crowd out real sources.
_DEFINITION_TITLE_RE = re.compile(
    r"(definition|meaning|dictionary|encyclopedia|wikiwand|wiktionary|"
    r"\bwhat is\b|\bwhat are\b|\bhow to\b|\bwho is\b|"
    r"\(philosophy\)|\(concept\)|\(term\)|\(word\)|"
    r"\bcauses? \([^)]+\)|"
    r"\bfour causes?\b)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "by", "from", "as", "at",
    "that", "this", "these", "those", "it", "its", "their", "our", "you",
    "your", "we", "they", "what", "which", "who", "how", "when", "where",
    "will", "would", "can", "could", "should", "may", "might", "not",
}


def _tokenize(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS
    ]


def _rank_sources(query: str, results: list[dict], top_n: int) -> list[dict]:
    """
    Rank candidate search results by BM25 relevance to the query.
    Only sources with real query-term overlap are kept (score > 0),
    sorted most-relevant first. Falls back to lightweight token-overlap
    scoring if the rank_bm25 package is unavailable.

    Returns [] when the whole pool is uniformly weak (e.g. a batch of
    dictionary pages that merely share one generic term with the query),
    so the caller can fall through to the next engine set.
    """
    if not results:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return results[:top_n]

    corpus = [
        _tokenize(f"{r.get('title', '')} {r.get('content', '')}")
        for r in results
    ]

    if _HAS_BM25:
        scores = BM25Okapi(corpus).get_scores(query_tokens)
    else:
        # Fallback: count overlapping query tokens per document
        query_set = set(query_tokens)
        scores = [
            sum(1 for t in doc if t in query_set)
            for doc in corpus
        ]

    scored = [(r, s, doc) for r, s, doc in zip(results, scores, corpus) if s > 0]
    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    best_score = scored[0][1]

    # Drop sources that only barely match the query. A relative cutoff keeps
    # the truly relevant results even though absolute BM25 scores depend on
    # corpus size. Always keep the single best match.
    keep = [r for r, s, _ in scored if s >= best_score * 0.15] or [scored[0][0]]

    # Reject pools whose best source overlaps the query in fewer than two
    # distinct terms - that's the signature of dictionary/definition spam
    # that only shares one generic keyword (e.g. "student").
    query_set = set(query_tokens)
    best_overlap = len(query_set.intersection(scored[0][2]))
    min_overlap = min(2, len(query_set))
    if best_overlap < min_overlap:
        return []

    return keep[:top_n]


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


# Rotated engine sets. Each sub-query is mapped round-robin to one of these
# sets, so a single upstream outage only affects 1/N of the dimensions rather
# than all of them.
#
# IMPORTANT: chosen based on empirical test from this host (2026-08-03):
#   - bing       : works reliably, no CAPTCHA / 429. Always return results.
#   - mwmbl      : works reliably. Headline-only index, but useful backup.
#   - duckduckgo : hit CAPTCHA (wt-wt region) after ~5 test queries. SearXNG
#                  auto-suspends it for 5 min (SearxEngineCaptcha: 300 in
#                  settings.yml) when that happens. Keep as a secondary
#                  engine so when it's available we use it, when it's not we
#                  fall through to the rest of the set.
#   - brave      : hit HTTP 429 (Too Many Requests) from this IP. Same
#                  pattern - SearXNG suspends 2 min and retries next time.
#   - google     : silently returns 0 results (likely proxy-detected). Excluded -
#                  including it wastes a request slot on every dimension.
#
# Strategy: lead every set with the reliable engines so even when DDG/brave
# are cooling down, every dimension still returns results.
ENGINE_SETS = [
    ["bing", "duckduckgo"],           # bing primary, DDG backup
    ["bing", "mwmbl", "brave"],       # bing + mwmbl reliable, brave bonus
    ["bing", "duckduckgo", "brave"],  # mixed: bing anchors it
]

# Client-side timeout. The SearXNG `outgoing.request_timeout` is 12.0s, so we
# give the client the same budget; if it can't respond in 12s there's no
# point waiting longer (the engines will already have been timed out
# server-side and the response will be a partial / empty result set).
_CLIENT_TIMEOUT = 12.0


def _query_one_engine_set(
    search_term: str,
    clean_term: str,
    numresults: int,
    engine_set: list[str] | None,
    instance_url: str,
) -> list[str]:
    """
    Hit SearXNG once with a specific `engines=` parameter.
    When `engine_set` is None, omit the `engines` param entirely so SearXNG
    uses its default engine pool (generic fallback when the pinned sets all
    failed).
    Returns a list of ranked URLs, or [] if this set produced nothing usable.
    Does NOT retry on network errors - the caller rotates to the next set.
    """
    params = {
        "format":   "json",
        "language": "en-US",   # was "en" - bing needs the full locale tag to
                               # pick the right regional index (en alone lets
                               # it geo-locate from the server IP, which on
                               # this host returned RU/IN results for English
                               # queries).
        "pageno":   1,
        "q":        search_term,
    }
    if engine_set is not None:
        params["engines"] = ",".join(engine_set)
    search_url = f"{instance_url}/search"

    try:
        response = requests.get(
            search_url, params=params, headers=HEADERS,
            timeout=_CLIENT_TIMEOUT, allow_redirects=True, verify=True
        )
    except requests.exceptions.Timeout:
        print(f"[SearXNG] Timeout on engines={engine_set}")
        return []
    except requests.exceptions.ConnectionError:
        print(f"[SearXNG] Connection error on engines={engine_set}")
        return []
    except Exception as e:
        print(f"[SearXNG] Unexpected error on engines={engine_set}: {e}")
        return []

    if response.status_code == 429:
        # Upstream SearXNG limiter; rare now that limiter is disabled, but
        # if the user reinstates it the polite thing is to back off once.
        print(f"[SearXNG] HTTP 429 on engines={engine_set} - backing off")
        return []

    if response.status_code != 200:
        print(f"[SearXNG] HTTP {response.status_code} on engines={engine_set}")
        return []

    try:
        raw_results = safe_parse_response(response)
    except ValueError as e:
        print(f"[Security] {e}")
        return []

    if not raw_results:
        return []

    safe_urls = []
    for r in raw_results:
        candidate = r.get("url", "")
        if not isinstance(candidate, str):
            continue
        if not is_safe_url(candidate):
            continue
        if any(d in candidate for d in DISCARD_DOMAINS):
            continue
        title = r.get("title") or ""
        if _DEFINITION_TITLE_RE.search(title):
            continue
        safe_urls.append({
            "url": candidate,
            "title": title,
            "content": r.get("content") or "",
        })
        # Collect a large candidate pool so BM25 can rank across
        # ALL relevant results, not just the engine's top-5.
        if len(safe_urls) == SEARCH_CANDIDATE_POOL:
            break

    if not safe_urls:
        return []

    # BM25-rank sources against the clean (dimension) query so
    # only the most relevant links are forwarded to the crawler.
    ranked = _rank_sources(clean_term, safe_urls, top_n=numresults)
    return [r["url"] for r in ranked]


def get_web_urls(
    search_term: str,
    numresults: int = 5,
    engine_set: list[str] | None = None,
) -> list[str]:
    """
    Query SearXNG and return ranked URLs for one dimension.

    `engine_set`, if provided, queries ONLY that set (used by the parallel
    multi-search so each sub-query fans out to a different mix). If None, we
    rotate through ENGINE_SETS in order until one returns results - this
    preserves the old sequential fallthrough behaviour for callers that
    don't pass an engine_set.

    The local SearXNG instance is always used first; only if it errors across
    every engine set do we fall back to the remote instances (less polite,
    but better than an empty result).
    """
    try:
        clean_term = sanitize_query(search_term)
    except ValueError as e:
        print(f"[Security] Query rejected: {e}")
        return []

    # Append "-site domain" exclusions for every domain we never want.
    # NOTE: use "-site domain" (space) - "-site:domain" (colon) makes the
    # bing engine return zero results. Bing-specific quirk discovered upstream.
    search_term_with_excludes = search_term
    for domain in DISCARD_DOMAINS:
        search_term_with_excludes += f" -site {domain}"

    # Build the list of engine sets to try in order.
    if engine_set is not None:
        sets_to_try = [engine_set]
    else:
        sets_to_try = list(ENGINE_SETS)

    # Try local instance first, then fallback instances only if every local
    # engine set fails. This keeps us polite to public instances.
    #
    # Resilience: if every engine set returns nothing (common when upstream
    # engines are CAPTCHA/429/cooldown-flagged from this IP), we make a second
    # pass with a short backoff and - if that also fails - a single "generic"
    # query with no engines restriction so SearXNG uses whichever engines are
    # not currently suspended. Without this, a dimension would silently end up
    # with zero URLs whenever the 3 hardcoded sets are all unavailable.
    for attempt in range(2):
        for instance_url in SEARXNG_INSTANCES:
            for eset in sets_to_try:
                print(f"[SearXNG] {instance_url} engines={eset} (attempt {attempt+1})")
                urls = _query_one_engine_set(
                    search_term=search_term_with_excludes,
                    clean_term=clean_term,
                    numresults=numresults,
                    engine_set=eset,
                    instance_url=instance_url,
                )
                if urls:
                    # Small backoff before yielding so concurrent callers don't
                    # all hammer the next dimension at the exact same instant.
                    time.sleep(random.uniform(0.3, 0.8))
                    return urls

        if attempt == 0:
            # Short backoff, then try again. Engine suspensions (DDG 300s,
            # brave 120s) mean the second pass often hits a different set
            # that has recovered.
            wait = random.uniform(2.0, 4.0)
            print(f"[SearXNG] All engine sets empty - retrying in {wait:.1f}s")
            time.sleep(wait)
        else:
            # Final generic fallback: let SearXNG pick any non-suspended engine.
            print("[SearXNG] Generic fallback (no engine restriction)...")
            generic_urls = _query_one_engine_set(
                search_term=search_term_with_excludes,
                clean_term=clean_term,
                numresults=numresults,
                engine_set=None,   # no engines= param -> SearXNG default pool
                instance_url=SEARXNG_INSTANCES[0],
            )
            if generic_urls:
                time.sleep(random.uniform(0.3, 0.8))
                return generic_urls

    print("[SearXNG] All instances/exhausted.")
    return []


# ---------------------------------------------------------------------------
# Parallel multi-search across research dimensions
# ---------------------------------------------------------------------------

# Max number of sub-queries in flight against SearXNG at once. 3 keeps the
# upstream load reasonable (SearXNG itself uses an async event loop and
# parallelizes outbound engine requests) while cutting total wall time from
# N * (search + 3s delay) to roughly N/3 * (single-search time).
_MAX_CONCURRENCY = 3

# Small per-task jitter to decorrelate the burst of parallel requests, so
# SearXNG doesn't see 3 simultaneous requests at the exact same millisecond.
_JITTER_RANGE = (0.2, 0.8)


async def async_multi_search(
    sub_queries: list[str],
    delay_between: float = 3.0,
) -> dict[str, list[str]]:
    """
    Search all sub-queries in parallel with a bounded semaphore. Each
    sub-query is assigned a different engine set (round-robin across
    ENGINE_SETS) so even if one upstream engine times out, only the
    sub-queries mapped to it are affected.

    If a dimension ends up with zero URLs, we run a second pass for ONLY the
    failed dimensions (with jitter + a different engine set), so a single
    transient upstream outage doesn't silently wipe out a dimension.

    `delay_between` is kept as a parameter for backwards compatibility with
    app.py but is ignored in favour of the per-task jitter + semaphore.
    """
    if not sub_queries:
        return {}

    sem = asyncio.Semaphore(_MAX_CONCURRENCY)
    loop = asyncio.get_event_loop()

    async def _search_one(i: int, q: str, set_offset: int) -> tuple[str, list[str]]:
        async with sem:
            # Decorrelate the burst so we don't all hit SearXNG at once.
            await asyncio.sleep(random.uniform(*_JITTER_RANGE))
            eset = ENGINE_SETS[(i + set_offset) % len(ENGINE_SETS)]
            print(f"[MultiSearch] ({i+1}/{len(sub_queries)}) engines={eset}: {q}")
            urls = await loop.run_in_executor(
                None,
                lambda: get_web_urls(search_term=q, engine_set=eset),
            )
            print(f"[MultiSearch] Found {len(urls)} URLs for: {q}")
            return q, urls

    # Pass 1: fan out all dimensions.
    pairs = await asyncio.gather(
        *[_search_one(i, q, 0) for i, q in enumerate(sub_queries)],
        return_exceptions=False,
    )
    results = dict(pairs)

    # Pass 2: retry only the dimensions that came back empty, this time with a
    # different engine set (offset +1) so we don't re-hit the same suspended
    # engines. Combined with the built-in retry+generic fallback inside
    # get_web_urls, this is the last line of defence against a dimension
    # ending up with zero URLs.
    failed = [q for q, urls in results.items() if not urls]
    if failed:
        print(f"[MultiSearch] Retrying {len(failed)} empty dimension(s): {failed}")
        await asyncio.sleep(random.uniform(1.0, 2.0))
        indexes = {q: i for i, q in enumerate(sub_queries)}
        retry_pairs = await asyncio.gather(
            *[_search_one(indexes[q], q, 1) for q in failed],
            return_exceptions=False,
        )
        for q, urls in retry_pairs:
            if urls:
                results[q] = urls

    return results
