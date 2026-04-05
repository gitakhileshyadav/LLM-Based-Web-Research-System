import os                                        # 1. env vars first
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"
os.environ["POSTHOG_DISABLED"]     = "true"
os.environ["POSTHOG_API_KEY"]      = ""

import asyncio          #2.  stdlib import
import re
import sys
import tempfile
import time
import random
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import streamlit as st                           # 3. streamlit import

st.set_page_config(                              # 4. FIRST streamlit command
    page_title="LLM with Web Search",
    layout="wide"
)
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ✅ Windows requires ProactorEventLoop for Playwright subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import requests
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Constants ─────────────────────────────────────────────────────────────────

CRAWLER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SEARXNG_INSTANCES = [
    "http://localhost:8080",
    "https://searx.tiekoetter.com",
    "https://searxng.world",
]

DISCARD_DOMAINS = [
    "youtube.com", "britannica.com", "vimeo.com",
    "spotify.com", "gaana.com",
    "reddit.com",                  # ✅ add — blocks crawlers
    "quora.com",                   # ✅ add — blocks crawlers
    "facebook.com",                # ✅ add — requires login
    "instagram.com",               # ✅ add — requires login
    "twitter.com", "x.com",        # ✅ add — requires login
]

BLOCKED_HOSTS = [
    "localhost", "127.", "192.168.", "10.", "172.16.",
    "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
    "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
    "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
    "0.0.0.0", "::1", "metadata.google.internal",
]

ALLOWED_SCHEMES  = {"http", "https"}
MAX_QUERY_LENGTH = 200

HEADERS = {
    "User-Agent": CRAWLER_USER_AGENT,
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Vector Database ───────────────────────────────────────────────────────────

def get_vector_collections() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db",
        settings=Settings(anonymized_telemetry=False)
    )
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,
    )


def normalize_url(url: str) -> str:
    return (
        url.replace("https://", "")
           .replace("http://", "")        # ✅ fixed — was missing
           .replace("www.", "")
           .replace("/", "_")
           .replace("-", "_")
           .replace(".", "_")
    )
#Deleting existing content the making og the structred rag
def reset_collection(chroma_client: chromadb.Client) -> None:
    """
    Delete and recreate the collection — use once to clear stale schema chunks.
    """
    try:
        chroma_client.delete_collection("web_llm")
        print("[VectorDB] Collection deleted — will be recreated on next run")
    except Exception as e:
        print(f"[VectorDB] Reset failed: {e}")

def clear_collection_for_urls(collection: chromadb.Collection, urls: list[str]) -> None:
    """
    Delete all existing chunks for the given URLs before storing new ones.
    Prevents stale data from previous sessions contaminating RAG context.
    """
    for url in urls:
        normalized = normalize_url(url)
        try:
            # ── Get all IDs that belong to this URL ───────────────────────────
            existing = collection.get(
                where={"source": url}
            )
            existing_ids = existing.get("ids", [])

            if existing_ids:
                collection.delete(ids=existing_ids)
                print(f"[VectorDB] Deleted {len(existing_ids)} stale chunks for {url}")
            else:
                print(f"[VectorDB] No existing chunks for {url}")

        except Exception as e:
            print(f"[VectorDB] Failed to clear chunks for {url}: {e}")

def clean_markdown_content(text: str) -> str:
    """
    Post-process crawled markdown to remove leftover noise before RAG storage.
    Runs after crawl4ai's own cleaning for a second pass.
    """
    if not text:
        return ""

    # ── Remove leftover HTML tags ─────────────────────────────────────────────
    text = re.sub(r"<[^>]+>", "", text)

    # ── Remove base64 encoded images ──────────────────────────────────────────
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "", text)

    # ── Remove URLs and markdown links ────────────────────────────────────────
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)   # [text](url) → text
    text = re.sub(r"https?://\S+", "", text)                 # bare URLs

    # ── Remove markdown image tags ────────────────────────────────────────────
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)

    # ── Remove excessive special characters ───────────────────────────────────
    text = re.sub(r"[|]{2,}", "", text)                      # table separators
    text = re.sub(r"[-]{3,}", "", text)                      # horizontal rules
    text = re.sub(r"[*]{3,}", "", text)                      # excessive asterisks
    text = re.sub(r"[#]{4,}", "", text)                      # deep heading levels

    # ── Remove cookie/ad/tracking text patterns ───────────────────────────────
    noise_patterns = [
        r"(?i)accept\s+all\s+cookies.*",
        r"(?i)we\s+use\s+cookies.*",
        r"(?i)privacy\s+policy.*",
        r"(?i)terms\s+of\s+service.*",
        r"(?i)subscribe\s+to\s+our\s+newsletter.*",
        r"(?i)sign\s+up\s+for.*newsletter.*",
        r"(?i)advertisement\b.*",
        r"(?i)sponsored\s+content.*",
        r"(?i)click\s+here\s+to.*",
        r"(?i)follow\s+us\s+on.*",
        r"(?i)share\s+this\s+article.*",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # ── Normalize whitespace ──────────────────────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)      # max 2 consecutive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)       # collapse spaces/tabs
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)  # trim each line

    return text.strip()
def add_to_vector_database(
    results: list[CrawlResult],
    prompt: str,
    collection: chromadb.Collection,
) -> str:
    """
    Store crawled results into ChromaDB with structured metadata.

    Metadata schema per chunk:
      - source        : original URL
      - session_id    : timestamp of this crawl session
      - chunk_index   : position of chunk within its source document
      - total_chunks  : total chunks from this source
      - query         : the search prompt that triggered this crawl
      - crawled_at    : ISO timestamp of when page was crawled
    """

    # ── Session ID — shared across all chunks in this run ────────────────────
    session_id = str(int(time.time()))
    crawled_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ── Step 1: Clear stale chunks for these URLs before inserting ────────────
    active_urls = [r.url for r in results if r.url]
    clear_collection_for_urls(collection, active_urls)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    MAX_CHUNKS_PER_URL = 20
    total_chunks = 0

    for result in results:
        documents, metadatas, ids = [], [], []

        # ── Extract content ───────────────────────────────────────────────────
        markdown_result = ""

        markdown_result = ""

        # ✅ fit_markdown first — but only if it has real content
        if hasattr(result, "fit_markdown") and result.fit_markdown and len(result.fit_markdown.strip()) > 100:
            markdown_result = clean_markdown_content(result.fit_markdown)
            print(f"[VectorDB] Using fit_markdown ({len(markdown_result)} chars) for {result.url}")

        # ✅ Fall back to raw markdown if fit_markdown is empty or too short
        elif result.markdown and len(str(result.markdown).strip()) > 100:
            markdown_result = clean_markdown_content(str(result.markdown))
            print(f"[VectorDB] Using raw markdown ({len(markdown_result)} chars) for {result.url}")

        elif result.html:
            markdown_result = clean_markdown_content(result.html)
            print(f"[VectorDB] Using html fallback for {result.url}")

        else:
            print(f"[VectorDB] No content for {result.url} — skipping")
            continue

        # ── Content length check ──────────────────────────────────────────────
        word_count = len(markdown_result.split())
        if word_count < 30:
            print(f"[VectorDB] Only {word_count} words after cleaning for {result.url} — skipping")
            continue

        print(f"[VectorDB] Final content: {word_count} words for {result.url}")

        # ── Write to temp file and split ──────────────────────────────────────
        all_splits = []                            # ✅ always initialized before try

        try:
            temp_file = tempfile.NamedTemporaryFile(
                "w", suffix=".md", delete=False, encoding="utf-8"
            )
            temp_file.write(markdown_result)
            temp_file.flush()
            temp_file.close()

            loader     = UnstructuredMarkdownLoader(temp_file.name, mode="single")
            docs       = loader.load()
            all_splits = text_splitter.split_documents(docs)

        except Exception as e:
            print(f"[VectorDB] Failed to process {result.url}: {e}")
            continue                               # ✅ skips to next result

        finally:
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

        # ✅ Safe — all_splits is always defined, continue already called on error
        all_splits = all_splits[:MAX_CHUNKS_PER_URL]
        print(f"[VectorDB] Capped to {len(all_splits)} chunks for {result.url}")

        if not all_splits:
            print(f"[VectorDB] No chunks generated for {result.url} — skipping")
            continue

        normalized_url   = normalize_url(result.url)
        total_url_chunks = len(all_splits)

        # ── Build structured documents, metadata, IDs ─────────────────────────
        for idx, split in enumerate(all_splits):
            content = split.page_content
            if not content or not content.strip():
                continue

            documents.append(content)
            metadatas.append({
                "source":       result.url,
                "session_id":   session_id,
                "chunk_index":  idx,
                "total_chunks": total_url_chunks,
                "query":        prompt,
                "crawled_at":   crawled_at,
            })
            ids.append(f"{normalized_url}__{session_id}__{idx}")

        if documents:
            print(f"[VectorDB] Upserting {len(documents)} chunks for {result.url}")
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            total_chunks += len(documents)

    # ── Display crawled results ───────────────────────────────────────────────
    st.subheader("Crawled Pages")
    for r in results:
        with st.expander(f"{'✅' if r.success else '❌'} {r.url}"):
            st.write(f"**Content length:** {len(r.markdown) if r.markdown else 0} chars")
            if r.markdown:
                st.text(r.markdown[:500])

    # ── Display vector DB status ──────────────────────────────────────────────
    st.subheader("Vector Database")
    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks This Session", total_chunks)
    col2.metric("Total Chunks in DB",  collection.count())
    col3.metric("Session ID",          session_id)

    # ── Display top 5 relevant chunks ─────────────────────────────────────────
    if total_chunks > 0 and collection.count() > 0:
        st.subheader("Top 5 Relevant Chunks")
        try:
            query_results = collection.query(
                query_texts=[prompt],
                n_results=min(5, collection.count()),
            )
            for i, (doc, meta) in enumerate(
                zip(query_results["documents"][0], query_results["metadatas"][0])
            ):
                chunk_index       = meta.get("chunk_index", "?")
                total_chunks_meta = meta.get("total_chunks", "?")
                source            = meta.get("source", "Unknown")
                session           = meta.get("session_id", "Unknown")
                crawled           = meta.get("crawled_at", "Unknown")

                with st.expander(
                    f"Chunk {i+1} | Index {chunk_index}/{total_chunks_meta} | {source}"
                ):
                    st.caption(f"Session: {session} | Crawled: {crawled}")
                    st.write(doc)

        except Exception as e:
            st.error(f"Failed to query vector DB: {e}")
            print(f"[VectorDB] Query error: {e}")

    return session_id


# ── Web Search ────────────────────────────────────────────────────────────────

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
                        if len(safe_urls) == 5:
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


# ── Robots.txt ────────────────────────────────────────────────────────────────


def check_robots_txt(urls: list[str]) -> list[str]:
    allowed_urls = []

    for url in urls:
        try:
            parsed     = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            rp = RobotFileParser(robots_url)

            # ✅ Set timeout — prevents hanging on slow robots.txt
            import urllib.request
            try:
                req      = urllib.request.Request(
                    robots_url,
                    headers={"User-Agent": CRAWLER_USER_AGENT}
                )
                response = urllib.request.urlopen(req, timeout=5)
                rp.parse(response.read().decode("utf-8", errors="ignore").splitlines())
            except Exception:
                # ✅ If robots.txt unreachable — assume allowed
                allowed_urls.append(url)
                continue

            # ✅ Check with real browser user agent, not wildcard "*"
            if rp.can_fetch(CRAWLER_USER_AGENT, url):
                allowed_urls.append(url)
                print(f"[Robots.txt] Allowed : {url}")
            else:
                # ✅ Double check with wildcard before final block decision
                if rp.can_fetch("*", url):
                    allowed_urls.append(url)
                    print(f"[Robots.txt] Allowed via wildcard: {url}")
                else:
                    print(f"[Robots.txt] Blocked : {url}")

        except Exception:
            # ✅ Any error — assume allowed, don't block valid URLs
            allowed_urls.append(url)

    return allowed_urls


# ── Crawling ──────────────────────────────────────────────────────────────────
# ── Domain Intelligence ───────────────────────────────────────────────────────

# JS-heavy news/media sites — need longer wait and full browser
JS_HEAVY_DOMAINS = [
    "theguardian.com", "timesofindia.com", "medium.com",
    "nytimes.com", "bloomberg.com", "forbes.com",
    "wsj.com", "economist.com", "bbc.com", "bbc.co.uk",
    "reuters.com", "apnews.com", "techcrunch.com",
    "wired.com", "theatlantic.com", "politico.com",
    "indianexpress.com", "ndtv.com", "hindustantimes.com",
    "businessinsider.com", "cnbc.com", "cnn.com","unacademy.com","byju's.com", "vedantu.com",
]

# Reputed scientific / research sources — highest priority for science queries
SCIENTIFIC_DOMAINS = [
    "arxiv.org", "paperswithcode.com",
    "huggingface.co", "openai.com",
    "deepmind.google", "research.google",
    "nature.com", "science.org",
    "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
    "semanticscholar.org", "distill.pub",
    "pytorch.org", "tensorflow.org",
    "ai.meta.com", "microsoft.com/research",
    "blog.research.google", "deeplearning.ai",
]

# Reputed news sources — highest priority for current events queries
NEWS_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "timesofindia.com", "ndtv.com",
    "bloomberg.com", "ft.com", "economist.com",
    "thehindu.com", "indianexpress.com", "hindustantimes.com",
    "aljazeera.com", "npr.org", "pbs.org",
]

# Keywords that indicate a scientific query
SCIENCE_KEYWORDS = [
    "research", "paper", "study", "algorithm", "model",
    "neural", "machine learning", "deep learning", "llm",
    "transformer", "diffusion", "quantum", "physics",
    "biology", "chemistry", "genome", "protein",
    "arxiv", "dataset", "benchmark", "sota",
    "artificial intelligence", "ai model", "published",
]

# Keywords that indicate a current events query
NEWS_KEYWORDS = [
    "latest", "today", "yesterday", "this week", "breaking",
    "current", "recent", "news", "happened", "announced",
    "election", "government", "minister", "president",
    "war", "conflict", "crisis", "economy", "market",
    "2024", "2025", "2026",
]


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
        bm25_threshold=1.5,
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

# ── LLM ───────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "gemma3:1b"

SYSTEM_PROMPTS = {
    "science": """You are a senior research scientist and technical writer.
Your task is to produce a comprehensive, well-structured research summary.

Instructions:
- Write a detailed, multi-paragraph answer covering all aspects of the question
- Organize your answer with clear sections: Overview, Key Findings, Technical Details, Implications
- Cite specific sources for each major claim using [Source N] notation
- Include specific data, statistics, dates, and figures from the context
- Explain technical concepts clearly with examples where relevant
- Highlight agreements and contradictions between different sources
- End with a summary of key takeaways
- If context is insufficient for any aspect, explicitly state what is missing""",

    "news": """You are a senior investigative journalist and analyst.
Your task is to produce a comprehensive, well-structured news analysis.

Instructions:
- Write a detailed, multi-paragraph analysis covering all aspects of the event
- Organize your answer with clear sections: What Happened, Background, Key Players, Impact, What's Next
- Cite specific sources for each claim using [Source N] notation
- Include specific dates, locations, names, and figures from the context
- Provide context and background to help understand the significance
- Highlight different perspectives and conflicting reports if present
- End with an assessment of the broader implications
- Clearly distinguish between confirmed facts and analysis""",

    "general": """You are a comprehensive research assistant and expert analyst.
Your task is to produce a detailed, well-structured research report.

Instructions:
- Write a thorough, multi-paragraph answer that fully addresses the question
- Organize your answer with clear sections relevant to the topic
- Cite specific sources for each major claim using [Source N] notation
- Include specific facts, dates, figures, and quotes from the context
- Cover multiple dimensions: historical background, current status, significance
- Compare information from different sources and note any discrepancies
- Provide analysis and insight beyond just summarizing the sources
- End with a concise summary of the most important points
- If the context lacks information on any important aspect, state this clearly
- Minimum length: 3-4 detailed paragraphs""",
}

# ── Query Expansion ───────────────────────────────────────────────────────────

def expand_query(prompt: str, query_type: str) -> list[str]:
    """
    Generate sub-queries covering different dimensions of the original query.
    Uses gemma3:1b to produce 4 targeted sub-queries for deeper research.
    """

    expansion_prompts = {
        "science": f"""You are a research assistant. Given this scientific query:
"{prompt}"

Generate exactly 4 different search queries that together cover ALL dimensions:
1. Core concept and definition
2. Latest research and recent developments
3. Technical implementation and methodology
4. Real-world applications and impact

Return ONLY the 4 queries, one per line, no numbering, no explanation.""",

        "news": f"""You are a news researcher. Given this news query:
"{prompt}"

Generate exactly 4 different search queries that together cover ALL dimensions:
1. What happened — core facts and timeline
2. Background and causes leading to this event
3. Key people and organizations involved
4. Impact, consequences and future implications

Return ONLY the 4 queries, one per line, no numbering, no explanation.""",

        "general": f"""You are a research assistant. Given this query:
"{prompt}"

Generate exactly 4 different search queries that together cover ALL dimensions:
1. Basic facts, definition and overview
2. Historical background and origin
3. Current status, achievements and significance
4. Criticism, controversies and different perspectives

Return ONLY the 4 queries, one per line, no numbering, no explanation.""",
    }

    system_prompt = "You generate precise search queries. Return only the queries, nothing else."
    expansion_prompt = expansion_prompts.get(query_type, expansion_prompts["general"])

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": expansion_prompt,
                "system": system_prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # ── Parse sub-queries — one per line ──────────────────────────────────
        sub_queries = [
            line.strip()
            for line in raw.strip().splitlines()
            if line.strip() and len(line.strip()) > 10
        ][:4]                          # cap at 4

        # ── Fallback if expansion failed ──────────────────────────────────────
        if len(sub_queries) < 2:
            print(f"[QueryExpansion] Failed to parse — using original query")
            return [prompt]

        print(f"[QueryExpansion] Generated {len(sub_queries)} sub-queries:")
        for i, q in enumerate(sub_queries):
            print(f"  {i+1}. {q}")

        return sub_queries

    except Exception as e:
        print(f"[QueryExpansion] Error: {e} — using original query")
        return [prompt]

# ── Async Multi-Search ────────────────────────────────────────────────────────

async def async_multi_search(
    sub_queries: list[str],
    delay_between: float = 3.0,
) -> dict[str, list[str]]:
    """
    Search all sub-queries with delay between each to avoid IP blocking.
    Returns dict mapping sub-query → list of URLs found.
    """
    import asyncio

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

# ── Iterative Generation ──────────────────────────────────────────────────────

def generate_dimension_answer(
    original_prompt: str,
    sub_query: str,
    context: str,
    dimension_num: int,
    total_dimensions: int,
    query_type: str,
) -> str:
    """
    Generate a detailed answer for one dimension/sub-query.
    """
    system_prompt = f"""You are a comprehensive research analyst writing section {dimension_num} of {total_dimensions} of a deep research report.

Your task: Answer this specific dimension of the research thoroughly.
Original question: {original_prompt}
This dimension: {sub_query}

Instructions:
- Write 2-3 detailed paragraphs specifically addressing this dimension
- Use specific facts, dates, names, and figures from the context
- Cite sources using [Source N] notation
- Be analytical — do not just summarize, provide insight
- This is ONE section of a larger report — stay focused on this dimension"""

    full_prompt = f"""Context:
{context}

Dimension to address: {sub_query}

Write a detailed section addressing this dimension:"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": full_prompt,
                "system": system_prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"[Generation] Error for dimension {dimension_num}: {e}")
        return f"_(Could not generate content for this dimension: {e})_"


def synthesize_final_report(
    original_prompt: str,
    dimension_answers: list[dict],
    query_type: str,
) -> str:
    """
    Combine all dimension answers into a cohesive final research report.
    """
    sections = "\n\n".join([
        f"=== Dimension {i+1}: {d['sub_query']} ===\n{d['answer']}"
        for i, d in enumerate(dimension_answers)
        if d['answer']
    ])

    system_prompt = """You are a senior research editor creating a final comprehensive report.
Your task is to synthesize multiple research sections into one cohesive, well-structured report.

Instructions:
- Combine all sections into a flowing, well-organized report
- Use clear headers for each major section
- Remove repetition but preserve all unique information
- Ensure logical flow between sections
- Add a brief Executive Summary at the start
- Add Key Takeaways at the end
- Maintain all source citations
- The final report should be comprehensive and detailed"""

    full_prompt = f"""Original Research Question: {original_prompt}

Research sections to synthesize:
{sections}

Write the final comprehensive research report:"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": full_prompt,
                "system": system_prompt,
                "stream": False,
            },
            timeout=180,               # longer timeout for synthesis
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        print(f"[Synthesis] Error: {e}")
        # ✅ Fallback — return all dimension answers concatenated
        return "\n\n".join([
            f"### {d['sub_query']}\n{d['answer']}"
            for d in dimension_answers
            if d['answer']
        ])

def check_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def query_llm(prompt: str, context: str, query_type: str) -> str:
    """
    Send retrieved context + prompt to gemma3:4b via Ollama.
    Uses query-type specific system prompt for better accuracy.
    """
    if not check_ollama_running():
        return "⚠️ Ollama is not running. Start it with `ollama serve` in terminal."

    system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["general"])

    full_prompt = f"""Context from web sources:
{context}

User Question: {prompt}

Answer based on the above context only:"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":       OLLAMA_MODEL,
                "prompt":      full_prompt,
                "system":      system_prompt,
                "stream":      False,
                "temperature": 0.3,
                "top_p":       0.9,
            },
            timeout=220,
        )
        response.raise_for_status()
        return response.json().get("response", "No response generated.")

    except requests.exceptions.Timeout:
        return "⚠️ LLM timed out. Try a shorter query."
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Run `ollama serve` in terminal."
    except Exception as e:
        return f"⚠️ LLM error: {e}"
# ── Streamlit UI ──────────────────────────────────────────────────────────────
def _answer_directly(prompt: str, query_type: str) -> None:
    """
    Answer using LLM general knowledge when no RAG context is available.
    Used when web search toggle is OFF and DB is empty.
    """
    DIRECT_SYSTEM_PROMPTS = {
        "science": "You are an expert research assistant. Answer the question accurately using your knowledge.",
        "news":    "You are a knowledgeable assistant. Answer based on your training data. Note that your knowledge has a cutoff date.",
        "general": "You are a helpful assistant. Answer the question clearly and accurately.",
    }

    system_prompt = DIRECT_SYSTEM_PROMPTS.get(query_type, DIRECT_SYSTEM_PROMPTS["general"])

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "No response generated.")

    except Exception as e:
        answer = f"⚠️ LLM error: {e}"

    st.divider()
    st.subheader("💡 Answer")
    st.write(answer)
    st.caption("⚠️ Answer based on model knowledge only — no web sources used.")

async def run():

    # ✅ Initialize DB first
    collection, chroma_client = get_vector_collections()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Database Management")
        st.write(f"Total chunks in DB: {collection.count()}")
        if st.button("🗑️ Reset Database", type="secondary"):
            reset_collection(chroma_client)
            st.success("Database reset. Please reload the page.")
            st.stop()

    # ── Main UI ───────────────────────────────────────────────────────────────
    st.header("🔍 LLM Web Search")
    prompt = st.text_area(
        label="Query",
        placeholder="Ask anything...",
        label_visibility="hidden",
    )
    is_web_search = st.toggle(
        "Enable Web Search",
        value=False,
        key="enable_web_search"         # ✅ key preserves toggle state
    )
    go = st.button("GO", type="primary")

    if prompt and go:

        session_id  = None
        query_type  = detect_query_type(prompt)

        # ── Web search path ───────────────────────────────────────────────────
        if is_web_search:

            # ── Step 1: Expand query into sub-queries ─────────────────────────
            with st.spinner("Expanding query into research dimensions..."):
                sub_queries = expand_query(prompt, query_type)

            st.subheader("🔍 Research Dimensions")
            for i, q in enumerate(sub_queries):
                st.write(f"**Dimension {i+1}:** {q}")

            # ── Step 2: Search all dimensions asynchronously ──────────────────
            with st.spinner("Searching all dimensions (with delays to avoid blocking)..."):
                search_results = await async_multi_search(
                    sub_queries=sub_queries,
                    delay_between=3.0,
                )

            # ── Step 3: Collect all unique URLs ───────────────────────────────
            all_urls = []
            seen_urls = set()
            url_to_queries = {}            # track which sub-query each URL came from

            for sub_query, urls in search_results.items():
                for url in urls:
                    if url not in seen_urls:
                        all_urls.append(url)
                        seen_urls.add(url)
                    if url not in url_to_queries:
                        url_to_queries[url] = []
                    url_to_queries[url].append(sub_query)

            st.subheader("🌐 URLs Found")
            st.write(f"**{len(all_urls)} unique URLs** across all dimensions")
            for url in all_urls:
                queries_for_url = url_to_queries.get(url, [])
                st.write(f"- {url} _(relevant to {len(queries_for_url)} dimension(s))_")

            if not all_urls:
                st.error("No URLs found across all dimensions.")
                st.stop()

            # ── Step 4: Robots.txt check ───────────────────────────────────────
            with st.spinner("Checking robots.txt for all URLs..."):
                allowed_urls = check_robots_txt(all_urls)

            if not allowed_urls:
                st.error("All URLs blocked by robots.txt.")
                st.stop()

            # ── Step 5: Crawl all URLs ─────────────────────────────────────────
            with st.spinner(f"Crawling {len(allowed_urls)} pages..."):
                results = await crawl_webpages(
                    urls=allowed_urls,
                    prompt=prompt,
                )

            # ── Step 6: Store in ChromaDB ──────────────────────────────────────
            session_id = add_to_vector_database(
                results=results,
                prompt=prompt,
                collection=collection,
            )

        # ── Step 7: Check DB has data ─────────────────────────────────────────
        if collection.count() == 0:
            st.info("Database empty — answering from model knowledge only.")
            _answer_directly(prompt, query_type)
            st.stop()

        # ── Step 8: Iterative generation — one answer per dimension ───────────
        st.divider()
        st.subheader("🔬 Deep Research Report")

        if is_web_search and 'sub_queries' in locals():
            dimensions = sub_queries
        else:
            # ✅ Toggle OFF — generate dimensions from existing DB
            dimensions = expand_query(prompt, query_type)

        dimension_answers = []
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, sub_query in enumerate(dimensions):

            status_text.text(f"Researching dimension {i+1}/{len(dimensions)}: {sub_query}")

            # ── Query DB for this dimension ────────────────────────────────────
            try:
                if session_id:
                    dim_results = collection.query(
                        query_texts=[sub_query],
                        n_results=min(5, collection.count()),
                        where={"session_id": session_id},
                    )
                    if not dim_results["documents"][0]:
                        dim_results = collection.query(
                            query_texts=[sub_query],
                            n_results=min(5, collection.count()),
                        )
                else:
                    dim_results = collection.query(
                        query_texts=[sub_query],
                        n_results=min(5, collection.count()),
                    )

                dim_docs  = dim_results.get("documents",  [[]])[0]
                dim_metas = dim_results.get("metadatas",  [[]])[0]

            except Exception as e:
                print(f"[Research] DB query error for dimension {i+1}: {e}")
                dim_docs, dim_metas = [], []

            # ── Build context for this dimension ──────────────────────────────
            if dim_docs:
                context_parts = []
                for j, (doc, meta) in enumerate(zip(dim_docs, dim_metas)):
                    source  = meta.get("source", "Unknown")
                    excerpt = doc[:800]
                    context_parts.append(f"[Source {j+1} — {source}]\n{excerpt}")
                dim_context = "\n\n".join(context_parts)

                if len(dim_context) > 6000:
                    dim_context = dim_context[:6000]
            else:
                dim_context = "No specific context found for this dimension."

            # ── Generate answer for this dimension ─────────────────────────────
            answer = generate_dimension_answer(
                original_prompt=prompt,
                sub_query=sub_query,
                context=dim_context,
                dimension_num=i + 1,
                total_dimensions=len(dimensions),
                query_type=query_type,
            )

            dimension_answers.append({
                "sub_query": sub_query,
                "answer":    answer,
                "sources":   list({m.get("source", "") for m in dim_metas}),
            })

            # ── Show dimension answer as it's generated ────────────────────────
            with st.expander(
                f"📖 Dimension {i+1}: {sub_query}",
                expanded=(i == 0),     # expand first one by default
            ):
                st.write(answer)
                if dim_metas:
                    st.caption("Sources: " + ", ".join({
                        m.get("source", "") for m in dim_metas
                    }))

            progress_bar.progress((i + 1) / len(dimensions))

        status_text.text("Synthesizing final report...")

        # ── Step 9: Synthesize all dimensions into final report ────────────────
        with st.spinner("Synthesizing comprehensive report..."):
            final_report = synthesize_final_report(
                original_prompt=prompt,
                dimension_answers=dimension_answers,
                query_type=query_type,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Research complete!")

        st.divider()
        st.subheader("📋 Final Research Report")
        st.write(final_report)

        # ── Step 10: All sources ───────────────────────────────────────────────
        st.subheader("📚 All Sources")
        all_sources = set()
        for d in dimension_answers:
            all_sources.update(d["sources"])
        for source in sorted(all_sources):
            if source:
                st.markdown(f"- {source}")

# ── Entry Point ───────────────────────────────────────────────────────────────

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())
else:
    asyncio.run(run())