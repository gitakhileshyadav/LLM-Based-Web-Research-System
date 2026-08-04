"""Centralized configuration for the LLM Deep Web Research Tool."""

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Crawler / HTTP ───────────────────────────────────────────────────────────

CRAWLER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": CRAWLER_USER_AGENT,
    "Accept":          "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Search (SearXNG) ─────────────────────────────────────────────────────────

SEARXNG_INSTANCES = [
    "http://localhost:8080",            # primary — local Docker instance
    "https://searx.tiekoetter.com",     # fallback 1
    "https://searxng.world",            # fallback 2
]

# Domains to exclude from every search via "-site domain".
# Trimmed from 18 to the critical few: the dictionary/definition sites are
# already dropped by the _DEFINITION_TITLE_RE + BM25 filters in search.py,
# so listing them here only bloated every query and caused Bing to misparse
# multi-word phrases (it started returning generic "Student" pages for a
# "student protests India 2024" query). Only keep domains that would waste a
# crawl slot (login walls / crawler blocks).
DISCARD_DOMAINS = [
    "youtube.com", "reddit.com", "quora.com",
    "facebook.com", "instagram.com", "twitter.com",
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
MAX_SEARCH_RESULTS = 5
SEARCH_CANDIDATE_POOL = 25

# ── Pre-crawl URL curation ────────────────────────────────────────────────────

# A non-trusted URL must share at least this many substantive query tokens with
# the prompt to survive the pre-crawl filter (kills single-word matches like
# impact.com matching only "impact"). Applied ONLY when the pool is large.
MIN_URL_TOKEN_OVERLAP = 2

# When the merged URL pool is at or below this size (e.g. most engines are
# suspended and a search only returned a handful of links), curation goes
# LENIENT: it only drops non-trusted root homepages and still applies the
# domain cap, but does NOT apply the token-overlap filter or the total cap.
# Over-filtering a tiny pool was the cause of a run ending with one useless
# portal URL and an empty answer.
CURATION_LENIENT_MAX_URLS = 6

# Max URLs kept per host after dedup (kills 5 near-identical impact.* variants).
MAX_URLS_PER_DOMAIN = 3

# Hard cap on the total URL list handed to the crawler (priority-ordered).
MAX_CRAWL_URLS = 12


# ── Vector Database ───────────────────────────────────────────────────────────

# Resolved against the project root so the DB directory is stable no matter
# where the app is launched from (streamlit run app/main.py, docker, tests).
CHROMA_DIR_PATH  = os.path.join(_PROJECT_ROOT, "web-search-llm-db")
COLLECTION_NAME  = "web_llm"
EMBEDDING_MODEL  = "nomic-embed-text"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 80
MAX_CHUNKS_PER_URL = 20


# ── Crawling / Content Filtering ─────────────────────────────────────────────

BM25_THRESHOLD = 1.5


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


# ── LLM / Ollama ──────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
#OLLAMA_MODEL    = "gemma3:1b"
OLLAMA_MODEL    = "gemma4:31b-cloud"

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
