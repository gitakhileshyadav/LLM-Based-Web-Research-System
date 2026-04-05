# 🔍 LLM Deep Web Research Tool

## Description

A production-grade, RAG-powered deep research tool that performs real-time multi-dimensional web research using a fully local AI stack. It solves the problem of shallow, single-source answers by automatically expanding your query into multiple research dimensions, crawling reputed web sources, storing structured content in a vector database, and generating comprehensive research reports — all on your own machine with zero external API costs.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Who This Tool Is For](#who-this-tool-is-for)
- [Scalability](#scalability)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Author](#author)

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| Docker Desktop | Latest | SearXNG container |
| Ollama | Latest | Local LLM inference |
| RAM | 8 GB minimum | Model loading |

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/llm-web-search.git
cd llm-web-search
```

### Step 2 — Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Playwright browser

```bash
playwright install chromium
```

### Step 5 — Set up Ollama models

Download Ollama from https://ollama.com and install it. Then pull the required models:

```bash
# LLM for generation — choose based on your available RAM
ollama pull gemma3:1b       # 1.5 GB — for 8 GB RAM systems
ollama pull gemma3:4b       # 3.3 GB — for 16 GB RAM systems (better quality)

# Embedding model — required for ChromaDB
ollama pull nomic-embed-text
```

Verify models are available:

```bash
ollama list
```

Expected output:

```
NAME                       SIZE
gemma3:1b                  1.5 GB
nomic-embed-text:latest    274 MB
```

### Step 6 — Set up SearXNG with Docker

**6a — Start the container:**

```bash
docker run -d --name searxng -p 8080:8080 -e SEARXNG_SECRET_KEY="your-random-secret-key" searxng/searxng:latest
```

**6b — Copy the default settings file:**

```bash
docker cp searxng:/etc/searxng/settings.yml settings.yml
```

**6c — Open settings.yml and make these changes:**

Find the search: section and add json to formats:

```yaml
search:
  safe_search: 1
  formats:
    - html
    - json          # add this line
```

Find suspended_times: and reduce ban durations:

```yaml
suspended_times:
  SearxEngineAccessDenied: 30
  SearxEngineCaptcha: 60
  SearxEngineTooManyRequests: 30
  cf_SearxEngineCaptcha: 60
  cf_SearxEngineAccessDenied: 30
  recaptcha_SearxEngineCaptcha: 1500
```

Disable all Google engines to prevent 403 bans:

```yaml
- name: google
  engine: google
  shortcut: go
  disabled: true
```

Repeat disabled: true for google images, google news, google videos, google scholar, google maps.

Update the outgoing: section:

```yaml
outgoing:
  request_timeout: 12.0
  max_request_timeout: 20.0
  pool_connections: 10
  pool_maxsize: 10
  enable_http2: true
```

**6d — Copy settings back and restart:**

```bash
docker cp settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6e — Create limiter.toml:**

Create a new file named limiter.toml with this content:

```toml
[botdetection.ip_limit]
link_token = true

[botdetection.ip_lists]
block_ip = []
pass_ip = [
  "127.0.0.1",
  "::1"
]
```

Copy it into the container:

```bash
docker cp limiter.toml searxng:/etc/searxng/limiter.toml
docker restart searxng
```

**6f — Verify SearXNG is working:**

```bash
# Windows PowerShell
Invoke-RestMethod "http://localhost:8080/search?q=test&format=json"

# macOS / Linux
curl "http://localhost:8080/search?q=test&format=json"
```

You should see a JSON response containing a results array with search results.

---

## Usage

### Step 1 — Start Ollama

```bash
ollama serve
```

Keep this terminal open while using the app.

### Step 2 — Start SearXNG

```bash
docker start searxng
```

### Step 3 — Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

### Deep Research Mode (recommended)

1. Type your research question in the text area
2. Toggle **Enable Web Search** ON
3. Click **GO**
4. Watch the pipeline execute step by step — query expansion, async search, crawling, vector storage, iterative generation, and final report

### Query from existing database

1. Type your question
2. Leave **Enable Web Search** OFF
3. Click **GO**
4. The LLM answers using previously stored ChromaDB data, or directly from model knowledge if the database is empty

### Reset the database

Use the **Reset Database** button in the sidebar to clear all stored chunks before starting a new research topic.

### Example queries

```
# General research
Who is Mahatma Gandhi and what was his impact on Indian independence?

# Current events (automatically uses news sources)
What are the latest developments in the Iran-Israel conflict in 2026?

# Scientific research (automatically uses arxiv, HuggingFace)
What are the recent breakthroughs in large language model reasoning?
```

---

## Features

- **Query expansion** — automatically generates 4 sub-queries covering different dimensions of your topic
- **Query-type detection** — classifies queries as science, news, or general and adjusts sources and prompts accordingly
- **Async multi-search** — searches all 4 dimensions simultaneously with IP-safe delays between requests
- **Dual crawl strategy** — lightweight mode for standard sites, full-browser JS rendering for BBC, Reuters, Guardian
- **Source prioritization** — science queries hit arxiv and HuggingFace first; news queries hit Reuters and BBC first
- **BM25 content filtering** — keeps only query-relevant sentences, discards all noise
- **Ad and tracker removal** — 24+ CSS selectors strip ads, cookie banners, and popups before storage
- **Structured vector storage** — every chunk tagged with session_id, chunk_index, source, crawled_at, and query
- **Stale chunk deletion** — old data for the same URLs removed before each new crawl session
- **Iterative RAG generation** — each dimension answered separately using its own retrieved context
- **Robots.txt compliance** — checks every URL before crawling in parallel using real Chrome user agent
- **IP ban protection** — adaptive delays, consistent user agent, robots.txt respect, engine rotation
- **Fully local** — Ollama, ChromaDB, and SearXNG all run locally with zero external API costs

---

## Project Structure

```
llm-web-search/
|
+-- app.py                  # Main application — all functions and Streamlit UI
+-- requirements.txt        # Python dependencies
+-- settings.yml            # SearXNG Docker configuration
+-- limiter.toml            # SearXNG rate limit configuration
+-- README.md               # This file
|
+-- web-search-llm-db/      # ChromaDB persistent storage (auto-created on first run)
    +-- chroma.sqlite3      # Vector database with embeddings and metadata
```

### Key functions in app.py

| Function | Purpose |
|---|---|
| expand_query() | LLM generates 4 research sub-queries |
| async_multi_search() | Async search all dimensions with IP-safe delays |
| check_robots_txt() | Parallel robots.txt compliance for all URLs |
| crawl_webpages() | Dual-mode crawling — standard and JS-heavy configs |
| detect_query_type() | Classify query as science / news / general |
| prioritize_urls() | Reorder URLs by source reputation |
| clean_markdown_content() | Strip ads, trackers, and HTML noise |
| add_to_vector_database() | Chunk, embed, and store with structured metadata |
| clear_collection_for_urls() | Delete stale chunks before new crawl |
| generate_dimension_answer() | RAG answer for one research dimension |
| synthesize_final_report() | Combine all dimensions into final report |
| query_llm() | Send prompt and context to Ollama |
| _answer_directly() | Direct LLM answer when database is empty |

---

## Tech Stack

- **Streamlit** — web UI framework
- **crawl4ai** — async web crawler with JS rendering support
- **Playwright** — headless Chromium browser for JS-heavy sites
- **ChromaDB** — local vector database with cosine similarity search
- **Ollama** — local LLM inference server
- **gemma3:1b / gemma3:4b** — generation model
- **nomic-embed-text** — text embedding model
- **SearXNG** — self-hosted privacy-first search aggregator
- **Docker** — container runtime for SearXNG
- **LangChain** — document loading and recursive text splitting
- **Requests** — HTTP client for Ollama and SearXNG APIs

---

## Environment Variables

This project does not use a .env file. All configuration is done directly in app.py via constants at the top of the file.

Key settings to adjust before running:

```python
# Model selection
OLLAMA_MODEL    = "gemma3:1b"           # 1.5 GB — for 8 GB RAM
# OLLAMA_MODEL  = "gemma3:4b"           # 3.3 GB — for 16 GB RAM (better quality)
OLLAMA_BASE_URL = "http://localhost:11434"

# SearXNG instances
SEARXNG_INSTANCES = [
    "http://localhost:8080",            # primary — local Docker instance
    "https://searx.tiekoetter.com",     # fallback 1
    "https://searxng.world",            # fallback 2
]

# Domains to exclude from search results
DISCARD_DOMAINS = [
    "youtube.com", "spotify.com", "reddit.com",
    "facebook.com", "instagram.com", "twitter.com",
]

# Content quality
bm25_threshold     = 1.0   # lower = more content, higher = stricter filtering
MAX_CHUNKS_PER_URL = 12    # max chunks stored per source URL
chunk_size         = 400   # characters per chunk
chunk_overlap      = 50    # overlap between consecutive chunks
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/yourusername/llm-web-search.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes, then commit
git commit -m "Add: description of your change"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please test with at least 3 different query types — science, news, and general — before submitting a PR.

---

## Roadmap

### v1.1 — Quality
- [ ] Streaming LLM output — show tokens as they generate instead of waiting
- [ ] Source credibility scoring — rank sources by domain authority
- [ ] Citation footnotes — auto-numbered references in the final report
- [ ] Answer confidence scoring — flag when context is insufficient

### v1.2 — Performance
- [ ] Embedding cache — skip re-embedding unchanged chunks
- [ ] Parallel dimension generation — generate all 4 answers concurrently
- [ ] Incremental crawl — only re-crawl changed URLs
- [ ] Async ChromaDB upsert — non-blocking vector storage during crawl

### v1.3 — Research Depth
- [ ] PDF support — extract and chunk linked PDF documents
- [ ] Academic paper parser — structured extraction from arxiv and PubMed
- [ ] Cross-reference detection — confirm facts across multiple sources
- [ ] Contradiction detection — flag and present conflicting source claims
- [ ] Timeline generation — auto-extract chronological event sequences

### v1.4 — User Experience
- [ ] Research history — save and reload past research sessions
- [ ] Export to PDF / DOCX — download formatted research reports
- [ ] Custom trusted domain lists — user-defined source priorities
- [ ] Follow-up questions — continue researching within the same session
- [ ] Comparison mode — research two topics side by side

### v2.0 — Architecture
- [ ] Modular file structure — search.py, crawler.py, vectordb.py, llm.py
- [ ] Docker Compose — single command to start all services
- [ ] REST API — expose the research pipeline as an endpoint
- [ ] Model switching UI — swap LLM from sidebar without restarting
- [ ] Multi-user support — session isolation for concurrent users

### v2.1 — Advanced RAG
- [ ] Hybrid search — BM25 keyword search combined with vector similarity
- [ ] Re-ranking — cross-encoder to re-rank retrieved chunks by relevance
- [ ] Hypothetical Document Embedding (HyDE) — improve retrieval accuracy
- [ ] Multi-hop reasoning — chain retrieval steps for complex questions
- [ ] Knowledge graph — build entity relationships from crawled content

---

## Who This Tool Is For

This tool is designed for anyone who needs answers backed by real, current, multi-source research — not just a single Wikipedia paragraph or a hallucinated LLM response.

---

### 👨‍🎓 Students & Researchers

| Use Case | How This Tool Helps |
|---|---|
| Literature review | Crawls arxiv, Semantic Scholar, PubMed and synthesizes findings across papers |
| Thesis background research | Generates structured multi-dimensional reports with cited sources |
| Fact-checking | Cross-references the same claim across multiple independent sources |
| Current events study | Retrieves and summarizes the latest news with source attribution |

A student researching **"impact of transformer architecture on NLP"** gets a report covering the original paper, recent improvements, benchmark comparisons, and real-world adoption — all cited and structured, not just a summary of one article.

---

### 🏢 Business & Industry

| Industry | Use Case |
|---|---|
| **Market Research** | Competitive analysis across news, company blogs, and financial reports |
| **Legal** | Case law research, regulatory monitoring, policy change tracking |
| **Finance** | Real-time aggregation of market news, earnings reports, analyst commentary |
| **Healthcare** | Clinical research summaries, drug interaction studies, medical news monitoring |
| **Consulting** | Rapid domain research for client engagements without expensive analyst tools |
| **Journalism** | Background research, source verification, story angle discovery |
| **Product Teams** | Monitoring competitor launches, user sentiment, and industry trends |

A market analyst researching **"EV battery supply chain disruptions 2026"** gets a structured report covering causes, key players, financial impact, and industry outlook — sourced from Reuters, Bloomberg, and technical publications simultaneously.

---

### 🔬 AI / ML Engineers & Data Scientists

| Use Case | How This Tool Helps |
|---|---|
| Keeping up with research | Crawls arxiv, HuggingFace, and OpenAI blogs for latest papers |
| Dataset discovery | Searches Papers with Code and academic sources for benchmarks |
| Model comparison research | Multi-dimensional analysis of different approaches |
| RAG pipeline prototyping | Ready-to-extend architecture for custom embedding and retrieval experiments |

---

### 🏛️ Government & Policy

| Use Case | How This Tool Helps |
|---|---|
| Policy research | Aggregates perspectives from think tanks, news, and official sources |
| Legislative monitoring | Tracks bills, amendments, and political commentary across sources |
| Geopolitical analysis | Multi-source synthesis of international events and implications |

---

### 🔒 Privacy-Conscious Users & Organizations

Unlike cloud-based research tools (Perplexity, ChatGPT with web browsing), this tool:

- Runs **entirely on your own machine** — no queries sent to external AI APIs
- Uses a **self-hosted SearXNG** — no search history sent to Google or Bing
- Stores all data in a **local ChromaDB** — no cloud vector database
- Uses **Ollama** for inference — model weights run locally, nothing leaves your network

This makes it suitable for organizations with strict data privacy requirements, classified research environments, or users who simply do not want their research queries logged by third-party services.

---

## Scalability

The current architecture is designed for single-user local research. Here is a clear picture of its current limits and the concrete path to scaling each component.

---

### Current Capacity (Single User, Local)

| Component | Current Limit | Bottleneck |
|---|---|---|
| Concurrent users | 1 | Single event loop, shared ChromaDB |
| Queries per session | Unlimited | RAM for model loading |
| URLs per research session | ~12 (4 dimensions × 3 URLs) | IP rate limiting |
| Chunks stored | Unlimited (disk-bound) | Disk space only |
| LLM throughput | 1 request at a time | Ollama single-thread inference |
| Crawl speed | ~3-4 min per session | JS render wait times |

---

### Scaling the Search Layer

**Current:** Single SearXNG instance on localhost, 3 URLs per sub-query, 2-3s delay between searches.

**Scale to:** Multiple SearXNG instances with load balancing, rotating instances per query to distribute request load.

```python
# Current — single instance
SEARXNG_INSTANCES = ["http://localhost:8080"]

# Scaled — multiple instances, round-robin selection
SEARXNG_INSTANCES = [
    "http://searxng-1:8080",
    "http://searxng-2:8080",
    "http://searxng-3:8080",
]
```

Deploy multiple SearXNG containers via Docker Compose:

```yaml
# docker-compose.yml
services:
  searxng-1:
    image: searxng/searxng:latest
    ports: ["8080:8080"]

  searxng-2:
    image: searxng/searxng:latest
    ports: ["8081:8080"]

  searxng-3:
    image: searxng/searxng:latest
    ports: ["8082:8080"]
```

**Result:** 3× search throughput, independent rate limit pools per instance, no single point of failure.

---

### Scaling the Crawl Layer

**Current:** Sequential JS crawling (one at a time), 35s timeout per JS site, max 4 JS URLs.

**Scale to:** Distributed crawling with a task queue.

```
Current:   app.py → crawl4ai (single process)
Scaled:    FastAPI → Celery task queue → N crawl4ai workers
```

Implementation path:

```python
# Add Celery worker per crawl task
from celery import Celery

app = Celery('crawler', broker='redis://localhost:6379/0')

@app.task
def crawl_single_url(url: str, prompt: str) -> dict:
    # crawl4ai logic here
    pass

# Dispatch all URLs in parallel
tasks = [crawl_single_url.delay(url, prompt) for url in urls]
results = [t.get(timeout=60) for t in tasks]
```

**Result:** N URLs crawled simultaneously instead of sequentially, linear throughput scaling with worker count.

---

### Scaling the Vector Database

**Current:** Local ChromaDB (`PersistentClient`) — single file, single user, no concurrent writes.

**Scale to:** ChromaDB server mode for multi-user access, or migrate to a dedicated vector database.

```python
# Current — local file
chroma_client = chromadb.PersistentClient(path="./web-search-llm-db")

# Scaled — ChromaDB HTTP server (supports concurrent connections)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Production — Qdrant or Weaviate for horizontal scaling
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
```

| Vector DB | Best For | Scaling Model |
|---|---|---|
| ChromaDB (current) | Single user, local | Vertical only |
| ChromaDB HTTP server | Small team (2-10 users) | Single server |
| Qdrant | Production multi-user | Horizontal clustering |
| Weaviate | Enterprise | Kubernetes native |
| Pinecone | Serverless | Fully managed |

---

### Scaling the LLM Layer

**Current:** Single Ollama instance, `gemma3:1b`, one request at a time, 120s timeout.

**Scale to:** Multiple Ollama instances behind a load balancer, or migrate to vLLM for batched inference.

```
Current:   app.py → Ollama (1 instance, 1 request at a time)
Scaled:    app.py → Nginx → [Ollama-1, Ollama-2, Ollama-3]
```

For higher throughput, replace Ollama with **vLLM** which supports continuous batching:

```python
# Current — Ollama
response = requests.post("http://localhost:11434/api/generate", ...)

# Scaled — vLLM (OpenAI-compatible API, batched inference)
response = requests.post("http://localhost:8000/v1/completions", ...)
```

| LLM Server | Concurrent Requests | Best Model Size | Hardware |
|---|---|---|---|
| Ollama (current) | 1 | Up to 8B | CPU or consumer GPU |
| Ollama (parallel) | 2-4 | Up to 8B | 16-32 GB RAM |
| vLLM | 10-100+ | 7B-70B | NVIDIA A100/H100 |
| TGI (HuggingFace) | 10-50+ | 7B-70B | NVIDIA GPU cluster |

---

### Scaling the Application Layer

**Current:** Streamlit single-page app, single event loop, no session isolation.

**Scale to:** FastAPI backend + React/Next.js frontend + proper session management.

```
Current architecture:
  Browser → Streamlit (all-in-one)

Scaled architecture:
  Browser → Next.js frontend
              → FastAPI backend
                → Redis (session state)
                → Celery (task queue)
                → ChromaDB / Qdrant
                → vLLM
                → SearXNG cluster
```

---

### Scaling Summary

| Component | Now | 10 Users | 100 Users | 1000 Users |
|---|---|---|---|---|
| Search | 1 SearXNG | 3 SearXNG | SearXNG cluster | SearXNG + proxy rotation |
| Crawling | Sequential | Celery 4 workers | Celery 20 workers | Distributed Scrapy cluster |
| Vector DB | ChromaDB file | ChromaDB HTTP | Qdrant single node | Qdrant cluster |
| LLM | Ollama 1B | Ollama 2 instances | vLLM 7B | vLLM cluster / API gateway |
| App server | Streamlit | FastAPI + Streamlit | FastAPI + React | Kubernetes + FastAPI |
| Session state | None | Redis | Redis cluster | Redis Sentinel |

---

### Cloud Deployment Path

For organizations wanting to deploy this tool to a team or production environment:

```
Step 1 — Containerize
  docker build -t llm-research-tool .
  docker-compose up (Streamlit + ChromaDB + SearXNG)

Step 2 — Add GPU server for Ollama
  Deploy Ollama on a GPU instance (NVIDIA T4 / A10)
  Point OLLAMA_BASE_URL to the GPU server IP

Step 3 — Separate the vector database
  Deploy ChromaDB HTTP server or Qdrant on a dedicated instance
  Update chroma_client connection string

Step 4 — Add authentication
  Deploy behind a reverse proxy (Nginx + basic auth or OAuth)
  Add Streamlit secrets for API keys

Step 5 — Monitor
  Add Prometheus + Grafana for request latency and error tracking
  Log all research sessions to a structured database
```

---


## Author

**Your Name**
GitHub: https://github.com/gitakhileshyadav
LinkedIn: https://www.linkedin.com/in/akhilesh-yadav/

---

*Built with crawl4ai · ChromaDB · Ollama · SearXNG · Streamlit*