import os                                        # 1. env vars first
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"
os.environ["POSTHOG_DISABLED"]     = "true"
os.environ["POSTHOG_API_KEY"]      = ""

import sys

# Ensure the project root is importable so package imports like
# "from app.crawler import ..." work no matter where streamlit is invoked from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio          #2.  stdlib import
from urllib.parse import urlparse

import streamlit as st                           # 3. streamlit import

st.set_page_config(                              # 4. FIRST streamlit command
    page_title="Deep Research",
    page_icon="🧪",
    layout="wide"
)
import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ✅ Windows requires ProactorEventLoop for Playwright subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ── Local modules ─────────────────────────────────────────────────────────────

from app.config import EMBEDDING_MODEL, OLLAMA_MODEL
from app.crawler import crawl_webpages, curate_urls, detect_query_type
from app.llm import (
    _answer_directly,
    expand_query,
    generate_dimension_answer,
    link_citations,
    synthesize_final_report,
)
from app.robots import check_robots_txt
from app.search import async_multi_search, is_safe_url
from app.vectordb import add_to_vector_database, get_vector_collections, reset_collection


# ── Lab theme — light ML-lab palette, instrument-panel chrome ─────────────────
# Display-only: one global stylesheet, injected before any UI renders.

LAB_CSS = """
<style>
:root {
    --lab-bg: #f5f7f9;
    --lab-panel: #ffffff;
    --lab-line: #dde5ec;
    --lab-ink: #1f2933;
    --lab-muted: #64748b;
    --lab-accent: #0e7c86;
    --lab-accent-soft: #e6f4f5;
    --lab-cyan: #2b8a9e;
    --lab-ok: #2e9e6b;
    --lab-warn: #b45309;
    --lab-danger: #dc4c3e;
    --lab-mono: ui-monospace, "Cascadia Mono", "JetBrains Mono", Consolas, "Courier New", monospace;
    --lab-sans: -apple-system, "Segoe UI", "Inter", Roboto, sans-serif;
}

/* ── App canvas: graph-paper grid ─────────────────────────────────────────── */
.stApp {
    background-color: var(--lab-bg);
    background-image:
        linear-gradient(to right, rgba(30, 60, 90, 0.04) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(30, 60, 90, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1, h2, h3 { font-family: var(--lab-sans); color: var(--lab-ink); letter-spacing: 0.01em; }

/* ── Micro label: uppercase mono tag above sections/inputs ────────────────── */
.lab-label {
    font-family: var(--lab-mono);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--lab-muted);
    margin: 0.4rem 0 0.15rem 0;
}

/* ── Mono readout line ────────────────────────────────────────────────────── */
.lab-mono {
    font-family: var(--lab-mono);
    font-size: 0.82rem;
    color: var(--lab-ink);
}

/* ── Instrument plate (app header) ────────────────────────────────────────── */
.lab-plate {
    background: var(--lab-panel);
    border: 1px solid var(--lab-line);
    border-left: 4px solid var(--lab-accent);
    border-radius: 8px;
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}
.lab-plate-title {
    font-family: var(--lab-sans);
    font-weight: 800;
    font-size: 1.3rem;
    letter-spacing: 0.18em;
    color: var(--lab-ink);
    text-transform: uppercase;
}
.lab-plate-readouts {
    font-family: var(--lab-mono);
    font-size: 0.7rem;
    color: var(--lab-muted);
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.lab-plate-readouts b { color: var(--lab-accent); font-weight: 700; }

/* ── Status strip ─────────────────────────────────────────────────────────── */
.lab-strip {
    font-family: var(--lab-mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--lab-ok);
    margin: 2px 0 14px 2px;
}
.lab-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--lab-ok);
    margin-right: 6px;
    box-shadow: 0 0 0 3px rgba(46, 158, 107, 0.15);
}

/* ── Protocol tracker ─────────────────────────────────────────────────────── */
.lab-protocol { display: flex; gap: 8px; margin: 4px 0 12px 0; flex-wrap: wrap; }
.lab-step {
    flex: 1 1 150px;
    background: var(--lab-panel);
    border: 1px solid var(--lab-line);
    border-radius: 6px;
    padding: 8px 10px;
    min-width: 140px;
}
.lab-step-dot { font-size: 0.8rem; margin-right: 6px; }
.lab-step-num {
    font-family: var(--lab-mono);
    font-size: 0.66rem;
    color: var(--lab-muted);
    margin-right: 6px;
}
.lab-step-name {
    font-family: var(--lab-mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--lab-ink);
}
.lab-step-cap {
    display: block;
    font-family: var(--lab-mono);
    font-size: 0.62rem;
    color: var(--lab-muted);
    margin-top: 3px;
}
.lab-step.pending .lab-step-dot { color: var(--lab-line); }
.lab-step.active {
    border-color: var(--lab-accent);
    box-shadow: 0 0 0 1px var(--lab-accent) inset;
}
.lab-step.active .lab-step-dot { color: var(--lab-accent); }
.lab-step.active .lab-step-name { color: var(--lab-accent); }
.lab-step.complete .lab-step-dot { color: var(--lab-ok); }

/* ── Citation chips (references) ──────────────────────────────────────────── */
.lab-cite {
    display: inline-block;
    font-family: var(--lab-mono);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--lab-accent);
    background: var(--lab-accent-soft);
    border: 1px solid rgba(14, 124, 134, 0.25);
    border-radius: 4px;
    padding: 0 5px;
    margin-right: 6px;
}

/* ── Lab notebook header (report card) ────────────────────────────────────── */
.lab-notebook-head {
    font-family: var(--lab-mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--lab-accent);
    border-bottom: 1px dashed var(--lab-line);
    padding-bottom: 6px;
    margin-bottom: 10px;
}

/* ── Bordered container = report card ─────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-left: 3px solid var(--lab-accent) !important;
    background: var(--lab-panel);
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--lab-mono);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid var(--lab-line);
    border-radius: 6px;
    color: var(--lab-ink);
    background: var(--lab-panel);
}
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: var(--lab-accent);
    border-color: var(--lab-accent);
    color: #ffffff;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* ── Inputs ───────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    border: 1px solid var(--lab-line) !important;
    border-radius: 6px !important;
    background: var(--lab-panel);
    color: var(--lab-ink);
}
.stTextArea textarea:focus {
    border-color: var(--lab-accent) !important;
    box-shadow: 0 0 0 1px var(--lab-accent) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--lab-panel);
    border-right: 1px solid var(--lab-line);
}

/* ── Expanders ────────────────────────────────────────────────────────────── */
.stExpander {
    border: 1px solid var(--lab-line) !important;
    border-radius: 6px !important;
    background: var(--lab-panel);
}

/* ── Metrics: digital readouts ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--lab-panel);
    border: 1px solid var(--lab-line);
    border-radius: 6px;
    padding: 10px 12px;
}
[data-testid="stMetricValue"] { font-family: var(--lab-mono); }
[data-testid="stMetricLabel"] {
    font-family: var(--lab-mono);
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--lab-muted);
}

/* ── Progress: teal fill ──────────────────────────────────────────────────── */
.stProgress div[role="progressbar"] > div {
    background-color: var(--lab-accent) !important;
}

/* ── Inline code = specimen tags ──────────────────────────────────────────── */
code {
    font-family: var(--lab-mono) !important;
    background: var(--lab-accent-soft);
    color: var(--lab-accent);
    border-radius: 4px;
    padding: 1px 5px;
}

/* ── Captions ─────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: var(--lab-muted); font-size: 0.78rem; }
</style>
"""
st.markdown(LAB_CSS, unsafe_allow_html=True)


# ── UI helpers ────────────────────────────────────────────────────────────────

TELEMETRY_KEYS = [
    "tel_dimensions", "tel_specimens", "tel_crawl_log",
    "tel_vector_db", "tel_top_chunks",
]

PROTOCOL_STEPS = [
    ("01", "INTAKE", "hypothesis parsed"),
    ("02", "EXPANSION", "query → dimensions"),
    ("03", "COLLECTION", "search · crawl · index"),
    ("04", "ASSAYS", "dimension readings"),
    ("05", "SYNTHESIS", "final report"),
]


def _protocol_html(completed: int) -> str:
    """
    Protocol tracker HTML: `completed` steps done, the next one active.
    Rendered through an st.empty slot so it updates in place as phases finish.
    """
    cells = []
    for i, (num, name, cap) in enumerate(PROTOCOL_STEPS):
        if i < completed:
            state, dot = "complete", "●"
        elif i == completed and completed < len(PROTOCOL_STEPS):
            state, dot = "active", "◐"
        else:
            state, dot = "pending", "○"
        cells.append(
            f'<div class="lab-step {state}">'
            f'<span class="lab-step-dot">{dot}</span>'
            f'<span class="lab-step-num">{num}</span>'
            f'<span class="lab-step-name">{name}</span>'
            f'<span class="lab-step-cap">{cap}</span>'
            f'</div>'
        )
    return '<div class="lab-protocol">' + "".join(cells) + "</div>"


def _parse_user_urls(raw: str) -> tuple[list[str], list[str]]:
    """
    Parse the custom-website text area into a clean URL list.

    One URL per line; lines without a scheme get https:// prepended.
    Returns (valid_urls, invalid_urls) — both deduped, order preserved.
    Lines with spaces or dotless hosts are rejected; is_safe_url() blocks
    localhost/private-IP/metadata hosts (SSRF guard).
    """
    valid, invalid, seen = [], [], set()
    for line in (raw or "").splitlines():
        url = line.strip()
        if not url:
            continue
        if " " in url:
            invalid.append(url)
            continue
        if "://" not in url:
            url = f"https://{url}"
        key = url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        host = urlparse(url).hostname or ""
        if is_safe_url(url) and "." in host:
            valid.append(url)
        else:
            invalid.append(url)
    return valid, invalid


def _result_content_len(result) -> int:
    if getattr(result, "fit_markdown", None):
        return len(result.fit_markdown)
    if getattr(result, "markdown", None):
        return len(str(result.markdown))
    return 0


def _render_user_website_status(
    user_urls: list[str],
    invalid_user_urls: list[str],
    allowed_urls: list[str],
    crawl_results: list,
) -> None:
    """
    Always-visible Specimen Status for the websites the user provided —
    kept separate from the spectrum (search) telemetry.

    Shows, per URL: ⛔ containment failure (robots), ✅ cultivated (content
    length + preview), ⚠️ degraded (JS-only shell), or ❌ non-viable, plus
    ⚠️ for rejected inputs.
    """
    if not user_urls and not invalid_user_urls:
        return

    st.divider()
    st.subheader("🧫 Specimen Status")
    st.caption("Live status for the samples you added — tracked separately from spectrum results.")

    allowed_set = set(allowed_urls or [])
    result_map = {
        r.url: r for r in (crawl_results or [])
        if getattr(r, "url", None)
    }

    for url in invalid_user_urls:
        st.markdown(
            f"⚠️ `{url}` — REJECTED INPUT (invalid or unsafe URL; "
            f"localhost/private-network hosts are blocked)"
        )

    for url in user_urls:
        if url not in allowed_set:
            st.markdown(f"⛔ `{url}` — CONTAINMENT FAILURE (blocked by robots.txt)")
            continue
        result = result_map.get(url)
        content_len = _result_content_len(result) if result else 0
        if result is None or content_len == 0:
            st.markdown(f"❌ `{url}` — NON-VIABLE (extraction failed)")
            continue
        if content_len < 100:
            # Matches the vectordb minimum — anything shorter is a JS-only
            # shell (e.g. msn.com), not real article content.
            st.markdown(
                f"⚠️ `{url}` — DEGRADED ({content_len} chars; "
                f"site renders content only via JS and blocks crawlers)"
            )
            continue
        with st.expander(f"✅ `{url}` — CULTIVATED · {content_len} chars"):
            content = result.fit_markdown or str(result.markdown or "")
            st.text(content[:500])


def _render_crawl_log(results: list) -> None:
    """Per-URL crawl telemetry (sidebar: Crawl log)."""
    st.subheader("🧬 Crawl Log")
    for r in results:
        ok = getattr(r, "success", False) and _result_content_len(r) > 0
        with st.expander(f"{'✅' if ok else '❌'} {r.url}"):
            st.write(f"**Extracted:** {_result_content_len(r)} chars")
            content = r.fit_markdown or str(r.markdown or "")
            if content:
                st.text(content[:500])


def _render_vector_db_panel(collection, session_id) -> None:
    """Vector DB telemetry (sidebar: Vector DB)."""
    st.subheader("🗄 Specimen Index")
    session_chunks = 0
    if session_id:
        try:
            stored = collection.get(where={"session_id": session_id}, include=["metadatas"])
            session_chunks = len(stored.get("ids") or [])
        except Exception as e:
            print(f"[UI] Session chunk count failed: {e}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks This Session", session_chunks)
    c2.metric("Total Chunks in DB", collection.count())
    c3.metric("Session ID", session_id or "—")


def _render_top_chunks(collection, prompt: str, n: int = 5) -> None:
    """The exact chunks the LLM reads first (sidebar: Top chunks)."""
    total = collection.count()
    if total == 0:
        return
    st.subheader("🔎 Top Chunks (LLM Context)")
    try:
        q = collection.query(query_texts=[prompt], n_results=min(n, total))
        for i, (doc, meta) in enumerate(zip(q["documents"][0], q["metadatas"][0])):
            with st.expander(f"Chunk {i+1} · {meta.get('source', 'Unknown')}"):
                st.caption(
                    f"index {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}"
                    f" · session {meta.get('session_id', '?')}"
                    f" · crawled {meta.get('crawled_at', '?')}"
                )
                st.write(doc)
    except Exception as e:
        st.error(f"Failed to query vector DB: {e}")
        print(f"[UI] Top-chunks query error: {e}")


# ── Streamlit UI ──────────────────────────────────────────────────────────────

async def run():

    # ✅ Initialize DB first
    collection, chroma_client = get_vector_collections()

    # ── Sidebar: Lab Controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="lab-label">🧰 Instrument Telemetry</p>', unsafe_allow_html=True)
        c_all, c_none = st.columns(2)
        if c_all.button("All", key="tel_all", use_container_width=True):
            for k in TELEMETRY_KEYS:
                st.session_state[k] = True
            st.rerun()
        if c_none.button("None", key="tel_none", use_container_width=True):
            for k in TELEMETRY_KEYS:
                st.session_state[k] = False
            st.rerun()
        show_dimensions = st.checkbox("Dimensions", value=False, key="tel_dimensions")
        show_specimens = st.checkbox("Specimens (URLs found)", value=False, key="tel_specimens")
        show_crawl_log = st.checkbox("Crawl log", value=False, key="tel_crawl_log")
        show_vector_db = st.checkbox("Vector DB", value=False, key="tel_vector_db")
        show_top_chunks = st.checkbox("Top chunks (LLM context)", value=False, key="tel_top_chunks")
        st.caption(
            "Full control — each research component can be revealed "
            "independently. Off by default."
        )

        st.divider()
        st.markdown('<p class="lab-label">Specimen Storage</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="lab-mono">DB CHUNKS&nbsp;&nbsp;{collection.count():04d}</p>',
                    unsafe_allow_html=True)
        if st.button("⚠️ Purge Sample Storage", type="secondary", use_container_width=True):
            reset_collection(chroma_client)
            st.success("Storage purged. Please reload the page.")
            st.stop()

    # ── Main UI: instrument plate ─────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="lab-plate">
          <div class="lab-plate-title">🧪 Deep Research</div>
          <div class="lab-plate-readouts">
            <span>MODEL <b>{OLLAMA_MODEL}</b></span>
            <span>EMBED <b>{EMBEDDING_MODEL}</b></span>
            <span>ENGINE <b>SEARXNG</b></span>
          </div>
        </div>
        <div class="lab-strip"><span class="lab-dot"></span>Systems nominal</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sample preparation ────────────────────────────────────────────────────
    st.markdown('<p class="lab-label">Sample Preparation</p>', unsafe_allow_html=True)
    prompt = st.text_area(
        label="Research Hypothesis",
        placeholder="State your research question...",
        label_visibility="collapsed",
    )
    is_web_search = st.toggle(
        "Web Spectrum Acquisition",
        value=False,
        key="enable_web_search"         # ✅ key preserves toggle state
    )
    go = st.button("▶ Run Experiment", type="primary", use_container_width=True)

    # ── Manual sample intake ──────────────────────────────────────────────────
    st.markdown('<p class="lab-label">Manual Sample Intake (optional — one URL per line)</p>',
                unsafe_allow_html=True)
    user_urls_input = st.text_area(
        label="Manual Sample Intake",
        placeholder="https://example.com/article\nhttps://another.org/page",
        height=90,
        label_visibility="collapsed",
    )
    user_urls, invalid_user_urls = _parse_user_urls(user_urls_input)
    if user_urls:
        st.caption(
            f"🧫 {len(user_urls)} manual specimen(s) queued — cultivated "
            f"alongside spectrum results when web search is on."
        )

    if go and (prompt or user_urls):

        # ── Protocol tracker (updates in place as phases finish) ──────────────
        protocol_slot = st.empty()
        protocol_slot.markdown(_protocol_html(0), unsafe_allow_html=True)

        session_id  = None
        query_type  = detect_query_type(prompt) if prompt else "general"
        # LLM calls need a non-empty prompt even in URLs-only runs.
        prompt_for_llm = prompt or "Summarize the crawled websites."

        # INTAKE complete → EXPANSION active
        protocol_slot.markdown(_protocol_html(1), unsafe_allow_html=True)

        # ── Web search path ───────────────────────────────────────────────────
        if is_web_search and prompt:

            # ── Step 1: Expand query into sub-queries ─────────────────────────
            with st.spinner("🧫 Cultivating research dimensions..."):
                sub_queries = expand_query(prompt, query_type)

            # Always search the ORIGINAL topic verbatim first - the exact title
            # is what returns the truly relevant links (e.g. searching "Student
            # protest India 2026" brings up the Wikipedia/Reuters/Al Jazeera
            # pages). LLM dimensions supplement it, never replace it.
            prompt_key = prompt.strip().lower()
            search_queries = [prompt] + [
                q for q in sub_queries
                if q.strip().lower() != prompt_key
            ]

            # EXPANSION complete → COLLECTION active
            protocol_slot.markdown(_protocol_html(2), unsafe_allow_html=True)

            if show_dimensions:
                st.subheader("🧬 Assay Conditions")
                for i, q in enumerate(search_queries):
                    label = "Original topic" if i == 0 else f"Assay {i:02d}"
                    st.write(f"**{label}:** {q}")

            # ── Step 2: Search all dimensions asynchronously ──────────────────
            with st.spinner("📡 Scanning the web spectrum..."):
                search_results = await async_multi_search(
                    sub_queries=search_queries,
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

            # ── Step 3b: Pre-crawl curation ───────────────────────────────────
            # Drop non-trusted root homepages, single-token matches (impact.com
            # matching only "impact"), domain dupes, and overflow. Trusted
            # news/science/priority domains pass through unconditionally.
            all_urls = curate_urls(all_urls, prompt, query_type)

            # ── Step 3c: Append user-provided URLs ────────────────────────────
            # Custom URLs bypass curation entirely (explicit user intent) but
            # still go through the robots.txt + SSRF checks below.
            for u in user_urls:
                if u not in all_urls:
                    all_urls.append(u)

            if show_specimens:
                st.subheader("📡 Specimens Retrieved")
                st.write(f"**{len(all_urls)} unique specimen(s)** across all dimensions")
                for url in all_urls:
                    n_dims = len(url_to_queries.get(url, []))
                    st.write(f"- {url} _(matched {n_dims} dimension(s))_")

            if not all_urls:
                st.error("No specimens found across all dimensions.")
                st.stop()

            # ── Step 4: Robots.txt check ───────────────────────────────────────
            with st.spinner("🚧 Checking containment protocols..."):
                allowed_urls = check_robots_txt(all_urls)

            if not allowed_urls:
                st.error("All specimens rejected by containment protocols (robots.txt).")
                _render_user_website_status(user_urls, invalid_user_urls, allowed_urls, [])
                st.stop()

            # ── Step 5: Crawl all URLs ─────────────────────────────────────────
            with st.spinner(f"🧬 Extracting specimens ({len(allowed_urls)} pages)..."):
                results = await crawl_webpages(
                    urls=allowed_urls,
                    prompt=prompt,
                )
            # crawl_webpages can append None for URLs that fail even the
            # fallback fetch — drop them so downstream result.url never sees it.
            results = [r for r in results if r is not None]

            # ── Step 5b: Specimen status — always visible ──────────────────────
            _render_user_website_status(user_urls, invalid_user_urls, allowed_urls, results)

            # ── Step 6: Store in ChromaDB ──────────────────────────────────────
            session_id = add_to_vector_database(
                results=results,
                prompt=prompt_for_llm,
                collection=collection,
            )

            # COLLECTION complete → ASSAYS active
            protocol_slot.markdown(_protocol_html(3), unsafe_allow_html=True)

            # ── Step 6b: Telemetry panels (per-component, sidebar-controlled) ──
            if show_crawl_log:
                _render_crawl_log(results)
            if show_vector_db:
                _render_vector_db_panel(collection, session_id)
            if show_top_chunks:
                _render_top_chunks(collection, prompt_for_llm)

        # ── Custom-URLs-only path ─────────────────────────────────────────────
        # No prompt, but the user gave websites: crawl them directly — no
        # SearXNG search, no expansion, no curation (explicit user intent).
        elif user_urls:
            # Expansion N/A for manual-only runs → mark complete
            protocol_slot.markdown(_protocol_html(2), unsafe_allow_html=True)

            st.info(f"🧫 Cultivating {len(user_urls)} manual specimen(s) — spectrum acquisition off.")

            # ── Step 4: Robots.txt check ───────────────────────────────────────
            with st.spinner("🚧 Checking containment protocols..."):
                allowed_urls = check_robots_txt(user_urls)

            if not allowed_urls:
                st.error("All manual specimens rejected by containment protocols (robots.txt).")
                _render_user_website_status(user_urls, invalid_user_urls, allowed_urls, [])
                st.stop()

            # ── Step 5: Crawl all URLs ─────────────────────────────────────────
            with st.spinner(f"🧬 Extracting specimens ({len(allowed_urls)} pages)..."):
                results = await crawl_webpages(
                    urls=allowed_urls,
                    prompt=prompt,
                )
            results = [r for r in results if r is not None]

            # ── Step 5b: Specimen status — always visible ──────────────────────
            _render_user_website_status(user_urls, invalid_user_urls, allowed_urls, results)

            # ── Step 6: Store in ChromaDB ──────────────────────────────────────
            session_id = add_to_vector_database(
                results=results,
                prompt=prompt_for_llm,
                collection=collection,
            )

            # COLLECTION complete → ASSAYS active
            protocol_slot.markdown(_protocol_html(3), unsafe_allow_html=True)

            # ── Step 6b: Telemetry panels (per-component, sidebar-controlled) ──
            if show_crawl_log:
                _render_crawl_log(results)
            if show_vector_db:
                _render_vector_db_panel(collection, session_id)
            if show_top_chunks:
                _render_top_chunks(collection, prompt_for_llm)

        # ── Step 7: Check DB has data ─────────────────────────────────────────
        if collection.count() == 0:
            protocol_slot.markdown(_protocol_html(5), unsafe_allow_html=True)
            st.info("Storage empty — inference from base knowledge only.")
            _answer_directly(prompt_for_llm, query_type)
            st.stop()

        # ── Step 7b: Global source registry — stable citation numbers ────────
        # One number per source URL, consistent across every dimension and
        # the final report, so [Source N] in ANY sentence always maps to the
        # Nth entry of the "References" list. Built from the chunks actually
        # stored for this session (falls back to the whole DB when the session
        # stored nothing or when answering from an existing DB).
        try:
            if session_id:
                stored = collection.get(
                    where={"session_id": session_id}, include=["metadatas"]
                )
                stored_metas = stored.get("metadatas") or []
                if not stored_metas:
                    stored = collection.get(include=["metadatas"])
                    stored_metas = stored.get("metadatas") or []
            else:
                stored = collection.get(include=["metadatas"])
                stored_metas = stored.get("metadatas") or []
            source_urls = sorted({
                m.get("source", "") for m in stored_metas if m.get("source")
            })
        except Exception as e:
            print(f"[Research] Could not build source registry: {e}")
            source_urls = []
        source_ids = {url: i + 1 for i, url in enumerate(source_urls)}
        print(f"[Research] Source registry: {len(source_urls)} sources")

        # ── Step 8: Iterative generation — one answer per dimension ───────────
        st.divider()
        st.subheader("🔬 Experimental Report")
        # COLLECTION complete (index already held data) → ASSAYS active
        protocol_slot.markdown(_protocol_html(3), unsafe_allow_html=True)

        if is_web_search and 'sub_queries' in locals():
            dimensions = sub_queries
        elif prompt:
            # ✅ Toggle OFF — generate dimensions from existing DB
            dimensions = expand_query(prompt, query_type)
        else:
            # Custom-URLs-only run (no prompt) — expand_query("") would fail,
            # so use one generic dimension over the crawled content.
            dimensions = ["Summarize the key information from the crawled websites."]

        dimension_answers = []
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, sub_query in enumerate(dimensions):

            status_text.text(f"🧪 Running assay {i+1}/{len(dimensions)}: {sub_query}")

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
            # Labels use the GLOBAL source number so [Source N] means the
            # same URL in every dimension, in the synthesis, and in the
            # References list. A source missing from the registry (e.g. a
            # dimension fell back to old-DB chunks) is appended with the next
            # free number instead of breaking the numbering.
            if dim_docs:
                context_parts = []
                for doc, meta in zip(dim_docs, dim_metas):
                    source  = meta.get("source", "Unknown")
                    sid = source_ids.get(source)
                    if sid is None:
                        sid = len(source_urls) + 1
                        source_urls.append(source)
                        source_ids[source] = sid
                        print(f"[Research] Late source #{sid}: {source}")
                    context_parts.append(f"[Source {sid} — {source}]\n{doc[:800]}")
                dim_context = "\n\n".join(context_parts)

                if len(dim_context) > 6000:
                    dim_context = dim_context[:6000]
            else:
                dim_context = "No specific context found for this dimension."

            # ── Generate answer for this dimension ─────────────────────────────
            answer = generate_dimension_answer(
                original_prompt=prompt_for_llm,
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

            # ── Show assay reading as it's generated ───────────────────────────
            with st.expander(
                f"🧪 Assay {i+1:02d} · {sub_query}",
                expanded=(i == 0),     # expand first one by default
            ):
                st.write(link_citations(answer, source_urls))
                if dim_metas:
                    st.caption("Sources: " + ", ".join({
                        m.get("source", "") for m in dim_metas
                    }))

            progress_bar.progress((i + 1) / len(dimensions))

        # ASSAYS complete → SYNTHESIS active
        protocol_slot.markdown(_protocol_html(4), unsafe_allow_html=True)
        status_text.text("📊 Compiling experimental report...")

        # ── Step 9: Synthesize all dimensions into final report ────────────────
        with st.spinner("📊 Compiling experimental report..."):
            final_report = synthesize_final_report(
                original_prompt=prompt_for_llm,
                dimension_answers=dimension_answers,
                query_type=query_type,
                source_urls=source_urls,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Experiment complete")
        protocol_slot.markdown(_protocol_html(5), unsafe_allow_html=True)

        st.divider()

        # ── Final report: lab-notebook card ────────────────────────────────────
        with st.container(border=True):
            st.markdown('<p class="lab-notebook-head">Experimental Report · Final Reading</p>',
                        unsafe_allow_html=True)
            st.write(link_citations(final_report, source_urls))

        # ── Step 10: References ────────────────────────────────────────────────
        # Numbered in the SAME order as the [Source N] citations, so the Nth
        # chip here is exactly what [Source N] in the report points to.
        st.subheader("📚 References")
        for i, url in enumerate(source_urls):
            if url:
                st.markdown(
                    f'<span class="lab-cite">{i+1:02d}</span>[{url}]({url})',
                    unsafe_allow_html=True,
                )

# ── Entry Point ───────────────────────────────────────────────────────────────

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())
else:
    asyncio.run(run())
