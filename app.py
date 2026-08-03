import os                                        # 1. env vars first
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"
os.environ["POSTHOG_DISABLED"]     = "true"
os.environ["POSTHOG_API_KEY"]      = ""

import asyncio          #2.  stdlib import
import sys

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

# ── Local modules ─────────────────────────────────────────────────────────────

from crawler import crawl_webpages, curate_urls, detect_query_type
from llm import (
    _answer_directly,
    expand_query,
    generate_dimension_answer,
    synthesize_final_report,
)
from robots import check_robots_txt
from search import async_multi_search
from vectordb import add_to_vector_database, get_vector_collections, reset_collection


# ── Streamlit UI ──────────────────────────────────────────────────────────────

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

            # Always search the ORIGINAL topic verbatim first - the exact title
            # is what returns the truly relevant links (e.g. searching "Student
            # protest India 2026" brings up the Wikipedia/Reuters/Al Jazeera
            # pages). LLM dimensions supplement it, never replace it.
            prompt_key = prompt.strip().lower()
            search_queries = [prompt] + [
                q for q in sub_queries
                if q.strip().lower() != prompt_key
            ]

            st.subheader("🔍 Research Dimensions")
            for i, q in enumerate(search_queries):
                label = "Original topic" if i == 0 else f"Dimension {i}"
                st.write(f"**{label}:** {q}")

            # ── Step 2: Search all dimensions asynchronously ──────────────────
            with st.spinner("Searching all dimensions (with delays to avoid blocking)..."):
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
