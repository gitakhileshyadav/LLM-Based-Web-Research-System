"""Vector database layer — ChromaDB persistent storage with chunking,
structured metadata, stale-chunk cleanup, and Streamlit status display."""

import os

# ✅ Telemetry guards — must run before chromadb is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY",     "false")
os.environ.setdefault("POSTHOG_DISABLED",     "true")
os.environ.setdefault("POSTHOG_API_KEY",      "")

import re
import tempfile
import time

import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from crawl4ai.models import CrawlResult
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_CHUNKS_PER_URL,
    OLLAMA_BASE_URL,
)


def get_vector_collections() -> tuple[chromadb.Collection, chromadb.Client]:
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return (
        chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
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


def reset_collection(chroma_client: chromadb.Client) -> None:
    """
    Delete and recreate the collection — use once to clear stale schema chunks.
    """
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )

    total_chunks = 0

    for result in results:
        documents, metadatas, ids = [], [], []

        # ── Extract content ───────────────────────────────────────────────────
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
