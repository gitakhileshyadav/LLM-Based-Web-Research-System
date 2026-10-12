"""LLM layer — all Ollama interactions: query expansion, iterative dimension
generation, final report synthesis, direct answers, and health checks."""

import re
import time

import requests
import streamlit as st

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, SYSTEM_PROMPTS


# ── Query Expansion ───────────────────────────────────────────────────────────

def expand_query(prompt: str, query_type: str) -> list[str]:
    """
    Generate sub-queries covering different dimensions of the original query.
    Uses the configured Ollama model to produce 4 targeted sub-queries.
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


# ── Pseudo-Relevance Feedback (PRF) query refinement ─────────────────────────

def refine_queries_from_results(
    prompt: str,
    evidence: list[dict],
    query_type: str = "general",
) -> list[str]:
    """
    Pseudo-relevance feedback: generate refined search queries grounded in
    the ACTUAL titles/snippets SearXNG returned for the original prompt,
    instead of the model's (training-cutoff) memory.

    The refinement prompt includes today's date so the LLM can add recency
    qualifiers ("latest", "2026") to time-sensitive topics.

    Returns up to 4 refined queries, or [] on any failure (the caller then
    falls back to the memory-based expand_query).
    """
    if not evidence:
        return []

    # ── Format evidence — numbered titles + snippets ──────────────────────────
    lines = []
    for i, r in enumerate(evidence, start=1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        snippet = snippet[:280]              # keep the prompt compact
        lines.append(f"{i}. TITLE: {title} | SNIPPET: {snippet}")
    evidence_block = "\n".join(lines)

    today = time.strftime("%Y-%m-%d")

    system_prompt = (
        "You generate precise search queries grounded in search evidence. "
        "Return only the queries, nothing else."
    )

    refinement_prompt = f"""You are a search-query specialist. Today's date is {today}.

The user's original query is:
"{prompt}"

Below are the TOP LIVE RESULTS returned by a search engine for that exact
query right now:

{evidence_block}

Based ONLY on the terms, entities, product names, people, version numbers and
phrasing you can actually SEE in these live results (not your memory), write
3-4 refined search queries that will dig deeper into this exact topic:

- Reuse the exact names, spellings and wording the live results use.
- If the results indicate a time-sensitive topic, include a recency qualifier
  with the current date or year (e.g. "latest", "{today[:4]}").
- Each query must be specific enough to retrieve DIFFERENT pages than the
  ones already shown.

Return ONLY the queries, one per line, no numbering, no explanation."""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": refinement_prompt,
                "system": system_prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # ── Parse refined queries — one per line, deduped, original dropped ──
        seen = set()
        refined = []
        for line in raw.strip().splitlines():
            q = line.strip().strip('"').strip()
            if not q or len(q) <= 10:
                continue
            if q.lower() == prompt.strip().lower() or q.lower() in seen:
                continue
            seen.add(q.lower())
            refined.append(q)
            if len(refined) >= 4:        # same fan-out cap as expand_query
                break

        print(f"[PRF] Refined {len(refined)} queries from {len(evidence)} live results:")
        for i, q in enumerate(refined):
            print(f"  {i+1}. {q}")

        return refined

    except Exception as e:
        print(f"[PRF] Refinement error: {e} — caller will fall back to memory expansion")
        return []


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
- Cite sources using [Source N] notation, where N is the source number shown in the context labels (e.g. "[Source 2 — url]")
- ONLY cite [Source N] numbers that appear in the context — never invent a number that is not labelled there
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
    source_urls: list[str] | None = None,
) -> str:
    """
    Combine all dimension answers into a cohesive final research report.

    `source_urls` is the global source registry (index N-1 = [Source N]).
    Passing it lets the model verify citation numbers against the real list,
    so every [Source N] in the final report maps to the Nth source URL.
    """
    sections = "\n\n".join([
        f"=== Dimension {i+1}: {d['sub_query']} ===\n{d['answer']}"
        for i, d in enumerate(dimension_answers)
        if d['answer']
    ])

    source_list_block = ""
    citation_rules = "- Maintain all source citations"
    if source_urls:
        source_list_block = "\nSource list (the ONLY valid citation numbers):\n" + "\n".join(
            f"[Source {i+1}] {url}" for i, url in enumerate(source_urls)
        ) + "\n"
        citation_rules = (
            "- Maintain all source citations\n"
            "- Use ONLY the [Source N] numbers from the source list above — "
            "if a section cites a number that is not in the list, remove or "
            "correct that citation; never invent new numbers"
        )

    system_prompt = f"""You are a senior research editor creating a final comprehensive report.
Your task is to synthesize multiple research sections into one cohesive, well-structured report.

Instructions:
- Combine all sections into a flowing, well-organized report
- Use clear headers for each major section
- Remove repetition but preserve all unique information
- Ensure logical flow between sections
- Add a brief Executive Summary at the start
- Add Key Takeaways at the end
{citation_rules}
- The final report should be comprehensive and detailed"""

    full_prompt = f"""Original Research Question: {original_prompt}
{source_list_block}
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


# ── Citation Post-Processing ──────────────────────────────────────────────────

# Matches every citation variant the LLM produces:
#   [Source 1]  [Sources 1, 2]  [Source 1, 2]  [Source 1, Source 2]
# The (?!\() lookahead skips citations that were ALREADY converted to
# markdown links, so link_citations() is safe to run twice.
_CITATION_RE = re.compile(
    r"\[\s*Sources?\s+\d+(?:\s*,\s*(?:Sources?\s+)?\d+)*\s*\](?!\()"
)


def link_citations(text: str, source_urls: list[str]) -> str:
    """
    Turn [Source N] citations in LLM output into clickable markdown links
    pointing at the Nth source URL, and strip hallucinated citations whose
    number is outside the source registry (they point at nothing real).

    With source_urls = ["a", "b", "c"]:
      "...spiked in 2026 [Source 2]."  ->  "...spiked in 2026 [Source 2](b)."
      "...unclear [Source 9]."         ->  "...unclear."      (number removed)
    """
    if not text or not source_urls:
        return text

    def _repl(match: re.Match) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
        valid = [n for n in numbers if 1 <= n <= len(source_urls)]
        if not valid:
            return ""
        links = [f"[Source {n}]({source_urls[n - 1]})" for n in valid]
        return links[0] if len(links) == 1 else "[" + ", ".join(links) + "]"

    return _CITATION_RE.sub(_repl, text)


def check_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def query_llm(prompt: str, context: str, query_type: str) -> str:
    """
    Send retrieved context + prompt to the configured Ollama model.
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
