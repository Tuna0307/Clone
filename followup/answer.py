"""
Conversational answer generation for artifact-first follow-up retrieval.
"""

from __future__ import annotations

import json
import re
from typing import Any

from followup.context import (
    FOLLOWUP_HIGH_ANOMALY_THRESHOLD,
    AnalysisContext,
    EvidenceItem,
    FollowupIntent,
    _markdown_table_cell,
)
from followup.intent import (
    _BROAD_ISSUE_TERMS,
    _SUMMARY_BROAD_MARKERS,
    _format_chat_history,
    _get_followup_llm,
)
from pipeline.progress import ProgressCallback
from followup.sources import (
    FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS,
    _debug_evidence_candidates,
    _debug_ref_candidates,
    _faiss_semantic_candidates,
    _is_api_followup_mode,
    _metadata_candidates,
    _preview_text,  # noqa: F401 - re-exported for existing tests/importers
    _raw_log_candidates,
    _vector_store_candidates,
)

DEFAULT_TOP_K_RESULTS = 10
FOLLOWUP_MAX_EVIDENCE_ITEMS = 12
FOLLOWUP_LLM_MAX_TOKENS = 4096
FOLLOWUP_EVIDENCE_BUDGET_CHARS = 40_000

_SOURCE_WEIGHTS = {
    "faiss": 1.0,
    "metadata": 0.9,
    "raw_log": 0.75,
    "vector_store": 0.7,
    "debug": 0.65,
}

_SOURCE_CAPS = {
    "faiss": 4,
    "metadata": 4,
    "raw_log": 3,
    "vector_store": 2,
    "debug": 2,
}


def _intent_terms(intent: FollowupIntent, query: str) -> list[str]:
    """
    Build consolidated retrieval terms from intent + query.

    Args:
        intent: Parsed intent
        query: User query text

    Returns:
        Deduplicated term list
    """
    query_tokens = re.findall(r"[A-Za-z0-9_./:-]{3,}", query)
    merged = [*intent.entities, *intent.primary_keys, *intent.must_include, *query_tokens]

    seen: set[str] = set()
    deduped: list[str] = []
    for token in merged:
        normalized = token.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(token.strip())

    query_lc = query.lower()
    is_broad_summary = (
        intent.ask_type in {"summary", "errors", "anomalies"}
        and any(marker in query_lc for marker in _SUMMARY_BROAD_MARKERS)
    )
    if is_broad_summary:
        for term in _BROAD_ISSUE_TERMS:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)

    return deduped[:30]


def _rank_and_select_evidence(
    intent: FollowupIntent,
    candidates: list[EvidenceItem],
    top_k: int,
) -> list[EvidenceItem]:
    """
    Rank evidence with weighted fusion and source diversity caps.

    Args:
        intent: Parsed follow-up intent
        candidates: Candidate evidence list
        top_k: Final item count target

    Returns:
        Selected evidence list
    """
    if not candidates:
        return []

    lowered_entities = [
        item.lower()
        for item in intent.entities + intent.must_include + intent.primary_keys
    ]
    intent_text = " ".join([intent.notes, *intent.entities, *intent.must_include]).lower()
    broad_summary_mode = (
        intent.ask_type in {"summary", "errors", "anomalies"}
        and any(marker in intent_text for marker in _SUMMARY_BROAD_MARKERS)
    )

    scored: list[tuple[float, EvidenceItem]] = []
    for item in candidates:
        score = item.relevance * _SOURCE_WEIGHTS.get(item.source, 0.5)
        lowered_text = item.raw_text.lower()
        for entity in lowered_entities:
            if entity and entity in lowered_text:
                score += 0.08
        if item.anomaly_score >= FOLLOWUP_HIGH_ANOMALY_THRESHOLD:
            score += 0.08
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[EvidenceItem] = []
    source_counts = {key: 0 for key in _SOURCE_CAPS}
    file_counts: dict[str, int] = {}

    for _, item in scored:
        cap = _SOURCE_CAPS.get(item.source, 2)
        if broad_summary_mode and item.source in {"faiss", "metadata"}:
            cap = min(cap, 3)
        if source_counts.get(item.source, 0) >= cap:
            continue
        if broad_summary_mode and file_counts.get(item.file_name, 0) >= 2:
            continue
        selected.append(item)
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        file_counts[item.file_name] = file_counts.get(item.file_name, 0) + 1
        if len(selected) >= max(1, min(top_k, FOLLOWUP_MAX_EVIDENCE_ITEMS)):
            break

    return selected


def _build_evidence_table(selected: list[EvidenceItem]) -> str:
    """
    Build markdown evidence table.

    Args:
        selected: Evidence items

    Returns:
        Markdown table
    """
    if not selected:
        return "No evidence items were selected."

    lines = [
        "### Evidence Table",
        "| REF ID | Source | File | Relevance | Anomaly | Excerpt |",
        "|---|---|---|---:|---:|---|",
    ]

    for item in selected:
        lines.append(
            "| "
            f"{_markdown_table_cell(item.evidence_id)} | {_markdown_table_cell(item.source)} | "
            f"{_markdown_table_cell(item.file_name)} | {item.relevance:.3f} | "
            f"{item.anomaly_score:.3f} | {_markdown_table_cell(item.excerpt)} |"
        )

    return "\n".join(lines)


def _build_evidence_block_for_prompt(selected: list[EvidenceItem]) -> str:
    """
    Build evidence text block with explicit character budgeting.

    Args:
        selected: Ranked evidence items

    Returns:
        Evidence block constrained by FOLLOWUP_EVIDENCE_BUDGET_CHARS
    """
    evidence_lines: list[str] = []
    used_chars = 0
    for item in selected:
        snippet = item.raw_text[:FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS]
        line = f"[{item.evidence_id}] ({item.source}) {snippet}"
        projected = used_chars + len(line) + 2
        if projected > FOLLOWUP_EVIDENCE_BUDGET_CHARS and evidence_lines:
            break
        if projected > FOLLOWUP_EVIDENCE_BUDGET_CHARS:
            line = line[:FOLLOWUP_EVIDENCE_BUDGET_CHARS]
            evidence_lines.append(line)
            break
        evidence_lines.append(line)
        used_chars = projected
    return "\n\n".join(evidence_lines)


def _generate_conversational_answer(
    context: AnalysisContext,
    query: str,
    intent: FollowupIntent,
    selected: list[EvidenceItem],
    chat_history: list[dict[str, str]] | None,
) -> str:
    """
    Generate grounded conversational answer using selected evidence.

    Args:
        context: Active analysis context
        query: User query
        intent: Parsed intent
        selected: Selected evidence
        chat_history: Prior session turns

    Returns:
        Final markdown response
    """
    llm = _get_followup_llm()

    history_text = _format_chat_history(chat_history)
    evidence_block = _build_evidence_block_for_prompt(selected)
    api_followup_mode = _is_api_followup_mode(context)
    intent_payload = {
        "ask_type": intent.ask_type,
        "entities": intent.entities,
        "primary_keys": intent.primary_keys,
        "must_include": intent.must_include,
        "confidence": intent.confidence,
        "notes": intent.notes,
    }

    system_prompt = (
        "You are an IAM forensic follow-up assistant. "
        "Provide a direct conversational answer to the user's follow-up. "
        "For short or broad prompts (for example: other issues, anything else, summarize), "
        "infer likely intent from current query + chat history + original report context. "
        "Use only the provided evidence; do not invent facts. "
        "When evidence is insufficient, explicitly say what is missing."
    )

    if api_followup_mode:
        system_prompt += (
            " In this conversation, cite only real [REF_...] IDs from provided evidence. "
            "Never invent citation tags such as [METADATA], [RAW_LOG], or [VECTOR_STORE]."
        )

    citation_instruction = "Respond in concise conversational markdown and cite evidence IDs inline like [M2], [F1]."
    if api_followup_mode:
        citation_instruction = (
            "Respond in concise conversational markdown and cite only actual [REF_...] IDs "
            "from the provided evidence."
        )

    ticket_block = ""
    if getattr(context, "ticket_text", "") and not api_followup_mode:
        t = context.ticket_text[:1200]
        ticket_block = f"Support ticket context (excerpt — this analysis was guided by the attached ticket):\n{t}{'...' if len(context.ticket_text) > 1200 else ''}\n\n"

    human_prompt = (
        f"Original analysis query: {context.query_text}\n"
        f"{ticket_block}"
        f"Recent chat turns:\n{history_text}\n\n"
        f"Follow-up query: {query}\n"
        f"Parsed intent JSON: {json.dumps(intent_payload)}\n\n"
        "Available evidence:\n"
        f"{evidence_block}\n\n"
        f"{citation_instruction}"
    )

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        answer = str(getattr(response, "content", "")).strip()
    except Exception as error:
        answer = (
            "I couldn't generate a grounded answer right now due to an LLM response error. "
            f"Please try rephrasing your question. Details: {error}"
        )

    return f"{answer}\n\n{_build_evidence_table(selected)}"


def answer_analysis_results_query(
    context: AnalysisContext,
    query: str,
    top_k: int = DEFAULT_TOP_K_RESULTS,
    chat_history: list[dict[str, str]] | None = None,
    vector_store: Any = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """
    Answer follow-up query via LLM intent understanding and multi-source retrieval.

    Args:
        context: Analysis context
        query: Follow-up query text
        top_k: Number of evidence rows to surface
        chat_history: Prior chat turns for conversational context
        vector_store: Persistent vector store adapter

    Returns:
        Conversational markdown answer with compact evidence table
    """
    if not query.strip():
        return "Please provide a follow-up question."

    if not context.entries:
        return (
            "No analysis artifacts are available for follow-up retrieval yet. "
            "Run a full analysis first."
        )

    from followup.server_sql import answer_server_monitoring_followup, is_server_monitoring_followup_mode

    if is_server_monitoring_followup_mode(context):
        duckdb_conns = None
        try:
            import streamlit as st

            duckdb_conns = getattr(st.session_state, "server_monitoring_conns", None)
        except Exception:
            duckdb_conns = None
        return answer_server_monitoring_followup(
            context=context,
            query=query,
            chat_history=chat_history,
            duckdb_conns=duckdb_conns,
            progress_callback=progress_callback,
        )

    from followup.intent import _parse_intent

    intent, rephrase_message = _parse_intent(context, query, chat_history)
    if intent is None:
        return rephrase_message or "I couldn't parse your request. Please rephrase."

    terms = _intent_terms(intent, query)

    api_followup_mode = _is_api_followup_mode(context)

    if api_followup_mode:
        candidates = _debug_ref_candidates(context, terms)
    else:
        metadata_items = _metadata_candidates(context)
        faiss_items = _faiss_semantic_candidates(context, query)
        debug_items = _debug_evidence_candidates(context, terms)
        raw_log_items = _raw_log_candidates(context, terms)
        vector_items = _vector_store_candidates(query, vector_store)
        candidates = [
            *faiss_items,
            *metadata_items,
            *raw_log_items,
            *debug_items,
            *vector_items,
        ]

    selected = _rank_and_select_evidence(intent, candidates, top_k)

    if not selected:
        return (
            "I couldn't find enough relevant persisted evidence for that follow-up. "
            "Please rephrase with a clearer target (error pattern, timeline range, or specific thread/key)."
        )

    return _generate_conversational_answer(
        context=context,
        query=query,
        intent=intent,
        selected=selected,
        chat_history=chat_history,
    )
