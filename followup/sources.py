"""Source-candidate extraction helpers for artifact-first follow-up retrieval.

This module holds the 12 evidence-source functions extracted from
`followup_retrieval.py` so that the retrieval orchestration layer stays thin:

- parse_debug_evidence_file
- _is_api_followup_mode
- _extract_ref_ids
- _debug_ref_candidates
- build_analysis_results_debug_markdown
- _metadata_candidates
- _faiss_semantic_candidates
- _split_candidate_snippets
- _debug_evidence_candidates
- _extract_raw_log_windows
- _raw_log_candidates
- _vector_store_candidates
"""

from __future__ import annotations

import os
import re
from collections import deque
from typing import Any

import numpy as np

from followup.context import AnalysisContext, EvidenceItem, _as_float, _load_metadata_rows
from followup.intent import _get_followup_embeddings

try:
    import faiss
except Exception:  # pragma: no cover - optional runtime dependency
    faiss = None  # type: ignore[assignment]

# Constants only used by source functions
FOLLOWUP_EVIDENCE_PREVIEW_CHARS = 320
FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE = 3
FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET = 7
FOLLOWUP_DEBUG_MAX_CHARS = 6000
FOLLOWUP_FAISS_TOP_K = 10
FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS = 900

_REF_ID_RE = re.compile(r"\[(REF_[^\]]+)\]")


def _preview_text(text: str, max_chars: int = FOLLOWUP_EVIDENCE_PREVIEW_CHARS) -> str:
    """
    Build one-line preview.

    Args:
        text: Input text
        max_chars: Preview cap

    Returns:
        Preview text
    """
    return text.replace("\n", " ")[:max_chars]


def parse_debug_evidence_file(file_path: str) -> dict[str, str]:
    """
    Parse debug evidence text into sections.

    Args:
        file_path: Debug evidence path

    Returns:
        Dict with system_prompt, user_prompt, and raw text
    """
    if not os.path.exists(file_path):
        return {"error": f"Debug evidence file not found: {file_path}"}

    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()
    except Exception as error:
        return {"error": f"Failed to read debug evidence: {error}"}

    system_marker = "=== SYSTEM PROMPT ==="
    user_marker = "=== USER PROMPT ==="

    system_prompt = ""
    user_prompt = text

    if system_marker in text and user_marker in text:
        try:
            _, after_system = text.split(system_marker, 1)
            system_prompt, user_prompt = after_system.split(user_marker, 1)
            system_prompt = system_prompt.strip()
            user_prompt = user_prompt.strip()
        except Exception:
            user_prompt = text

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw": text,
    }


def _is_api_followup_mode(context: AnalysisContext) -> bool:
    """
    Determine whether follow-up should enforce API REF-only citation behavior.

    Args:
        context: Active analysis context

    Returns:
        True when any entry belongs to API request category
    """
    return any(entry.category == "api_request" for entry in context.entries)


def _extract_ref_ids(text: str) -> list[str]:
    """
    Extract REF IDs from evidence text.

    Args:
        text: Input text block

    Returns:
        Ordered list of unique REF IDs
    """
    seen: set[str] = set()
    ids: list[str] = []
    for match in _REF_ID_RE.findall(text):
        ref_id = match.strip()
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        ids.append(ref_id)
    return ids


def _debug_ref_candidates(context: AnalysisContext, terms: list[str]) -> list[EvidenceItem]:
    """
    Build REF-grounded evidence candidates from debug evidence prompts.

    Args:
        context: Active context
        terms: Query/intent terms

    Returns:
        Evidence items with evidence_id set to real REF IDs
    """
    items: list[EvidenceItem] = []
    lowered_terms = [term.lower() for term in terms if term.strip()]

    for entry in context.entries:
        if entry.category != "api_request":
            continue

        parsed = parse_debug_evidence_file(entry.debug_evidence_file)
        if "error" in parsed:
            continue

        user_prompt = parsed.get("user_prompt", "")
        if not user_prompt.strip():
            continue

        ref_ids = _extract_ref_ids(user_prompt)
        if not ref_ids:
            continue

        for ref_id in ref_ids:
            marker = f"[{ref_id}]"
            pos = user_prompt.find(marker)
            if pos < 0:
                continue

            snippet = user_prompt[pos:pos + FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS]
            lowered_snippet = snippet.lower()
            score = 0.45
            if lowered_terms:
                for term in lowered_terms:
                    score += lowered_snippet.count(term) * 0.2
            else:
                score += 0.2

            items.append(
                EvidenceItem(
                    evidence_id=ref_id,
                    source="api_ref",
                    file_name=entry.file_name,
                    relevance=min(max(score, 0.0), 1.0),
                    anomaly_score=0.0,
                    excerpt=_preview_text(snippet),
                    raw_text=snippet,
                )
            )

    return items


def build_analysis_results_debug_markdown(context: AnalysisContext, max_chars: int = 7000) -> str:
    """
    Build markdown summary for debug evidence files.

    Args:
        context: Analysis context
        max_chars: Truncation cap for rendering safety

    Returns:
        Markdown string
    """
    if not context.entries:
        return "No analysis context available for debug evidence."

    lines: list[str] = ["### Debug Evidence"]

    for entry in context.entries:
        parsed = parse_debug_evidence_file(entry.debug_evidence_file)
        if "error" in parsed:
            lines.append(f"- {entry.file_name}: {parsed['error']}")
            continue

        system_prompt = parsed.get("system_prompt", "")
        user_prompt = parsed.get("user_prompt", "")

        if len(system_prompt) > max_chars:
            system_prompt = system_prompt[:max_chars] + "\n...[truncated]"
        if len(user_prompt) > max_chars:
            user_prompt = user_prompt[:max_chars] + "\n...[truncated]"

        lines.append("")
        lines.append(f"#### {entry.file_name}")
        lines.append(f"- Debug file: {entry.debug_evidence_file}")
        lines.append("- System prompt excerpt:")
        lines.append("```")
        lines.append(system_prompt or "(empty)")
        lines.append("```")
        lines.append("- User prompt + evidence excerpt:")
        lines.append("```")
        lines.append(user_prompt or "(empty)")
        lines.append("```")

    return "\n".join(lines)


def _metadata_candidates(context: AnalysisContext) -> list[EvidenceItem]:
    """
    Convert metadata rows into evidence candidates.

    Args:
        context: Active context

    Returns:
        Metadata evidence items
    """
    items: list[EvidenceItem] = []
    counter = 1

    for entry in context.entries:
        rows = _load_metadata_rows(entry)
        for row in rows:
            content = str(row.get("content", "")).strip()
            if not content:
                continue

            anomaly = _as_float(row.get("anomaly_score", 0.0))
            iam_critical = bool(row.get("iam_critical", False))
            relevance = 0.45 + min(max(anomaly, 0.0), 8.0) / 12.0
            if iam_critical:
                relevance += 0.2

            key = str(row.get("primary_key", "")).strip()
            start = str(row.get("start_time", "")).strip()
            end = str(row.get("end_time", "")).strip()

            item_text = (
                f"file={entry.file_name} primary_key={key or 'n/a'} "
                f"anomaly_score={anomaly:.3f} time={start or 'n/a'} -> {end or 'n/a'}\n"
                f"{content}"
            )
            items.append(
                EvidenceItem(
                    evidence_id=f"M{counter}",
                    source="metadata",
                    file_name=entry.file_name,
                    relevance=min(relevance, 1.0),
                    anomaly_score=anomaly,
                    excerpt=_preview_text(item_text),
                    raw_text=item_text,
                )
            )
            counter += 1

    return items


def _faiss_semantic_candidates(context: AnalysisContext, query: str) -> list[EvidenceItem]:
    """
    Query persisted FAISS indexes to retrieve semantically close chunks.

    Args:
        context: Active context
        query: User follow-up query

    Returns:
        FAISS evidence items
    """
    if faiss is None or np is None:
        return []

    embeddings = _get_followup_embeddings()
    if embeddings is None:
        return []

    try:
        query_vector = embeddings.embed_query(query)
    except Exception:
        return []

    if not query_vector:
        return []

    query_matrix = np.array([query_vector], dtype="float32")
    items: list[EvidenceItem] = []
    counter = 1

    for entry in context.entries:
        index_path = os.path.join(entry.faiss_index_dir, "index.faiss")
        if not os.path.exists(index_path):
            continue

        rows = _load_metadata_rows(entry)
        if not rows:
            continue

        try:
            index = faiss.read_index(index_path)
            k_value = min(FOLLOWUP_FAISS_TOP_K, len(rows))
            distances, indices = index.search(query_matrix, k_value)
        except Exception:
            continue

        if len(indices) == 0:
            continue

        for rank, row_index in enumerate(indices[0]):
            if row_index < 0 or row_index >= len(rows):
                continue
            row = rows[int(row_index)]
            distance = float(distances[0][rank]) if len(distances) else 999.0
            anomaly = _as_float(row.get("anomaly_score", 0.0))
            relevance = 1.0 / (1.0 + max(distance, 0.0))
            content = str(row.get("content", "")).strip()
            key = str(row.get("primary_key", "")).strip()
            if not content:
                continue

            item_text = (
                f"file={entry.file_name} distance={distance:.4f} "
                f"primary_key={key or 'n/a'} anomaly_score={anomaly:.3f}\n"
                f"{content}"
            )
            items.append(
                EvidenceItem(
                    evidence_id=f"F{counter}",
                    source="faiss",
                    file_name=entry.file_name,
                    relevance=min(max(relevance, 0.0), 1.0),
                    anomaly_score=anomaly,
                    excerpt=_preview_text(item_text),
                    raw_text=item_text,
                )
            )
            counter += 1

    return items


def _split_candidate_snippets(text: str, terms: list[str], max_snippets: int = 2) -> list[tuple[float, str]]:
    """
    Build relevance-scored snippets from a large text blob.

    Args:
        text: Source text
        terms: Query terms
        max_snippets: Maximum snippets to return

    Returns:
        List of tuples (score, snippet)
    """
    if not text.strip():
        return []

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    scored: list[tuple[float, str]] = []
    lowered_terms = [term.lower() for term in terms if term.strip()]

    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        score = 0.2
        for term in lowered_terms:
            score += paragraph_lower.count(term) * 0.25
        scored.append((score, paragraph[:FOLLOWUP_DEBUG_MAX_CHARS]))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:max_snippets]


def _debug_evidence_candidates(context: AnalysisContext, terms: list[str]) -> list[EvidenceItem]:
    """
    Retrieve relevant snippets from debug evidence files.

    Args:
        context: Active context
        terms: Query/intent terms

    Returns:
        Debug evidence items
    """
    items: list[EvidenceItem] = []
    counter = 1

    for entry in context.entries:
        parsed = parse_debug_evidence_file(entry.debug_evidence_file)
        if "error" in parsed:
            continue

        merged_text = (
            f"System Prompt:\n{parsed.get('system_prompt', '')[:FOLLOWUP_DEBUG_MAX_CHARS]}\n\n"
            f"User Prompt and Evidence:\n{parsed.get('user_prompt', '')[:FOLLOWUP_DEBUG_MAX_CHARS]}"
        )
        for relevance, snippet in _split_candidate_snippets(merged_text, terms, max_snippets=2):
            item_text = f"[DEBUG] file={entry.file_name}\n{snippet}"
            items.append(
                EvidenceItem(
                    evidence_id=f"D{counter}",
                    source="debug",
                    file_name=entry.file_name,
                    relevance=min(max(relevance, 0.0), 1.0),
                    anomaly_score=0.0,
                    excerpt=_preview_text(item_text),
                    raw_text=item_text,
                )
            )
            counter += 1

    return items


def _extract_raw_log_windows(lines: list[str], hit_indexes: list[int]) -> list[str]:
    """
    Build contextual windows around matching line indexes.

    Args:
        lines: Full file lines
        hit_indexes: Matched line indexes

    Returns:
        Context windows
    """
    windows: list[str] = []
    for line_index in hit_indexes[:FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE]:
        start = max(0, line_index - 2)
        end = min(len(lines), line_index + 3)
        snippet_lines = []
        for cursor in range(start, end):
            snippet_lines.append(f"L{cursor + 1}: {lines[cursor].rstrip()}")
        windows.append("\n".join(snippet_lines[:FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET]))
    return windows


def _format_numbered_window(numbered_lines: list[tuple[int, str]]) -> str:
    return "\n".join(
        f"L{line_number}: {line.rstrip()}"
        for line_number, line in numbered_lines[:FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET]
    )


def _stream_raw_log_windows(file_handle: Any, lowered_terms: list[str]) -> list[str]:
    """
    Build raw-log snippets without loading the full source file into memory.

    Args:
        file_handle: Iterable text file handle
        lowered_terms: Lowercase search terms

    Returns:
        Context windows around hits, or bounded first/last fallback windows
    """
    windows: list[str] = []
    before: deque[tuple[int, str]] = deque(maxlen=2)
    pending: list[dict[str, Any]] = []
    first_lines: list[tuple[int, str]] = []
    last_lines: deque[tuple[int, str]] = deque(maxlen=FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET)

    for line_number, line in enumerate(file_handle, start=1):
        numbered_line = (line_number, line)
        if len(first_lines) < FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET:
            first_lines.append(numbered_line)
        last_lines.append(numbered_line)

        next_pending: list[dict[str, Any]] = []
        for item in pending:
            item["lines"].append(numbered_line)
            item["remaining"] -= 1
            if item["remaining"] <= 0:
                windows.append(_format_numbered_window(item["lines"]))
            else:
                next_pending.append(item)
        pending = next_pending

        lowered_line = line.lower()
        has_hit = bool(lowered_terms) and any(term in lowered_line for term in lowered_terms)
        if has_hit and len(windows) + len(pending) < FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            pending.append(
                {
                    "remaining": 2,
                    "lines": [*before, numbered_line],
                }
            )

        before.append(numbered_line)

        if len(windows) >= FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            break

    for item in pending:
        if len(windows) >= FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            break
        windows.append(_format_numbered_window(item["lines"]))

    if windows:
        return windows[:FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE]

    fallback_windows: list[str] = []
    if first_lines:
        fallback_windows.append(_format_numbered_window(first_lines))
    last_window = list(last_lines)
    if last_window and last_window != first_lines:
        fallback_windows.append(_format_numbered_window(last_window))
    return fallback_windows[:FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE]


def _raw_log_candidates(context: AnalysisContext, terms: list[str]) -> list[EvidenceItem]:
    """
    Retrieve relevant snippets from source logs (remaining raw content).

    Args:
        context: Active context
        terms: Query/intent terms

    Returns:
        Raw-log evidence items
    """
    items: list[EvidenceItem] = []
    counter = 1
    lowered_terms = [term.lower() for term in terms if term.strip()]

    for entry in context.entries:
        if not entry.source_path or not os.path.exists(entry.source_path):
            continue

        try:
            with open(entry.source_path, "r", encoding="utf-8", errors="replace") as file_handle:
                windows = _stream_raw_log_windows(file_handle, lowered_terms)
        except Exception:
            continue

        for window in windows:
            score = 0.3
            lowered_window = window.lower()
            for term in lowered_terms:
                score += lowered_window.count(term) * 0.2

            item_text = f"file={entry.file_name}\n{window}"
            items.append(
                EvidenceItem(
                    evidence_id=f"R{counter}",
                    source="raw_log",
                    file_name=entry.file_name,
                    relevance=min(max(score, 0.0), 1.0),
                    anomaly_score=0.0,
                    excerpt=_preview_text(item_text),
                    raw_text=item_text,
                )
            )
            counter += 1

    return items


def _vector_store_candidates(
    query: str,
    vector_store: Any,
) -> list[EvidenceItem]:
    """
    Retrieve semantically similar historical context from persistent vector store.

    Args:
        query: User follow-up query
        vector_store: ChatVectorStore instance

    Returns:
        Vector store evidence items
    """
    if vector_store is None:
        return []

    try:
        docs = vector_store.retrieve_context(query, k=3)
    except Exception:
        return []

    items: list[EvidenceItem] = []
    for index, doc in enumerate(docs, start=1):
        content = str(getattr(doc, "page_content", "")).strip()
        metadata = getattr(doc, "metadata", {}) or {}
        file_name = str(metadata.get("log_path", metadata.get("analysis_id", "historical_context")))
        if not content:
            continue

        item_text = content
        items.append(
            EvidenceItem(
                evidence_id=f"V{index}",
                source="vector_store",
                file_name=file_name,
                relevance=0.55,
                anomaly_score=0.0,
                excerpt=_preview_text(item_text),
                raw_text=item_text[:2000],
            )
        )

    return items
