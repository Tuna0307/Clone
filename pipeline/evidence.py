"""Evidence selection for the Map phase.

Extracted from iam_log_intelligence_agent_hybridChunking2.py as part of a
conservative modular refactor.
"""

from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from pipeline.constants import (
    BENIGN_CHUNK_MAX_CHARS,
    ERROR_CHUNK_MAX_CHARS,
    MAP_EVIDENCE_BUDGET_CHARS,
    MAP_MAX_CHUNKS,
    MAP_NEIGHBOUR_RADIUS,
    MAP_TOP_N_CHUNKS,
    _DEDUP_WS_RE,
)
from pipeline.chunking import decode_url_encoded_errors
from pipeline.references import _line_reference_from_metadata, _source_reference_from_doc


def select_evidence_chunks(
    scored_docs: list[Document],
    top_n: int = MAP_TOP_N_CHUNKS,
    neighbour_radius: int = MAP_NEIGHBOUR_RADIUS,
    max_total_chars: int = MAP_EVIDENCE_BUDGET_CHARS,
    category: str = 'server_monitoring',
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """
    Select evidence chunks using a unified strategy:
      1. Top-N anomaly-ranked chunks from the signal-filtered candidate set
      2. Temporal/thread neighbours for context

    URL-encoded error messages are decoded for LLM readability.
    Error-bearing chunks are NOT truncated to preserve diagnostic detail.
    A hard total character budget prevents context window overflow.

    Args:
        scored_docs:       Documents sorted by anomaly_score descending
        top_n:             Number of highest-scoring chunks to select
        neighbour_radius:  Number of chunks before/after to include for context
        max_total_chars:   Hard cap on total evidence characters

    Returns:
        Tuple of (formatted evidence text, selected row IDs, source reference rows)
    """
    n = len(scored_docs)
    if n == 0:
        return "", [], []

    # Build a lookup from (source_file, primary_key) -> list of indices in
    # scored_docs so we can find neighbours within the same group
    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(scored_docs):
        group_key = (
            doc.metadata.get('source_file', ''),
            doc.metadata.get('primary_key', ''),
        )
        key_to_indices[str(group_key)].append(idx)

    def _seed_sort_key(idx: int) -> tuple:
        doc = scored_docs[idx]
        is_signal = doc.metadata.get('signal_candidate', False)
        is_critical = doc.metadata.get('iam_critical', False)
        has_error = doc.metadata.get('has_error_signal', False)
        score = doc.metadata.get('anomaly_score', 0.0)
        return (not is_signal, not is_critical, not has_error, -score)

    # Select top-N ranked seeds, explicitly preferring signal-bearing chunks.
    selected_indices: set[int] = set()
    top_indices = sorted(range(n), key=_seed_sort_key)[:min(top_n, n)]
    top_index_set = set(top_indices)
    selected_indices.update(top_indices)

    # Add neighbours within the same thread/session group
    seed_indices = list(selected_indices)
    for idx in seed_indices:
        if idx >= n:
            continue
        doc = scored_docs[idx]
        group_key = str((
            doc.metadata.get('source_file', ''),
            doc.metadata.get('primary_key', ''),
        ))
        group_indices = key_to_indices[group_key]
        pos = group_indices.index(idx) if idx in group_indices else -1
        if pos >= 0:
            for delta in range(-neighbour_radius, neighbour_radius + 1):
                ni = pos + delta
                if 0 <= ni < len(group_indices):
                    selected_indices.add(group_indices[ni])

    # Cap total to prevent prompt overflow.
    # Keep ranked seeds ahead of neighbours, then prefer signal-bearing chunks over
    # fallback anomalies so noisy startup/log4j chunks do not consume the budget.
    def _evidence_sort_key(idx: int) -> tuple:
        doc = scored_docs[idx]
        is_ranked_seed = idx in top_index_set
        is_signal = doc.metadata.get('signal_candidate', False)
        is_critical = doc.metadata.get('iam_critical', False)
        has_error = doc.metadata.get('has_error_signal', False)
        score = doc.metadata.get('anomaly_score', 0.0)
        return (not is_ranked_seed, not is_signal, not is_critical, not has_error, -score)

    selected_list = sorted(selected_indices, key=_evidence_sort_key)[:MAP_MAX_CHUNKS]

    # Format evidence text with total budget enforcement
    parts: list[str] = []
    selected_row_ids: list[str] = []
    source_reference_map: list[dict[str, Any]] = []
    total_chars = 0
    seen_error_line_signatures: set[str] = set()
    duplicate_error_chunks_skipped = 0

    def _first_error_line_signature(text: str) -> str:
        for line in text.splitlines():
            if any(
                kw in line for kw in
                ['Exception', 'Error', 'Failed', 'FATAL', 'CRITICAL',
                 'SecurityException', 'SessionInvalid', 'Caused by:']
            ):
                return _DEDUP_WS_RE.sub(' ', line).strip().lower()
        return ''

    for rank, idx in enumerate(selected_list):
        doc = scored_docs[idx]
        score = doc.metadata.get('anomaly_score', 0.0)
        pk = doc.metadata.get('primary_key', 'unknown')
        key_type = str(doc.metadata.get('key_type', ''))
        src = doc.metadata.get('source_file', 'unknown')
        start = doc.metadata.get('start_time', '')
        end = doc.metadata.get('end_time', '')
        ref_id = f"REF_{src}_{pk}_{rank}"
        original_line_ref = _line_reference_from_metadata(doc.metadata)
        original_ref_line = (
            f"Chunk Reference: [{ref_id}]\n"
            f"Original Log Reference: {src}, {original_line_ref}"
        )

        is_ranked_seed = idx in top_index_set
        if is_ranked_seed:
            source_tag = 'ranked:signal' if doc.metadata.get('signal_candidate', False) else 'ranked:fallback'
        else:
            source_tag = 'neighbour'

        is_api_header = category == 'api_request' and key_type in {'api_request', 'api_signal_event'}
        if is_api_header:
            request_key = 'error_line' if key_type == 'api_signal_event' else pk
            header = (
                f"--- Request [score={score:.2f}] [{ref_id}] "
                f"[source={source_tag}] "
                f"[file={src}] [request_key={request_key}] [type=api_request] "
                f"[{original_line_ref}] [{start} -> {end}] ---\n"
                f"{original_ref_line}"
            )
        else:
            header = (
                f"--- Chunk [score={score:.2f}] [{ref_id}] "
                f"[source={source_tag}] "
                f"[file={src}] [key={pk}] [{original_line_ref}] [{start} -> {end}] ---\n"
                f"{original_ref_line}"
            )

        content = doc.page_content

        # Smart truncation: do NOT truncate error-bearing chunks
        has_error_content = any(
            kw in content for kw in
            ['Exception', 'Error', 'Failed', 'FATAL', 'CRITICAL',
             'SecurityException', 'SessionInvalid', 'Caused by:']
        )
        if has_error_content:
            # Preserve full content for error chunks
            if len(content) > ERROR_CHUNK_MAX_CHARS:
                content = content[:ERROR_CHUNK_MAX_CHARS] + "\n... [truncated] ..."

            signature = _first_error_line_signature(content)
            if signature:
                if signature in seen_error_line_signatures:
                    duplicate_error_chunks_skipped += 1
                    continue
                seen_error_line_signatures.add(signature)
        else:
            # Truncate benign chunks to save token budget
            if len(content) > BENIGN_CHUNK_MAX_CHARS:
                content = content[:BENIGN_CHUNK_MAX_CHARS] + "\n... [truncated] ..."

        # Decode URL-encoded error messages for LLM readability
        content = decode_url_encoded_errors(content)

        chunk_text = f"{header}\n{content}"

        # Enforce total evidence budget
        if total_chars + len(chunk_text) > max_total_chars:
            print(f"  [Evidence] Budget cap reached at {total_chars:,} chars "
                  f"({len(parts)} chunks). Remaining chunks skipped.")
            break
        total_chars += len(chunk_text)
        parts.append(chunk_text)
        source_reference_map.append(_source_reference_from_doc(doc, ref_id))
        row_id = str(doc.metadata.get('row_id', '')).strip()
        if row_id:
            selected_row_ids.append(row_id)

    evidence_text = "\n\n".join(parts)

    anomaly_count = sum(1 for i in selected_list if i in top_indices)
    neighbour_count = len(selected_list) - anomaly_count
    print(f"  [Evidence] Selected {len(selected_list):,} chunks "
          f"({anomaly_count} ranked, {neighbour_count} neighbours)")
    if duplicate_error_chunks_skipped:
        print(f"  [Evidence] Skipped {duplicate_error_chunks_skipped:,} duplicate error chunks")
    return evidence_text, selected_row_ids, source_reference_map
