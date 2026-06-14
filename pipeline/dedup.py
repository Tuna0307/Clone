"""Deduplication, downselection, and signal filtering for chunk lists.

Extracted from iam_log_intelligence_agent_hybridChunking2.py as part of
a conservative modular refactor.
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from langchain_core.documents import Document

from pipeline.progress import emit_ui_progress
from pipeline.constants import (
    DEDUP_NUMERIC_TOKEN_MIN_LEN,
    LARGE_LOG_CHUNK_TRIGGER,
    MAX_EMBEDDING_CHUNKS_VERY_LARGE,
    VERY_LARGE_LOG_CHUNK_TRIGGER,
    _DEFAULT_API_REQUEST_BOUNDARIES,
    _DEFAULT_ERROR_KEYWORDS,
    _DEFAULT_IAM_CRITICAL_KEYWORDS,
    _DEFAULT_NOISE_PATTERNS,
    _DEDUP_HEX_ADDR_RE,
    _DEDUP_LONG_HEX_RE,
    _DEDUP_UUID_RE,
    _DEDUP_WS_RE,
    _STACK_TRACE_LINE_RE,
    SIGNAL_FILTER_MIN_CANDIDATES,
)
from pipeline.files import stream_file_lines
from pipeline.parsing import _parse_line
from pipeline.query import _line_overlaps_query_window, load_retrieval_signals
from pipeline.text_utils import (
    _contains_any_marker,
    _extract_diagnostic_entities_from_line,
    _is_error_bearing,
    _is_iam_critical_text,
    _is_noisy_text,
)


def build_metadata_rows_from_docs(docs: list[Document]) -> list[dict[str, Any]]:
    """
    Build JSON-safe metadata rows for UI/follow-up consumption.

    Args:
        docs: Chunk documents from current map-stage run

    Returns:
        List of metadata row dictionaries
    """
    def _to_json_safe(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        row = {
            'content': doc.page_content,
            **{k: _to_json_safe(v) for k, v in doc.metadata.items()},
        }
        source_file = str(row.get('source_file', 'unknown'))
        row.setdefault('row_id', f"{source_file}::{idx}")

        key_type = str(row.get('key_type', ''))
        if key_type in {'api_request', 'api_signal_event'}:
            request_key_value = row.pop('primary_key', '')
            row['request_key'] = 'error_line' if key_type == 'api_signal_event' else request_key_value

            if 'chunk_level' in row:
                row['request_level'] = row.pop('chunk_level')

        rows.append(row)
    return rows


def extract_global_evidence_profile(
    file_path: str,
    schema: dict,
    retrieval_signals: dict,
    query_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a deterministic file-wide evidence profile without using the LLM.

    The profile gives the map prompt aggregate context that selected evidence
    chunks may not show on their own: total lines in scope, observed time range,
    error-line count, exception distribution, diagnostic entities, and request
    lifecycle health.

    Args:
        file_path: Source log path
        schema: Detected log schema
        retrieval_signals: Loaded retrieval keywords and request boundaries
        query_context: Optional incident time-window filter

    Returns:
        JSON-safe evidence profile dictionary
    """
    iam_keywords: list[str] = retrieval_signals.get('iam_critical_keywords', [])
    error_keywords: list[str] = retrieval_signals.get('error_keywords', [])
    boundaries: dict[str, list[str]] = retrieval_signals.get(
        'api_request_boundaries',
        dict(_DEFAULT_API_REQUEST_BOUNDARIES),
    )
    start_markers: list[str] = boundaries.get('start_markers', [])
    end_markers: list[str] = boundaries.get('end_markers', [])

    exception_class_re = re.compile(r'\b([A-Za-z_][\w.$]*(?:Exception|Error))\b')
    property_kv_re = re.compile(r'(?:^|[\s&])([a-zA-Z_][\w.$-]*)\s*=\s*([^\s,;&]+)')
    file_path_re = re.compile(
        r'(?:file|path|config|property|resource)[\s:=]+'
        r'([/\\][^\s,;]+|(?:[A-Za-z]:\\[^\s,;]+)|(?:[\w.-]+\.properties))',
        re.IGNORECASE,
    )
    error_code_re = re.compile(r'\b([A-Z]{3,6}[-_]\d{4,}|\bcode=\d{3,}\b|\bextendedCode=\d+\b)')
    http_status_re = re.compile(r'\bHTTP\s+(\d{3})\b', re.IGNORECASE)
    base64_like_re = re.compile(r'\b(?:[A-Za-z0-9+/]{24,}={0,2})\b')

    total_lines = 0
    error_line_count = 0
    min_ts: Optional[datetime] = None
    max_ts: Optional[datetime] = None

    critical_signals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {'count': 0, 'first_seen': '', 'last_seen': '', 'threads': set()},
    )
    seen_entities: set[str] = set()
    diagnostic_entities: list[dict[str, str]] = []
    entity_max = 200

    active_entries: dict[str, int] = defaultdict(int)
    matched_requests = 0
    unmatched_exits = 0

    print("  [Profiler] Scanning file for global evidence profile...")
    emit_ui_progress("Scanning file for global evidence profile...")

    for raw_line in stream_file_lines(file_path):
        ts, pk, clean = _parse_line(raw_line, schema)
        if not _line_overlaps_query_window(ts, query_context):
            continue

        total_lines += 1
        if ts is not None:
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts

        is_error = _is_error_bearing(clean, error_keywords)
        is_iam_critical = _is_iam_critical_text(clean, iam_keywords)
        if is_error:
            error_line_count += 1

        if is_error or is_iam_critical:
            for exc_match in exception_class_re.finditer(clean):
                exc_name = exc_match.group(1)
                sig = critical_signals[exc_name]
                sig['count'] += 1
                if ts is not None:
                    ts_iso = ts.isoformat()
                    if not sig['first_seen'] or ts_iso < sig['first_seen']:
                        sig['first_seen'] = ts_iso
                    if not sig['last_seen'] or ts_iso > sig['last_seen']:
                        sig['last_seen'] = ts_iso
                if pk and len(sig['threads']) < 10:
                    sig['threads'].add(pk)

        if len(diagnostic_entities) < entity_max:
            _extract_diagnostic_entities_from_line(
                clean,
                seen_entities,
                diagnostic_entities,
                entity_max,
                property_kv_re,
                file_path_re,
                error_code_re,
                http_status_re,
                base64_like_re,
                schema,
            )

        if start_markers or end_markers:
            is_entry = _contains_any_marker(clean, start_markers) if start_markers else False
            is_exit = _contains_any_marker(clean, end_markers) if end_markers else False
            thread_key = pk if pk else '__no_thread__'
            if is_entry:
                active_entries[thread_key] += 1
            if is_exit:
                if active_entries.get(thread_key, 0) > 0:
                    active_entries[thread_key] -= 1
                    matched_requests += 1
                else:
                    unmatched_exits += 1

    unmatched_entries = sum(active_entries.values())
    critical_signals_serializable: dict[str, dict[str, Any]] = {}
    for exc_name, sig in sorted(
        critical_signals.items(),
        key=lambda item: int(item[1].get('count', 0)),
        reverse=True,
    ):
        critical_signals_serializable[exc_name] = {
            'count': sig['count'],
            'first_seen': sig['first_seen'],
            'last_seen': sig['last_seen'],
            'affected_threads': sorted(sig['threads'])[:10],
        }

    total_requests = matched_requests + unmatched_entries + unmatched_exits
    if total_requests > 0:
        request_health = {
            'matched_requests': matched_requests,
            'unmatched_entries': unmatched_entries,
            'unmatched_exits': unmatched_exits,
            'error_rate': round((unmatched_entries + unmatched_exits) / total_requests, 4),
        }
    else:
        request_health = {
            'matched_requests': 0,
            'unmatched_entries': 0,
            'unmatched_exits': 0,
            'error_rate': 0.0,
        }

    profile = {
        'total_lines': total_lines,
        'time_range': {
            'start': min_ts.isoformat() if min_ts else '',
            'end': max_ts.isoformat() if max_ts else '',
        },
        'error_line_count': error_line_count,
        'critical_signals': critical_signals_serializable,
        'diagnostic_entities_extracted': diagnostic_entities[:entity_max],
        'request_lifecycle_health': request_health,
    }

    print(
        f"  [Profiler] Extracted {len(critical_signals_serializable)} critical signal types, "
        f"{len(diagnostic_entities)} diagnostic entities, {error_line_count} error lines "
        f"from {total_lines:,} total lines"
    )
    return profile


def _canonicalize_for_dedup(text: str, schema: dict) -> str:
    """
    Produce a conservative canonical form for pre-embedding deduplication.

    Notes:
        - This is intentionally conservative (accuracy-first).
        - It removes volatile identifiers while preserving lexical context.
        - It does NOT do semantic collapsing or fuzzy matching.

    Args:
        text: Raw chunk content
        schema: Detected schema from detect_log_structure

    Returns:
        Canonicalized text for exact/canonical hash dedup
    """
    canonical = text

    timestamp_re = schema.get('timestamp_re')
    if timestamp_re is not None and schema.get('timestamp_fmt'):
        canonical = timestamp_re.sub('<TS>', canonical)

    canonical = _DEDUP_UUID_RE.sub('<UUID>', canonical)
    canonical = _DEDUP_HEX_ADDR_RE.sub('<HEXADDR>', canonical)
    canonical = _DEDUP_LONG_HEX_RE.sub('<HEX>', canonical)

    canonical = re.sub(
        rf'\b\d{{{DEDUP_NUMERIC_TOKEN_MIN_LEN},}}\b',
        '<NUM>',
        canonical,
    )
    canonical = _DEDUP_WS_RE.sub(' ', canonical).strip()
    return canonical


def deduplicate_chunks_safe(docs: list[Document], schema: dict) -> list[Document]:
    """
    Perform conservative pre-embedding deduplication.

    Strategy:
        - Exact/canonical dedup only (no fuzzy matching).
        - Error-bearing chunks are deduped only with other error-bearing chunks.
        - Representative keeps original text/metadata; occurrence stats are merged.

    Args:
        docs: Chunk list from Stage 2
        schema: Detected schema for timestamp-aware canonicalization

    Returns:
        Deduplicated chunk list
    """
    if not docs:
        return docs

    print(f"  [Dedup] Conservative dedup on {len(docs):,} chunks...")

    unique_docs: list[Document] = []
    key_to_index: dict[str, int] = {}
    merged_count = 0

    for doc in docs:
        content = doc.page_content
        error_flag = _is_error_bearing(content)
        canonical = _canonicalize_for_dedup(content, schema)
        digest = hashlib.sha1(canonical.encode('utf-8', errors='ignore')).hexdigest()
        dedup_key = f"err={int(error_flag)}|h={digest}"

        if dedup_key not in key_to_index:
            rep = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            rep.metadata['dedup_count'] = 1
            rep.metadata['dedup_first_start'] = rep.metadata.get('start_time', '')
            rep.metadata['dedup_last_end'] = rep.metadata.get('end_time', '')
            unique_docs.append(rep)
            key_to_index[dedup_key] = len(unique_docs) - 1
            continue

        merged_count += 1
        rep_idx = key_to_index[dedup_key]
        rep = unique_docs[rep_idx]
        rep.metadata['dedup_count'] = int(rep.metadata.get('dedup_count', 1)) + 1

        current_first = rep.metadata.get('dedup_first_start', '')
        current_last = rep.metadata.get('dedup_last_end', '')
        candidate_start = doc.metadata.get('start_time', '')
        candidate_end = doc.metadata.get('end_time', '')

        if candidate_start and (not current_first or candidate_start < current_first):
            rep.metadata['dedup_first_start'] = candidate_start
        if candidate_end and (not current_last or candidate_end > current_last):
            rep.metadata['dedup_last_end'] = candidate_end

    reduction_pct = ((len(docs) - len(unique_docs)) / len(docs)) * 100.0
    print(f"    {len(docs):,} -> {len(unique_docs):,} unique "
          f"({reduction_pct:.1f}% reduction, {merged_count:,} merged)")
    return unique_docs


def downselect_chunks_for_embedding(docs: list[Document], max_chunks: int) -> list[Document]:
    """
    Downselect chunks only for very large files before embedding.

    Selection policy (accuracy-first):
        1. Keep all IAM-critical and error-bearing chunks first.
        2. Fill remaining capacity using round-robin across key_type buckets
           to preserve thread/time/no-timestamp diversity.

    Args:
        docs: Deduplicated chunk list
        max_chunks: Hard cap for chunks passed to embedding stage

    Returns:
        Selected chunk list (<= max_chunks)
    """
    if len(docs) <= max_chunks:
        return docs

    retrieval_signals = load_retrieval_signals()
    iam_critical_keywords: list[str] = retrieval_signals['iam_critical_keywords']

    selected: list[Document] = []
    selected_ids: set[int] = set()

    def _priority(doc: Document) -> tuple:
        content = doc.page_content
        is_critical = _is_iam_critical_text(content, iam_critical_keywords)
        is_error = _is_error_bearing(content, retrieval_signals['error_keywords'])
        dedup_count = int(doc.metadata.get('dedup_count', 1))
        line_count = int(doc.metadata.get('line_count', 0))
        return (is_critical, is_error, dedup_count, line_count)

    ordered_indices = sorted(
        range(len(docs)),
        key=lambda i: _priority(docs[i]),
        reverse=True,
    )

    for idx in ordered_indices:
        content = docs[idx].page_content
        is_critical = _is_iam_critical_text(content, iam_critical_keywords)
        is_error = _is_error_bearing(content, retrieval_signals['error_keywords'])
        if not (is_critical or is_error):
            continue
        selected.append(docs[idx])
        selected_ids.add(idx)
        if len(selected) >= max_chunks:
            break

    if len(selected) >= max_chunks:
        print(f"  [Downselect] {len(docs):,} -> {len(selected):,} (critical/error priority)")
        return selected[:max_chunks]

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx in ordered_indices:
        if idx in selected_ids:
            continue
        key_type = str(docs[idx].metadata.get('key_type', 'unknown'))
        buckets[key_type].append(idx)

    bucket_keys = sorted(buckets.keys())
    bucket_pos: dict[str, int] = {k: 0 for k in bucket_keys}
    made_progress = True

    while len(selected) < max_chunks and made_progress:
        made_progress = False
        for key in bucket_keys:
            pos = bucket_pos[key]
            if pos >= len(buckets[key]):
                continue
            idx = buckets[key][pos]
            bucket_pos[key] += 1
            selected.append(docs[idx])
            selected_ids.add(idx)
            made_progress = True
            if len(selected) >= max_chunks:
                break

    print(f"  [Downselect] {len(docs):,} -> {len(selected):,} "
          f"(very-large pre-embedding cap={max_chunks:,})")
    return selected


def filter_chunks_by_signal(
    docs: list[Document],
    min_candidates: int = SIGNAL_FILTER_MIN_CANDIDATES,
) -> list[Document]:
    """
    Reduce the anomaly-scoring candidate pool to high-signal chunks first.

     Strategy:
          1. Keep IAM-critical or error-bearing chunks based on diagnostic lines,
              not stack-frame-only content.
          2. Avoid pure structural noise where possible.
          3. Backfill only to a modest floor so fallback noise does not dominate
              the anomaly-ranked seed set.

    Args:
        docs: Chunk list after chunking / pre-embedding compression
        min_candidates: Minimum number of chunks to retain for anomaly scoring

    Returns:
        Candidate chunk list for anomaly scoring
    """
    if not docs:
        return docs

    retrieval_signals = load_retrieval_signals()
    iam_keywords = retrieval_signals['iam_critical_keywords']
    error_keywords = retrieval_signals['error_keywords']
    noise_patterns = retrieval_signals['noise_patterns']

    signal_indices: list[int] = []
    fallback_candidates: list[int] = []
    noisy_fallback_candidates: list[int] = []

    for idx, doc in enumerate(docs):
        content = doc.page_content
        has_iam = _is_iam_critical_text(content, iam_keywords)
        has_error = _is_error_bearing(content, error_keywords)
        is_noisy = _is_noisy_text(content, noise_patterns)
        is_structural_noise = doc.metadata.get('key_type') == 'no_timestamp'

        doc.metadata['iam_critical'] = has_iam
        doc.metadata['has_error_signal'] = has_error
        doc.metadata['signal_candidate'] = False

        if has_iam or has_error:
            if has_iam or not (is_noisy and is_structural_noise):
                signal_indices.append(idx)
                doc.metadata['signal_candidate'] = True
                continue

        if not is_noisy and not is_structural_noise:
            fallback_candidates.append(idx)
        elif not is_structural_noise:
            noisy_fallback_candidates.append(idx)

    signal_count = len(signal_indices)
    target_candidates = min(len(docs), max(signal_count, min_candidates))
    selected_indices: list[int] = list(signal_indices)
    selected_set = set(selected_indices)

    def _fallback_sort_key(idx: int) -> tuple:
        doc = docs[idx]
        key_type = str(doc.metadata.get('key_type', ''))
        key_priority = 2 if key_type == 'thread' else 1 if key_type == 'time_window' else 0
        line_count = int(doc.metadata.get('line_count', 0))
        return (key_priority, line_count)

    if len(selected_indices) < target_candidates:
        for idx in sorted(fallback_candidates, key=_fallback_sort_key, reverse=True):
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            if len(selected_indices) >= target_candidates:
                break

    if len(selected_indices) < target_candidates:
        for idx in sorted(noisy_fallback_candidates, key=_fallback_sort_key, reverse=True):
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            selected_set.add(idx)
            if len(selected_indices) >= target_candidates:
                break

    filtered_docs = [docs[idx] for idx in selected_indices]
    fallback_count = max(0, len(filtered_docs) - signal_count)
    print(
        f"  [Signal] {len(docs):,} -> {len(filtered_docs):,} candidates "
        f"({signal_count} signal, {fallback_count} fallback)"
    )
    return filtered_docs
