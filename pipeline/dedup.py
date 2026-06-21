"""Metadata shaping and file-wide evidence profiling for the API request path."""

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from langchain_core.documents import Document

from pipeline.constants import _DEFAULT_API_REQUEST_BOUNDARIES
from pipeline.files import stream_file_lines
from pipeline.parsing import _parse_line
from pipeline.progress import emit_ui_progress
from pipeline.query import _line_overlaps_query_window
from pipeline.text_utils import (
    _contains_any_marker,
    _extract_diagnostic_entities_from_line,
    _is_error_bearing,
    _is_iam_critical_text,
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