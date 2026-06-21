"""Deterministic API request extraction and URL-encoded error decoding."""

import os
import re
import urllib.parse
from typing import Any, Optional

from langchain_core.documents import Document

from pipeline.constants import (
    ERROR_SCORE_BOOST,
    IAM_CRITICAL_SCORE_BOOST,
    _DEFAULT_API_REQUEST_BOUNDARIES,
    _DEFAULT_ERROR_KEYWORDS,
    _DEFAULT_IAM_CRITICAL_KEYWORDS,
)
from pipeline.files import stream_file_lines
from pipeline.parsing import _extract_session_label, _parse_line
from pipeline.query import _line_overlaps_query_window
from pipeline.references import _compact_line_ranges, _error_line_ranges_from_numbered_lines
from pipeline.text_utils import _contains_any_marker, _is_error_bearing, _is_iam_critical_text, _is_noisy_text


def extract_api_request_docs_deterministic(
    file_path: str,
    schema: dict,
    boundaries: dict[str, list[str]],
    retrieval_signals: dict,
    query_context: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """
    Deterministically extract API-request evidence without embedding/scoring.

    Args:
        file_path: Log file path
        schema: Detected schema
        boundaries: API entry/exit markers
        retrieval_signals: Loaded retrieval config signals
        query_context: Query context dict

    Returns:
        Request/event docs scored with deterministic signal weights
    """
    file_name = os.path.basename(file_path)
    file_abs_path = os.path.abspath(file_path)
    start_markers = boundaries.get('start_markers', _DEFAULT_API_REQUEST_BOUNDARIES['start_markers'])
    end_markers = boundaries.get('end_markers', _DEFAULT_API_REQUEST_BOUNDARIES['end_markers'])
    iam_keywords = retrieval_signals.get('iam_critical_keywords', list(_DEFAULT_IAM_CRITICAL_KEYWORDS))
    error_keywords = retrieval_signals.get('error_keywords', list(_DEFAULT_ERROR_KEYWORDS))
    noise_patterns = retrieval_signals.get('noise_patterns', [])

    active_by_thread: dict[str, dict[str, Any]] = {}
    docs: list[Document] = []
    request_counter = 0
    singleton_counter = 0

    def _deterministic_score(text: str, matched_start: bool, matched_end: bool) -> tuple[float, bool, bool, bool]:
        has_error = _is_error_bearing(text, error_keywords)
        iam_critical = _is_iam_critical_text(text, iam_keywords)
        is_noise = _is_noisy_text(text, noise_patterns)
        score = 0.0
        if iam_critical:
            score += IAM_CRITICAL_SCORE_BOOST
        if has_error:
            score += ERROR_SCORE_BOOST
        if matched_start or matched_end:
            score += 1.0
        if is_noise:
            score = min(score, 1.0)
        signal_candidate = iam_critical or has_error or matched_start or matched_end
        return score, signal_candidate, iam_critical, has_error

    def _flush_request(thread_key: str, unmatched_exit: bool) -> None:
        nonlocal request_counter
        state = active_by_thread.get(thread_key)
        if state is None:
            return

        lines = state.get('lines', [])
        if not lines:
            active_by_thread.pop(thread_key, None)
            return

        line_numbers = [int(n) for n in state.get('line_numbers', []) if int(n) > 0]
        numbered_lines = list(zip(line_numbers, lines))
        timestamps = [ts for ts in state.get('timestamps', []) if ts is not None]
        start_time = min(timestamps) if timestamps else None
        end_time = max(timestamps) if timestamps else None
        content = '\n'.join(lines)
        matched_start = bool(state.get('matched_start', False))
        matched_end = bool(state.get('matched_end', False))
        score, signal_candidate, iam_critical, has_error = _deterministic_score(content, matched_start, matched_end)

        docs.append(Document(
            page_content=content,
            metadata={
                'source_file': file_name,
                'source_path': file_abs_path,
                'primary_key': thread_key,
                'key_type': 'api_request',
                'chunk_level': 'request',
                'request_span_id': f"{thread_key}_req_{request_counter}",
                'unmatched_exit': unmatched_exit,
                'start_time': start_time.isoformat() if start_time else '',
                'end_time': end_time.isoformat() if end_time else '',
                'line_count': len(lines),
                'start_line': min(line_numbers) if line_numbers else None,
                'end_line': max(line_numbers) if line_numbers else None,
                'line_ranges': _compact_line_ranges(line_numbers),
                'error_line_ranges': _error_line_ranges_from_numbered_lines(numbered_lines, error_keywords),
                'session_labels': ';'.join(sorted(state.get('sessions', set()))),
                'session_label_count': len(state.get('sessions', set())),
                'signal_candidate': signal_candidate,
                'iam_critical': iam_critical,
                'has_error_signal': has_error,
                'anomaly_score': score,
                'raw_distance': 0.0,
            },
        ))
        request_counter += 1
        active_by_thread.pop(thread_key, None)

    for line_number, raw_line in enumerate(stream_file_lines(file_path), start=1):
        ts, pk, clean = _parse_line(raw_line, schema)
        if not _line_overlaps_query_window(ts, query_context):
            continue

        thread_key = pk if pk else 'no_thread'
        session_label = _extract_session_label(clean, schema)
        is_start = _contains_any_marker(clean, start_markers)
        is_end = _contains_any_marker(clean, end_markers)

        if is_start:
            if thread_key in active_by_thread:
                _flush_request(thread_key, unmatched_exit=True)
            active_by_thread[thread_key] = {
                'lines': [clean],
                'line_numbers': [line_number],
                'timestamps': [ts] if ts is not None else [],
                'sessions': ({session_label} if session_label else set()),
                'matched_start': True,
                'matched_end': bool(is_end),
            }
            if is_end:
                _flush_request(thread_key, unmatched_exit=False)
            continue

        if thread_key in active_by_thread:
            state = active_by_thread[thread_key]
            state['lines'].append(clean)
            state.setdefault('line_numbers', []).append(line_number)
            if ts is not None:
                state['timestamps'].append(ts)
            if session_label:
                state['sessions'].add(session_label)
            if is_end:
                state['matched_end'] = True
                _flush_request(thread_key, unmatched_exit=False)
            continue

        score, signal_candidate, iam_critical, has_error = _deterministic_score(clean, is_start, is_end)
        if signal_candidate:
            docs.append(Document(
                page_content=clean,
                metadata={
                    'source_file': file_name,
                    'source_path': file_abs_path,
                    'primary_key': thread_key,
                    'key_type': 'api_signal_event',
                    'chunk_level': 'line',
                    'request_span_id': f"event_{singleton_counter}",
                    'unmatched_exit': False,
                    'start_time': ts.isoformat() if ts else '',
                    'end_time': ts.isoformat() if ts else '',
                    'line_count': 1,
                    'start_line': line_number,
                    'end_line': line_number,
                    'line_ranges': str(line_number),
                    'error_line_ranges': str(line_number) if has_error else '',
                    'session_labels': session_label,
                    'session_label_count': 1 if session_label else 0,
                    'signal_candidate': signal_candidate,
                    'iam_critical': iam_critical,
                    'has_error_signal': has_error,
                    'anomaly_score': score,
                    'raw_distance': 0.0,
                },
            ))
            singleton_counter += 1

    for thread_key in list(active_by_thread.keys()):
        _flush_request(thread_key, unmatched_exit=True)

    docs.sort(
        key=lambda d: (
            -float(d.metadata.get('anomaly_score', 0.0)),
            str(d.metadata.get('start_time', '')),
        )
    )
    print(f"  [API-Deterministic] Extracted {len(docs):,} request/event docs")
    return docs


def decode_url_encoded_errors(text: str) -> str:
    """
    Decode URL-encoded segments within error messages to make them
    human-readable for LLM analysis.

    Handles common URL encoding in Java stack traces and IAM log messages
    (e.g., %24 -> $, %3D -> =, %26amp; -> &, + -> space).

    Args:
        text: Raw log text possibly containing URL-encoded segments

    Returns:
        Text with URL-encoded segments decoded
    """
    lines = text.split('\n')
    decoded_lines: list[str] = []
    url_pattern = re.compile(r'%[0-9A-Fa-f]{2}')

    for line in lines:
        if url_pattern.search(line):
            try:
                cleaned = line.replace('%26amp;', '&')
                cleaned = cleaned.replace('%26gt;', '>')
                cleaned = cleaned.replace('%26lt;', '<')
                decoded = urllib.parse.unquote_plus(cleaned)
                decoded_lines.append(decoded)
            except Exception:
                decoded_lines.append(line)
        else:
            decoded_lines.append(line)

    return '\n'.join(decoded_lines)