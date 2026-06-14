"""Log chunking strategies: deterministic API extraction, server-monitoring windows,
hierarchical API request splitting, generic hybrid thread+time grouping, and
URL-encoded error decoding.
"""

import os
import re
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.documents import Document

from pipeline.constants import (
    CHUNK_OVERLAP_CHARS,
    ERROR_SCORE_BOOST,
    IAM_CRITICAL_SCORE_BOOST,
    MAX_GROUP_CHARS,
    NO_TS_CATCH_ALL_CHARS,
    SERVER_MONITOR_WINDOW_SECONDS,
    UNGROUPED_MAX_LINES_PER_CHUNK,
    UNGROUPED_WINDOW_SECONDS,
    _DEFAULT_API_REQUEST_BOUNDARIES,
    _DEFAULT_ERROR_KEYWORDS,
    _DEFAULT_IAM_CRITICAL_KEYWORDS,
)
from pipeline.files import stream_file_lines
from pipeline.parsing import _extract_session_label, _parse_line
from pipeline.query import _line_overlaps_query_window
from pipeline.text_utils import _contains_any_marker, _is_error_bearing, _is_iam_critical_text, _is_noisy_text

from pipeline.references import _compact_line_ranges, _error_line_ranges_from_numbered_lines

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


def chunk_server_monitoring_log(
    file_path: str,
    schema: dict,
    query_context: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """
    Chunk server monitoring logs by fixed time windows.

    Args:
        file_path: Log file path
        schema: Detected schema

    Returns:
        Document chunk list
    """
    file_name = os.path.basename(file_path)
    file_abs_path = os.path.abspath(file_path)
    windows: dict[datetime, list[tuple[Optional[datetime], str, str, int]]] = defaultdict(list)
    no_ts_lines: list[tuple[int, str, str]] = []
    no_ts_sessions: set[str] = set()

    for line_number, raw_line in enumerate(stream_file_lines(file_path), start=1):
        ts, _, clean = _parse_line(raw_line, schema)
        if not _line_overlaps_query_window(ts, query_context):
            continue
        session_label = _extract_session_label(clean, schema)
        if ts is None:
            no_ts_lines.append((line_number, clean, session_label))
            if session_label:
                no_ts_sessions.add(session_label)
            continue
        window_seconds = max(1, SERVER_MONITOR_WINDOW_SECONDS)
        seconds_since_day = (ts.hour * 3600) + (ts.minute * 60) + ts.second
        window_offset = seconds_since_day % window_seconds
        window_start = ts - timedelta(
            seconds=window_offset,
            microseconds=ts.microsecond,
        )
        windows[window_start].append((ts, clean, session_label, line_number))

    docs: list[Document] = []
    for window_start in sorted(windows.keys()):
        lines = windows[window_start]
        sub_index = 0
        batch: list[tuple[Optional[datetime], str, str, int]] = []
        batch_chars = 0

        for entry in lines:
            batch.append(entry)
            batch_chars += len(entry[1]) + 1

            if batch_chars >= MAX_GROUP_CHARS or len(batch) >= UNGROUPED_MAX_LINES_PER_CHUNK:
                timestamps = [t for t, _, _, _ in batch if t is not None]
                session_labels = sorted({session for _, _, session, _ in batch if session})
                line_numbers = [line_no for _, _, _, line_no in batch]
                numbered_lines = [(line_no, text) for _, text, _, line_no in batch]
                start_time = min(timestamps) if timestamps else None
                end_time = max(timestamps) if timestamps else None
                content = '\n'.join(text for _, text, _, _ in batch)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        'source_file': file_name,
                        'source_path': file_abs_path,
                        'primary_key': f"window:{window_start.isoformat()}_{sub_index}",
                        'key_type': 'monitor_window',
                        'start_time': start_time.isoformat() if start_time else '',
                        'end_time': end_time.isoformat() if end_time else '',
                        'line_count': len(batch),
                        'start_line': min(line_numbers) if line_numbers else None,
                        'end_line': max(line_numbers) if line_numbers else None,
                        'line_ranges': _compact_line_ranges(line_numbers),
                        'error_line_ranges': _error_line_ranges_from_numbered_lines(numbered_lines),
                        'sub_index': sub_index,
                        'session_labels': ';'.join(session_labels),
                        'session_label_count': len(session_labels),
                    },
                ))
                sub_index += 1
                batch = []
                batch_chars = 0

        if batch:
            timestamps = [t for t, _, _, _ in batch if t is not None]
            session_labels = sorted({session for _, _, session, _ in batch if session})
            line_numbers = [line_no for _, _, _, line_no in batch]
            numbered_lines = [(line_no, text) for _, text, _, line_no in batch]
            start_time = min(timestamps) if timestamps else None
            end_time = max(timestamps) if timestamps else None
            content = '\n'.join(text for _, text, _, _ in batch)
            docs.append(Document(
                page_content=content,
                metadata={
                    'source_file': file_name,
                    'source_path': file_abs_path,
                    'primary_key': f"window:{window_start.isoformat()}_{sub_index}",
                    'key_type': 'monitor_window',
                    'start_time': start_time.isoformat() if start_time else '',
                    'end_time': end_time.isoformat() if end_time else '',
                    'line_count': len(batch),
                    'start_line': min(line_numbers) if line_numbers else None,
                    'end_line': max(line_numbers) if line_numbers else None,
                    'line_ranges': _compact_line_ranges(line_numbers),
                    'error_line_ranges': _error_line_ranges_from_numbered_lines(numbered_lines),
                    'sub_index': sub_index,
                    'session_labels': ';'.join(session_labels),
                    'session_label_count': len(session_labels),
                },
            ))

    if no_ts_lines:
        buffer: list[str] = []
        buffer_line_numbers: list[int] = []
        chars = 0
        idx = 0
        for line_number, line, _ in no_ts_lines:
            buffer.append(line)
            buffer_line_numbers.append(line_number)
            chars += len(line) + 1
            if chars >= NO_TS_CATCH_ALL_CHARS:
                docs.append(Document(
                    page_content='\n'.join(buffer),
                    metadata={
                        'source_file': file_name,
                        'source_path': file_abs_path,
                        'primary_key': f'no_ts_monitor_{idx}',
                        'key_type': 'monitor_no_timestamp',
                        'start_time': '',
                        'end_time': '',
                        'line_count': len(buffer),
                        'start_line': min(buffer_line_numbers) if buffer_line_numbers else None,
                        'end_line': max(buffer_line_numbers) if buffer_line_numbers else None,
                        'line_ranges': _compact_line_ranges(buffer_line_numbers),
                        'error_line_ranges': _error_line_ranges_from_numbered_lines(list(zip(buffer_line_numbers, buffer))),
                        'sub_index': idx,
                        'session_labels': ';'.join(sorted(no_ts_sessions)),
                        'session_label_count': len(no_ts_sessions),
                    },
                ))
                idx += 1
                buffer = []
                buffer_line_numbers = []
                chars = 0

        if buffer:
            docs.append(Document(
                page_content='\n'.join(buffer),
                metadata={
                    'source_file': file_name,
                    'source_path': file_abs_path,
                    'primary_key': f'no_ts_monitor_{idx}',
                    'key_type': 'monitor_no_timestamp',
                    'start_time': '',
                    'end_time': '',
                    'line_count': len(buffer),
                    'start_line': min(buffer_line_numbers) if buffer_line_numbers else None,
                    'end_line': max(buffer_line_numbers) if buffer_line_numbers else None,
                    'line_ranges': _compact_line_ranges(buffer_line_numbers),
                    'error_line_ranges': _error_line_ranges_from_numbered_lines(list(zip(buffer_line_numbers, buffer))),
                    'sub_index': idx,
                    'session_labels': ';'.join(sorted(no_ts_sessions)),
                    'session_label_count': len(no_ts_sessions),
                },
            ))

    print(f"  [Chunk:ServerMonitoring] Produced {len(docs):,} window chunks")
    return docs


def chunk_api_requests_hierarchical(
    docs: list[Document],
    schema: dict,
    boundaries: dict[str, list[str]],
) -> list[Document]:
    """
    Build additive API request chunks for thread docs from entry/exit markers.

    Base input docs are preserved unchanged. Generated request chunks are appended.

    Args:
        docs: Base thread/time chunks
        schema: Detected log schema
        boundaries: Request boundary markers

    Returns:
        Base docs plus hierarchically generated API docs
    """
    start_markers = boundaries.get('start_markers', _DEFAULT_API_REQUEST_BOUNDARIES['start_markers'])
    end_markers = boundaries.get('end_markers', _DEFAULT_API_REQUEST_BOUNDARIES['end_markers'])

    refined_docs: list[Document] = list(docs)
    request_counter = 0
    generated_request_count = 0

    for doc in docs:
        key_type = str(doc.metadata.get('key_type', ''))
        if key_type != 'thread':
            continue

        lines = doc.page_content.splitlines()
        active_lines: list[str] = []
        active_timestamps: list[datetime] = []
        active_sessions: set[str] = set()

        child_docs: list[Document] = []

        def flush_active(unmatched_exit: bool) -> None:
            nonlocal request_counter
            if not active_lines:
                return
            start_time = min(active_timestamps) if active_timestamps else None
            end_time = max(active_timestamps) if active_timestamps else None
            request_span_id = f"{doc.metadata.get('primary_key', 'thread')}_req_{request_counter}"
            child_docs.append(Document(
                page_content='\n'.join(active_lines),
                metadata={
                    **doc.metadata,
                    'key_type': 'api_request',
                    'chunk_level': 'request',
                    'parent_key_type': key_type,
                    'parent_primary_key': doc.metadata.get('primary_key', ''),
                    'request_index': request_counter,
                    'request_span_id': request_span_id,
                    'unmatched_exit': unmatched_exit,
                    'start_time': start_time.isoformat() if start_time else doc.metadata.get('start_time', ''),
                    'end_time': end_time.isoformat() if end_time else doc.metadata.get('end_time', ''),
                    'line_count': len(active_lines),
                    'session_labels': ';'.join(sorted(active_sessions)),
                    'session_label_count': len(active_sessions),
                },
            ))
            request_counter += 1

        for line in lines:
            ts, _, clean = _parse_line(line, schema)
            session_label = _extract_session_label(clean, schema)
            is_start = _contains_any_marker(clean, start_markers)
            is_end = _contains_any_marker(clean, end_markers)

            if is_start:
                if active_lines:
                    flush_active(unmatched_exit=True)
                    active_lines = []
                    active_timestamps = []
                    active_sessions = set()

                active_lines = [clean]
                active_timestamps = [ts] if ts is not None else []
                if session_label:
                    active_sessions.add(session_label)
                continue

            if active_lines:
                active_lines.append(clean)
                if ts is not None:
                    active_timestamps.append(ts)
                if session_label:
                    active_sessions.add(session_label)

                if is_end:
                    flush_active(unmatched_exit=False)
                    active_lines = []
                    active_timestamps = []
                    active_sessions = set()
                continue

        if active_lines:
            flush_active(unmatched_exit=True)

        if child_docs:
            refined_docs.extend(child_docs)
            generated_request_count += len(child_docs)

    if generated_request_count > 0:
        print(
            f"  [Chunk:API] Generated {generated_request_count:,} request chunks "
            "(additive; base chunks preserved)"
        )
    else:
        print("  [Chunk:API] No request boundaries detected; no additive request chunks created")
    return refined_docs


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
    # Only decode lines that look URL-encoded (contain %XX patterns)
    lines = text.split('\n')
    decoded_lines: list[str] = []
    url_pattern = re.compile(r'%[0-9A-Fa-f]{2}')

    for line in lines:
        if url_pattern.search(line):
            try:
                # First pass: decode HTML entities like %26amp;
                cleaned = line.replace('%26amp;', '&')
                cleaned = cleaned.replace('%26gt;', '>')
                cleaned = cleaned.replace('%26lt;', '<')
                # Second pass: standard URL decoding
                decoded = urllib.parse.unquote_plus(cleaned)
                decoded_lines.append(decoded)
            except Exception:
                decoded_lines.append(line)
        else:
            decoded_lines.append(line)

    return '\n'.join(decoded_lines)


def hybrid_chunk_log(file_path: str, schema: dict) -> list[Document]:
    """
    Stream a log file and produce semantically coherent Document chunks.

    Strategy:
        1. Parse every line for (timestamp, primary_key, text).
        2. Group lines sharing the same primary_key together.
           - Stack-trace continuation lines (no timestamp) inherit the key
             of the preceding line.
           - If a group exceeds 15 000 chars, split chronologically with
             500-char overlap.
        3. Lines with no primary_key are collected separately and chunked
           using 2-minute sliding windows (30-second step) sorted by timestamp.

    Args:
        file_path: Absolute path to the log file
        schema:    Dict returned by detect_log_structure

    Returns:
        List of Document objects with rich metadata
    """
    file_name = os.path.basename(file_path)
    file_abs_path = os.path.abspath(file_path)

    # ---- Pass 1: Stream & parse lines ----
    grouped: dict[str, list[tuple[Optional[datetime], str, int]]] = defaultdict(list)
    ungrouped: list[tuple[Optional[datetime], str, int]] = []
    grouped_sessions: dict[str, set[str]] = defaultdict(set)
    ungrouped_sessions: set[str] = set()

    last_pk: Optional[str] = None
    line_count = 0

    print(f"  [Chunk] Parsing lines from {file_name}...")
    for line_number, raw_line in enumerate(stream_file_lines(file_path), start=1):
        line_count += 1
        ts, pk, clean = _parse_line(raw_line, schema)
        session_label = _extract_session_label(clean, schema)

        # Stack-trace continuation inherits previous primary key
        if pk is None and schema['stack_trace_re'].match(clean):
            pk = last_pk

        if pk is not None:
            grouped[pk].append((ts, clean, line_number))
            if session_label:
                grouped_sessions[pk].add(session_label)
            last_pk = pk
        else:
            ungrouped.append((ts, clean, line_number))
            if session_label:
                ungrouped_sessions.add(session_label)

    print(f"    {line_count:,} lines parsed -> {len(grouped):,} thread groups, "
          f"{len(ungrouped):,} ungrouped lines")

    # ---- Helper: build Document from a list of (ts, text) ----
    def _make_doc(
        lines: list[tuple[Optional[datetime], str, int]],
        key_label: str,
        key_value: str,
        session_labels: Optional[set[str]] = None,
        sub_index: int = 0,
    ) -> Document:
        """
        Build a Document from parsed lines with metadata.

        Args:
            lines:     List of (timestamp, text) tuples
            key_label: Type of key (thread, time_window, no_timestamp)
            key_value: Value of the primary key
            sub_index: Sub-chunk index for split groups

        Returns:
            Document with page_content and metadata
        """
        timestamps = [t for t, _, _ in lines if t is not None]
        line_numbers = [line_no for _, _, line_no in lines]
        numbered_lines = [(line_no, text) for _, text, line_no in lines]
        start = min(timestamps) if timestamps else None
        end = max(timestamps) if timestamps else None
        content = '\n'.join(text for _, text, _ in lines)
        return Document(
            page_content=content,
            metadata={
                'source_file': file_name,
                'source_path': file_abs_path,
                'primary_key': key_value,
                'key_type': key_label,
                'start_time': start.isoformat() if start else '',
                'end_time': end.isoformat() if end else '',
                'line_count': len(lines),
                'start_line': min(line_numbers) if line_numbers else None,
                'end_line': max(line_numbers) if line_numbers else None,
                'line_ranges': _compact_line_ranges(line_numbers),
                'error_line_ranges': _error_line_ranges_from_numbered_lines(numbered_lines),
                'sub_index': sub_index,
                'session_labels': ';'.join(sorted(session_labels or set())),
                'session_label_count': len(session_labels or set()),
            },
        )

    docs: list[Document] = []

    # ---- Pass 2a: Grouped chunks (thread / session) ----
    for pk, entries in grouped.items():
        total_chars = sum(len(t) for _, t, _ in entries)

        if total_chars <= MAX_GROUP_CHARS:
            docs.append(_make_doc(entries, 'thread', pk, grouped_sessions.get(pk, set())))
        else:
            # Split chronologically with overlap
            # Sort by timestamp (None timestamps go to the end)
            entries.sort(key=lambda x: x[0] if x[0] else datetime.max)
            chunk_lines: list[tuple[Optional[datetime], str, int]] = []
            chunk_chars = 0
            sub = 0
            overlap_buffer: list[tuple[Optional[datetime], str, int]] = []

            for entry in entries:
                chunk_lines.append(entry)
                chunk_chars += len(entry[1]) + 1  # +1 for newline

                if chunk_chars >= MAX_GROUP_CHARS:
                    docs.append(_make_doc(chunk_lines, 'thread', pk, grouped_sessions.get(pk, set()), sub))
                    sub += 1
                    # Keep last CHUNK_OVERLAP_CHARS worth of lines as overlap
                    overlap_buffer = []
                    buf_chars = 0
                    for e in reversed(chunk_lines):
                        buf_chars += len(e[1]) + 1
                        overlap_buffer.insert(0, e)
                        if buf_chars >= CHUNK_OVERLAP_CHARS:
                            break
                    chunk_lines = list(overlap_buffer)
                    chunk_chars = sum(len(t) + 1 for _, t, _ in chunk_lines)

            if chunk_lines:
                docs.append(_make_doc(chunk_lines, 'thread', pk, grouped_sessions.get(pk, set()), sub))

    # ---- Pass 2b: Ungrouped lines -> time windows ----
    if ungrouped:
        # Filter to only lines with timestamps, sort
        with_ts = [(t, txt, line_no) for t, txt, line_no in ungrouped if t is not None]
        without_ts = [(t, txt, line_no) for t, txt, line_no in ungrouped if t is None]

        if with_ts:
            with_ts.sort(key=lambda x: x[0])
            window_duration = timedelta(seconds=UNGROUPED_WINDOW_SECONDS)
            # Non-overlapping windows reduce duplicate chunk generation in dense logs.
            step = window_duration
            window_start = with_ts[0][0]
            window_end = with_ts[-1][0] + timedelta(seconds=1)
            win_idx = 0
            ptr = 0
            total_with_ts = len(with_ts)

            while window_start < window_end:
                w_end = window_start + window_duration
                window_lines: list[tuple[Optional[datetime], str, int]] = []

                while ptr < total_with_ts and with_ts[ptr][0] < window_start:
                    ptr += 1

                scan_ptr = ptr
                while scan_ptr < total_with_ts and with_ts[scan_ptr][0] < w_end:
                    window_lines.append(with_ts[scan_ptr])
                    scan_ptr += 1

                if window_lines:
                    label = f"time:{window_start.strftime('%H:%M:%S')}-{w_end.strftime('%H:%M:%S')}"
                    total_win_chars = sum(len(txt) + 1 for _, txt, _ in window_lines)
                    if total_win_chars <= MAX_GROUP_CHARS and len(window_lines) <= UNGROUPED_MAX_LINES_PER_CHUNK:
                        docs.append(_make_doc(window_lines, 'time_window', label, ungrouped_sessions, win_idx))
                        win_idx += 1
                    else:
                        # Split oversized time-window into sub-chunks
                        tw_batch: list[tuple[Optional[datetime], str, int]] = []
                        tw_chars = 0
                        tw_lines = 0
                        for entry in window_lines:
                            tw_batch.append(entry)
                            tw_chars += len(entry[1]) + 1
                            tw_lines += 1
                            if tw_chars >= MAX_GROUP_CHARS or tw_lines >= UNGROUPED_MAX_LINES_PER_CHUNK:
                                docs.append(_make_doc(tw_batch, 'time_window', label, ungrouped_sessions, win_idx))
                                win_idx += 1
                                tw_batch = []
                                tw_chars = 0
                                tw_lines = 0
                        if tw_batch:
                            docs.append(_make_doc(tw_batch, 'time_window', label, ungrouped_sessions, win_idx))
                            win_idx += 1
                ptr = scan_ptr
                window_start += step

        # Lines with no timestamp at all -> single catch-all chunk (capped at ~10K)
        if without_ts:
            sub = 0
            batch: list[tuple[Optional[datetime], str, int]] = []
            batch_chars = 0
            for entry in without_ts:
                batch.append(entry)
                batch_chars += len(entry[1]) + 1
                if batch_chars >= NO_TS_CATCH_ALL_CHARS:
                    docs.append(_make_doc(batch, 'no_timestamp', 'unstructured', ungrouped_sessions, sub))
                    sub += 1
                    batch = []
                    batch_chars = 0
            if batch:
                docs.append(_make_doc(batch, 'no_timestamp', 'unstructured', ungrouped_sessions, sub))

    print(f"    {len(docs):,} chunks produced")
    return docs
