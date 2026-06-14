"""Log-structure detection and per-line parsing helpers."""

import os
import re
from datetime import datetime
from typing import Optional, Tuple

from pipeline.constants import _QUERY_DATE_ONLY_FORMATS, _QUERY_DATETIME_FORMATS
from pipeline.files import stream_file_lines


# ---------------------------------------------------------------------------
# Line-prefix regex (strips "[Line NNN] " added by pre-filter tools)
# ---------------------------------------------------------------------------
_LINE_PREFIX_RE = re.compile(r'^\[Line \d+\]\s*')

# ---------------------------------------------------------------------------
# Common timestamp patterns ordered by specificity (most specific first)
# ---------------------------------------------------------------------------
_TIMESTAMP_PATTERNS: list[tuple[str, str]] = [
    # ISO-style with millis — "2025-09-12 14:27:45.798"
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', '%Y-%m-%d %H:%M:%S.%f'),
    # ISO-style no millis — "2025-09-12 14:27:45"
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
    # ISO-8601 T separator — "2025-09-12T14:27:45.798"
    (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}', '%Y-%m-%dT%H:%M:%S.%f'),
    # US date with slashes + 4-digit year — "09/12/2025 14:27:45"
    (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '%m/%d/%Y %H:%M:%S'),
    # WebSphere colon-millis + 2-digit year — "[11/24/25 13:59:01:674 BNT]"
    # Hour can be single-digit.  strptime can't parse colon-millis directly;
    # _parse_line normalises the last colon to a dot.
    (r'\d{1,2}/\d{1,2}/\d{2} \d{1,2}:\d{2}:\d{2}:\d{3}', '%m/%d/%y %H:%M:%S.%f'),
    # WebSphere colon-millis + 4-digit year — "[11/24/2025 13:59:01:674 BNT]"
    (r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}:\d{3}', '%m/%d/%Y %H:%M:%S.%f'),
]

# ---------------------------------------------------------------------------
# Common thread-name patterns
# ---------------------------------------------------------------------------
_THREAD_PATTERNS: list[str] = [
    r'\[(https?-[\w-]+-exec-\d+)\]',          # Tomcat NIO executor
    r'\[(https?-[\w-]+-\d+)\]',               # Tomcat-style (other)
    r'\[([\w]+-[\w]+-\d+-\d+)\]',             # Generic word-word-num-num
    r'\[([A-Z][A-Za-z]+ [\w-]+-\d+-\d+)\]',  # Named workers
    r'\]\s+([0-9a-f]{8})\s',                   # WebSphere hex thread ID
    r'\[([\w.-]+)\]',                         # Broad fallback
]

# ---------------------------------------------------------------------------
# Session / transaction key patterns (key=value or key:value or key/value)
# ---------------------------------------------------------------------------
_SESSION_KEY_PATTERNS: list[tuple[str, str]] = [
    (r'(?:txId|tx_id|TXID)[=:/]\s*(\S+)', 'txId'),
    (r'(?:sesId|SESSION_ID|session_id|sessionId)[=:/]\s*(\S+)', 'sesId'),
    (r'(?:IID|iid)[=:/]\s*(\S+)', 'iid'),
    (r'(?:USER_ID|userId|user_id)[=:/]\s*(\S+)', 'userId'),
    (r'(?:e2eeSid)[=:/]\s*(\S+)', 'e2eeSid'),
    (r'(?:correlationId|CORRELATION_ID|corrId)[=:/]\s*(\S+)', 'correlationId'),
]


def _extract_session_label(line: str, schema: dict) -> str:
    """
    Extract session label from a line without using it for chunk grouping.

    Args:
        line: Clean line text
        schema: Detected schema

    Returns:
        Session label string or empty string
    """
    for compiled, key_name in schema.get('session_keys', []):
        match = compiled.search(line)
        if match:
            return f"{key_name}:{match.group(1)}"
    return ''


def _apply_day_boundary(value: datetime, use_end_of_day: bool) -> datetime:
    """
    Apply day boundary for date-only incident inputs.

    Args:
        value: Parsed date value
        use_end_of_day: Whether to set to end-of-day

    Returns:
        Datetime adjusted to start/end of day
    """
    if use_end_of_day:
        return value.replace(hour=23, minute=59, second=59, microsecond=999999)
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_query_datetime(
    value: Optional[str],
    *,
    use_end_of_day_for_date_only: bool = False,
    additional_formats: Optional[list[str]] = None,
) -> Optional[datetime]:
    """
    Parse query datetime text into a datetime object.

    Args:
        value: Query datetime text
        use_end_of_day_for_date_only: Whether date-only value maps to end-of-day
        additional_formats: Extra strptime formats to attempt

    Returns:
        Parsed datetime or None
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    # Date-only inputs first so _apply_day_boundary is honoured.
    for fmt in _QUERY_DATE_ONLY_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return _apply_day_boundary(parsed, use_end_of_day_for_date_only)
        except Exception:
            continue

    normalized = text.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        pass

    candidate_formats = list(_QUERY_DATETIME_FORMATS)
    if additional_formats:
        for fmt in additional_formats:
            if fmt and fmt not in candidate_formats:
                candidate_formats.append(fmt)

    for fmt in candidate_formats:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue

    return None


def _parse_iso_timestamp(value: str) -> Optional[datetime]:
    """
    Parse ISO timestamp metadata.

    Args:
        value: Timestamp text

    Returns:
        Datetime or None
    """
    if not value:
        return None
    return parse_query_datetime(value)


def detect_log_structure(sample_lines: list[str]) -> dict:
    """
    Detect the log schema from the first 800-1200 sample lines using regex
    heuristics.  Returns a dict with compiled regexes and field names so
    downstream functions can parse efficiently without re-compiling.

    Args:
        sample_lines: First 800-1200 lines of the log file

    Returns:
        Schema dict with keys:
            timestamp_re   — compiled regex with one group for the timestamp string
            timestamp_fmt  — strptime format string
            thread_re      — compiled regex with one group for thread name (or None)
            session_keys   — list of (compiled_regex, key_name) tuples
            stack_trace_re — compiled regex to identify continuation / stack trace lines
    """
    schema: dict = {
        'timestamp_re': None,
        'timestamp_fmt': None,
        'thread_re': None,
        'session_keys': [],
        'stack_trace_re': re.compile(
            r'^(?:\s+at |\s*Caused by:|\s*\.\.\. \d+ more|'
            r'\s*[a-zA-Z][\w.$]*(?:Exception|Error|Throwable))'
        ),
    }

    # Strip optional "[Line N] " prefixes from sample lines
    cleaned: list[str] = [_LINE_PREFIX_RE.sub('', l) for l in sample_lines]

    # Pre-filter: exclude stack-trace continuation lines from detection pool.
    # Files with many exceptions can have >80% stack-trace lines, which dilute
    # the hit rate for timestamp/thread patterns below detection thresholds.
    _continuation_re = re.compile(
        r'^(?:\s+at |\s*Caused by:|\s*\.\.\. \d+ more'
        r'|\s+[a-zA-Z][\w.$]*(?:Exception|Error|Throwable)'
        r'|\s*~\[)'                      # Gradle/Maven module hints  ~[am.jar:?]
    )
    primary_lines: list[str] = [
        l for l in cleaned[:1200]
        if l.strip() and not _continuation_re.match(l)
    ]
    # Use up to 200 primary (non-continuation) lines for detection
    detect_pool: list[str] = primary_lines[:200]
    detect_pool_size: int = len(detect_pool) if detect_pool else 1  # avoid div-by-zero

    # --- Detect timestamp pattern ---
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        compiled = re.compile(pattern)
        hits = sum(1 for l in detect_pool if compiled.search(l))
        if hits > detect_pool_size * 0.3:  # 30 % threshold
            schema['timestamp_re'] = re.compile(f'({pattern})')
            schema['timestamp_fmt'] = fmt
            break

    # Fallback: if no timestamp detected, use a permissive regex that won't match
    if schema['timestamp_re'] is None:
        schema['timestamp_re'] = re.compile(r'(NOTIMESTAMP)')
        schema['timestamp_fmt'] = ''

    # --- Detect thread pattern ---
    for pat in _THREAD_PATTERNS:
        compiled = re.compile(pat)
        hits = sum(1 for l in detect_pool if compiled.search(l))
        if hits > detect_pool_size * 0.2:  # 20 % threshold
            schema['thread_re'] = compiled
            break

    # --- Detect session / transaction keys ---
    # Use a wider window (up to 400 primary lines) for session key detection
    session_pool: list[str] = primary_lines[:400]
    for pat, key_name in _SESSION_KEY_PATTERNS:
        compiled = re.compile(pat, re.IGNORECASE)
        hits = sum(1 for l in session_pool if compiled.search(l))
        if hits > 0:
            schema['session_keys'].append((compiled, key_name))

    return schema


def _parse_line(line: str, schema: dict) -> tuple[Optional[datetime], Optional[str], str]:
    """
    Parse one log line using the detected schema.

    Args:
        line:   Raw log line (may include "[Line N] " prefix)
        schema: Dict returned by detect_log_structure

    Returns:
        (timestamp or None, primary_key or None, cleaned line text)
    """
    clean = _LINE_PREFIX_RE.sub('', line).rstrip('\n')

    # --- Timestamp ---
    ts: Optional[datetime] = None
    ts_match = schema['timestamp_re'].search(clean)
    if ts_match and schema['timestamp_fmt']:
        try:
            ts_str = ts_match.group(1)
            # Normalise WebSphere colon-millis (HH:MM:SS:mmm -> HH:MM:SS.mmm)
            # so strptime's %f can parse it correctly.
            if '.%f' in schema['timestamp_fmt'] and ':' == ts_str[-4:-3]:
                ts_str = ts_str[:-4] + '.' + ts_str[-3:]
            ts = datetime.strptime(ts_str, schema['timestamp_fmt'])
        except (ValueError, IndexError):
            pass

    # --- Primary key: thread only (session is metadata label only) ---
    pk: Optional[str] = None
    if schema['thread_re'] is not None:
        m = schema['thread_re'].search(clean)
        if m:
            pk = m.group(1)

    return ts, pk, clean
