"""Query utilities for routing, classification, and time-window handling."""

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from pipeline.constants import *
from pipeline.files import stream_file_lines
from pipeline.parsing import _parse_iso_timestamp, _parse_line, parse_query_datetime

llm = get_llm()


def load_search_config(config_path: str = "search_config.json") -> dict:
    """
    Load search configuration from JSON file.
    Returns default config if file is missing or invalid.

    Args:
        config_path: Path to the search config JSON file

    Returns:
        Dict containing config-backed retrieval signals
    """
    default_config: dict = {
        "search_strategy": "signal_first_anomaly_ranking",
        "iam_critical_keywords": list(_DEFAULT_IAM_CRITICAL_KEYWORDS),
        "error_keywords": list(_DEFAULT_ERROR_KEYWORDS),
        "noise_patterns": list(_DEFAULT_NOISE_PATTERNS),
        "category_keywords": dict(_DEFAULT_CATEGORY_KEYWORDS),
        "api_known_error_keywords": list(_DEFAULT_API_KNOWN_ERROR_KEYWORDS),
        "api_request_boundaries": dict(_DEFAULT_API_REQUEST_BOUNDARIES),
    }

    if not os.path.exists(config_path):
        print(f"[Config] Warning: {config_path} not found. Using defaults.")
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            if not isinstance(config, dict):
                print(f"[Config] Invalid format in {config_path}. Using defaults.")
                return default_config

            merged = default_config.copy()
            if isinstance(config.get("search_strategy"), str):
                merged["search_strategy"] = config["search_strategy"]
            for key in (
                "iam_critical_keywords",
                "error_keywords",
                "noise_patterns",
                "api_known_error_keywords",
            ):
                value = config.get(key)
                if isinstance(value, list):
                    merged[key] = value
            if isinstance(config.get("category_keywords"), dict):
                merged["category_keywords"] = config["category_keywords"]
            if isinstance(config.get("api_request_boundaries"), dict):
                merged["api_request_boundaries"] = config["api_request_boundaries"]
            return merged
    except Exception as e:
        print(f"[Config] Error loading {config_path}: {e}")
        return default_config


def load_retrieval_signals(config_path: str = "search_config.json") -> dict:
    """
    Load and normalize retrieval signals used by candidate filtering and ranking.

    Args:
        config_path: Path to the search config JSON file

    Returns:
        Dict with IAM keywords, error keywords, and compiled noise patterns
    """
    config = load_search_config(config_path)
    iam_keywords = [str(keyword) for keyword in config.get('iam_critical_keywords', []) if str(keyword).strip()]
    error_keywords = [str(keyword) for keyword in config.get('error_keywords', []) if str(keyword).strip()]
    noise_patterns = [
        re.compile(str(pattern), re.IGNORECASE)
        for pattern in config.get('noise_patterns', [])
        if str(pattern).strip()
    ]
    return {
        'iam_critical_keywords': iam_keywords,
        'error_keywords': error_keywords,
        'noise_patterns': noise_patterns,
        'category_keywords': config.get('category_keywords', dict(_DEFAULT_CATEGORY_KEYWORDS)),
        'api_known_error_keywords': [
            str(keyword).lower() for keyword in config.get('api_known_error_keywords', []) if str(keyword).strip()
        ],
        'api_request_boundaries': config.get('api_request_boundaries', dict(_DEFAULT_API_REQUEST_BOUNDARIES)),
    }


def _schema_query_formats(schema: dict) -> list[str]:
    """
    Build query parse formats using detected log timestamp format.

    Args:
        schema: Detected schema

    Returns:
        Additional datetime formats to try for query inputs
    """
    formats: list[str] = []
    schema_fmt = str(schema.get('timestamp_fmt') or '').strip()
    if schema_fmt:
        formats.append(schema_fmt)
    if '%H:%M:%S.%f' in schema_fmt:
        ws_variant = schema_fmt.replace('%H:%M:%S.%f', '%H:%M:%S:%f')
        if ws_variant not in formats:
            formats.append(ws_variant)
    return formats


def build_query_context(
    query_text: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build structured query context for routing and validation.

    Args:
        query_text: User incident/query text
        start_time: Optional incident start datetime text
        end_time: Optional incident end datetime text

    Returns:
        Query context dict
    """
    parsed_start = parse_query_datetime(start_time, use_end_of_day_for_date_only=False)
    parsed_end = parse_query_datetime(end_time, use_end_of_day_for_date_only=True)

    time_parse_errors: list[str] = []
    if (start_time or '').strip() and parsed_start is None:
        time_parse_errors.append(
            f"Incident start could not be parsed: '{start_time}'."
        )
    if (end_time or '').strip() and parsed_end is None:
        time_parse_errors.append(
            f"Incident end could not be parsed: '{end_time}'."
        )

    return {
        'query_text': query_text.strip(),
        'start_time': parsed_start,
        'end_time': parsed_end,
        'start_time_raw': (start_time or '').strip(),
        'end_time_raw': (end_time or '').strip(),
        'time_parse_errors': time_parse_errors,
    }


def build_query_filter_summary(query_context: Optional[dict[str, Any]]) -> Optional[str]:
    """
    Build a human-readable query time-filter summary.

    Args:
        query_context: Query context dict

    Returns:
        Summary string, or None when no time filter is provided
    """
    if query_context is None:
        return None

    start_raw = str(query_context.get('start_time_raw', '')).strip()
    end_raw = str(query_context.get('end_time_raw', '')).strip()

    if start_raw and end_raw:
        return f"Log filtered from {start_raw} to {end_raw}."
    if start_raw:
        return f"Log filtered {start_raw} onwards."
    if end_raw:
        return f"Log filtered up to {end_raw}."
    return None


def _classify_query_category_fallback(query_text: str, category_keywords: dict[str, list[str]]) -> str:
    """
    Deterministic category fallback based on keyword overlap.

    Args:
        query_text: User query text
        category_keywords: Category keyword map

    Returns:
        Category label
    """
    text = query_text.lower()
    scores: dict[str, int] = {}
    for category, keywords in category_keywords.items():
        scores[category] = sum(1 for keyword in keywords if str(keyword).lower() in text)

    if scores.get('server_monitoring', 0) > scores.get('api_request', 0):
        return 'server_monitoring'
    return 'api_request'


def _extract_json_payload(text: str) -> Optional[dict[str, Any]]:
    """
    Extract first valid JSON object from an LLM response.

    Args:
        text: Raw model text

    Returns:
        Parsed JSON object or None
    """
    stripped = text.strip()
    if not stripped:
        return None

    candidate = stripped
    if not candidate.startswith('{'):
        start_idx = candidate.find('{')
        end_idx = candidate.rfind('}')
        if start_idx >= 0 and end_idx > start_idx:
            candidate = candidate[start_idx:end_idx + 1]

    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None

    return None


def _build_category_router_prompt(
    category_keywords: dict[str, list[str]],
    api_known_error_keywords: list[str],
) -> str:
    """
    Build a strict routing prompt with category semantics and examples.

    Args:
        category_keywords: Category keyword map
        api_known_error_keywords: IAM/API known error signature keywords

    Returns:
        System prompt for routing
    """
    api_kw = ', '.join(str(k) for k in category_keywords.get('api_request', [])[:20])
    monitor_kw = ', '.join(str(k) for k in category_keywords.get('server_monitoring', [])[:20])
    known_err_kw = ', '.join(str(k) for k in api_known_error_keywords[:24])

    return (
        "You are a high-precision query router for IAM incident analysis. "
        "Classify the user query into exactly one label: api_request or server_monitoring.\n\n"
        "Category definitions:\n"
        "- api_request: authentication/authorization/login/logout/session/token/user-activation/user-access/API errors, "
        "service-to-service auth, crypto/HSM/token verification failures, request/response failures.\n"
        "- server_monitoring: infrastructure health/performance/resource diagnostics such as CPU, memory, heap, "
        "GC, latency, throughput, thread-pool saturation, host/server slowdowns/timeouts not tied to API auth failures.\n\n"
        "Strong routing guidance:\n"
        "- Any user activation/access/login/token/authentication/authorization failure should route to api_request.\n"
        "- Prefer api_request for IAM/business-flow failures unless the query is clearly about infrastructure metrics.\n"
        "- server_monitoring is for system performance/health incidents.\n\n"
        f"api_request hint keywords: {api_kw}\n"
        f"server_monitoring hint keywords: {monitor_kw}\n"
        f"IAM known error signatures: {known_err_kw}\n\n"
        "Examples:\n"
        "- 'Failure to activate a user on our platform' -> api_request\n"
        "- 'Users fail login with token verification error' -> api_request\n"
        "- 'CPU spikes and GC pauses causing slowness' -> server_monitoring\n"
        "- 'Thread pool saturation and high latency on server' -> server_monitoring\n\n"
        "Return JSON only with this schema:\n"
        "{\"category\": \"api_request|server_monitoring\", \"confidence\": 0.0-1.0, \"reason\": \"short rationale\"}."
    )


def classify_query_category(
    query_text: str,
    category_keywords: dict[str, list[str]],
    api_known_error_keywords: Optional[list[str]] = None,
) -> tuple[str, float, str, bool]:
    """
    Classify the primary category for a query.

    Args:
        query_text: User query text
        category_keywords: Category keyword map
        api_known_error_keywords: Known API/IAM error signature hints

    Returns:
        Tuple(category, confidence, reason, fallback_used)
    """
    stripped = query_text.strip()
    if not stripped:
        return 'api_request', 1.0, 'empty query defaults to api_request', True

    known_error_keywords = api_known_error_keywords or []
    fallback_category = _classify_query_category_fallback(stripped, category_keywords)
    fallback_reason = 'deterministic keyword fallback'

    system_text = _build_category_router_prompt(category_keywords, known_error_keywords)
    parser_followup = (
        "Return JSON only. Do not include markdown, code fences, prose, or additional keys. "
        "Required keys: category, confidence, reason."
    )

    for attempt in range(1, ROUTER_MAX_ATTEMPTS + 1):
        try:
            user_text = f"Query:\n{stripped}"
            if attempt > 1:
                user_text += f"\n\n{parser_followup}"

            response = llm.invoke([
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ])
            response_text = str(getattr(response, 'content', '')).strip()
            payload = _extract_json_payload(response_text)
            if payload is None:
                print(f"  [Route] Router parse failed on attempt {attempt}.")
                continue

            category = str(payload.get('category', '')).strip().lower()
            if category not in {'api_request', 'server_monitoring'}:
                print(f"  [Route] Router returned invalid category on attempt {attempt}: {category}")
                continue

            raw_confidence = payload.get('confidence', 0.0)
            try:
                confidence = float(raw_confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reason = str(payload.get('reason', '')).strip() or 'llm router decision'
            if confidence < ROUTER_MIN_CONFIDENCE:
                print(
                    f"  [Route] Low-confidence router output ({confidence:.2f}) -> fallback {fallback_category}"
                )
                return fallback_category, confidence, reason, True

            return category, confidence, reason, False
        except Exception as e:
            print(f"  [Route] LLM category classification failed on attempt {attempt}: {e}")

    return fallback_category, 0.0, fallback_reason, True


def classify_api_subcategory(query_text: str, known_error_keywords: list[str]) -> str:
    """
    Classify API query into known or unknown error mode.

    Args:
        query_text: User query text
        known_error_keywords: Known signature keywords

    Returns:
        Subcategory string
    """
    text = query_text.lower()
    if any(keyword in text for keyword in known_error_keywords):
        return 'known_error'
    return 'unknown_error'


def _lazy_get_detect_log_structure_hybrid() -> Optional[Callable[..., dict]]:
    """
    Lazily import hybrid schema detector to avoid eager import side effects.

    Returns:
        detect_log_structure_hybrid callable or None when unavailable
    """
    try:
        from schema import detect_log_structure_hybrid  # type: ignore
        return detect_log_structure_hybrid
    except Exception as e:
        print(f"  [Schema] Hybrid detector unavailable: {e}")
        return None


def _should_try_hybrid_schema(schema: dict) -> bool:
    """
    Decide whether regex schema detection is insufficient.

    Args:
        schema: Schema from local regex detector

    Returns:
        True when hybrid detection should be attempted
    """
    timestamp_missing = schema.get('timestamp_fmt', '') == ''
    thread_missing = schema.get('thread_re') is None
    return timestamp_missing or thread_missing


def _align_datetime_timezone(value: Optional[datetime], reference: Optional[datetime]) -> Optional[datetime]:
    """
    Align naive/aware datetime with reference timezone style.

    Args:
        value: Candidate datetime
        reference: Reference datetime

    Returns:
        Datetime with compatible timezone semantics
    """
    if value is None or reference is None:
        return value
    value_has_tz = value.tzinfo is not None
    reference_has_tz = reference.tzinfo is not None
    if value_has_tz == reference_has_tz:
        return value
    if reference_has_tz:
        return value.replace(tzinfo=reference.tzinfo)
    return value.replace(tzinfo=None)


def compute_docs_time_coverage(docs: list[Document]) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Compute min/max timestamps across chunk metadata.

    Args:
        docs: Document list

    Returns:
        (min_ts, max_ts)
    """
    start_values: list[datetime] = []
    end_values: list[datetime] = []
    for doc in docs:
        start_dt = _parse_iso_timestamp(str(doc.metadata.get('start_time', '')))
        end_dt = _parse_iso_timestamp(str(doc.metadata.get('end_time', '')))
        if start_dt is not None:
            start_values.append(start_dt)
        if end_dt is not None:
            end_values.append(end_dt)

    min_ts = min(start_values) if start_values else None
    max_ts = max(end_values) if end_values else (max(start_values) if start_values else None)
    return min_ts, max_ts


def validate_query_window(
    query_context: Optional[dict[str, Any]],
    min_ts: Optional[datetime],
    max_ts: Optional[datetime],
) -> tuple[bool, str, str]:
    """
    Validate query incident window against log coverage.

    Args:
        query_context: Query context dict
        min_ts: Minimum log timestamp
        max_ts: Maximum log timestamp

    Returns:
        (is_valid, reason_code, message)
    """
    if query_context is None:
        return True, 'ok', 'No query window provided.'

    start_time = query_context.get('start_time')
    end_time = query_context.get('end_time')

    if start_time is None and end_time is None:
        return True, 'ok', 'No query window provided.'

    if min_ts is None or max_ts is None:
        return False, 'no_log_timestamps', 'Log has no timestamp coverage for window validation.'

    reference_ts = min_ts if min_ts is not None else max_ts
    if reference_ts is not None:
        start_time = _align_datetime_timezone(start_time, reference_ts)
        end_time = _align_datetime_timezone(end_time, reference_ts)
        min_ts = _align_datetime_timezone(min_ts, reference_ts)
        max_ts = _align_datetime_timezone(max_ts, reference_ts)

    if start_time is not None and end_time is not None and start_time > end_time:
        return False, 'invalid_time', 'Query start_time is later than end_time.'

    if start_time is not None and start_time < min_ts:
        return False, 'pre_log_boundary', (
            f"Requested start_time ({start_time.isoformat()}) is before "
            f"available log coverage start ({min_ts.isoformat()})."
        )

    if end_time is not None and end_time < min_ts:
        return False, 'pre_log_boundary', (
            f"Requested end_time ({end_time.isoformat()}) is before "
            f"available log coverage start ({min_ts.isoformat()})."
        )

    window_start = start_time if start_time is not None else min_ts
    window_end = end_time if end_time is not None else max_ts

    overlaps = not (window_end < min_ts or window_start > max_ts)
    if overlaps:
        return True, 'ok', 'Query window overlaps log coverage.'

    if window_end.date() < min_ts.date() or window_start.date() > max_ts.date():
        return False, 'invalid_date', (
            f"Requested date range ({window_start.date()} to {window_end.date()}) "
            f"is outside log coverage ({min_ts.date()} to {max_ts.date()})."
        )

    return False, 'invalid_time', (
        f"Requested time window ({window_start.isoformat()} to {window_end.isoformat()}) "
        f"does not overlap available log timestamps ({min_ts.isoformat()} to {max_ts.isoformat()})."
    )


def filter_docs_by_query_window(docs: list[Document], query_context: Optional[dict[str, Any]]) -> list[Document]:
    """
    Filter chunk docs to those overlapping the query window.

    Args:
        docs: Chunk docs
        query_context: Query context dict

    Returns:
        Filtered docs
    """
    if query_context is None:
        return docs

    start_time = query_context.get('start_time')
    end_time = query_context.get('end_time')
    if start_time is None and end_time is None:
        return docs

    result: list[Document] = []
    for doc in docs:
        doc_start = _parse_iso_timestamp(str(doc.metadata.get('start_time', '')))
        doc_end = _parse_iso_timestamp(str(doc.metadata.get('end_time', '')))

        if doc_start is None and doc_end is None:
            # Keep no-timestamp chunks: they cannot be proven outside the
            # requested window and may still contain critical/error signals.
            result.append(doc)
            continue

        reference_ts = doc_start if doc_start is not None else doc_end
        start_time_aligned = _align_datetime_timezone(start_time, reference_ts)
        end_time_aligned = _align_datetime_timezone(end_time, reference_ts)

        left = start_time_aligned if start_time_aligned is not None else doc_start
        right = end_time_aligned if end_time_aligned is not None else doc_end

        if left is None or right is None:
            result.append(doc)
            continue

        if doc_start is None:
            doc_start = doc_end
        if doc_end is None:
            doc_end = doc_start

        if doc_start is None or doc_end is None:
            continue

        overlaps = not (doc_end < left or doc_start > right)
        if overlaps:
            result.append(doc)
    return result


def compute_file_time_coverage(file_path: str, schema: dict) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Compute min/max timestamps directly from raw file lines.

    Args:
        file_path: Log file path
        schema: Detected schema

    Returns:
        (min_ts, max_ts) from parsed log lines
    """
    min_ts: Optional[datetime] = None
    max_ts: Optional[datetime] = None

    for raw_line in stream_file_lines(file_path):
        ts, _, _ = _parse_line(raw_line, schema)
        if ts is None:
            continue
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts

    return min_ts, max_ts


def _line_overlaps_query_window(ts: Optional[datetime], query_context: Optional[dict[str, Any]]) -> bool:
    """
    Check whether a line timestamp overlaps the query window.

    Args:
        ts: Parsed line timestamp
        query_context: Query context dict

    Returns:
        True if line should be retained
    """
    if query_context is None:
        return True

    start_time = query_context.get('start_time')
    end_time = query_context.get('end_time')
    if start_time is None and end_time is None:
        return True

    if ts is None:
        return True

    start_time = _align_datetime_timezone(start_time, ts)
    end_time = _align_datetime_timezone(end_time, ts)

    if start_time is not None and ts < start_time:
        return False
    if end_time is not None and ts > end_time:
        return False
    return True
