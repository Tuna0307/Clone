"""Text-classification helpers used by chunking, dedup, reporting, and scoring.

Extracted from iam_log_intelligence_agent_hybridChunking2.py to break
circular import dependencies.
"""

from __future__ import annotations

import re
from typing import Optional

from pipeline.constants import (
    _DEFAULT_ERROR_KEYWORDS,
    _DEDUP_UUID_RE,
    _STACK_TRACE_LINE_RE,
)


def _contains_any_marker(text: str, markers: list[str]) -> bool:
    """
    Check if line text contains any marker.

    Args:
        text: Line text
        markers: Marker list

    Returns:
        True when marker is found
    """
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _extract_diagnostic_entities_from_line(
    clean: str,
    seen_entities: set[str],
    diagnostic_entities: list[dict[str, str]],
    max_entities: int,
    prop_kv_re: re.Pattern,
    file_path_re: re.Pattern,
    error_code_re: re.Pattern,
    http_status_re: re.Pattern,
    base64_like_re: re.Pattern,
    schema: dict,
) -> None:
    """
    Extract diagnostic entities from one line into a deduplicated entity list.

    Args:
        clean: Parsed log line
        seen_entities: Normalized values already collected
        diagnostic_entities: Output entity list
        max_entities: Collection cap
        prop_kv_re: Property key/value matcher
        file_path_re: File/config path matcher
        error_code_re: Error code matcher
        http_status_re: HTTP status matcher
        base64_like_re: Long token matcher
        schema: Detected schema for session key extraction
    """
    line_excerpt = clean[:200]

    def _append_entity(entity_type: str, value: str, normalized: str) -> bool:
        value_normalized = f"value:{value.lower()}"
        if normalized in seen_entities or value_normalized in seen_entities:
            return False
        seen_entities.add(normalized)
        seen_entities.add(value_normalized)
        diagnostic_entities.append({
            'type': entity_type,
            'value': value,
            'line_excerpt': line_excerpt,
        })
        return len(diagnostic_entities) >= max_entities

    for match in prop_kv_re.finditer(clean):
        if _append_entity(
            'property',
            f"{match.group(1)}={match.group(2)}",
            f"kv:{match.group(1).lower()}={match.group(2).lower()}",
        ):
            return

    for value in re.findall(r'\b(?:(?:conf|config|etc)[/\\][^\s,;&]+|[\w.-]+\.properties)\b', clean):
        if _append_entity('file_path', value, f"path:{value.lower()}"):
            return

    for match in file_path_re.finditer(clean):
        if _append_entity('file_path', match.group(1), f"path:{match.group(1).lower()}"):
            return

    for match in _DEDUP_UUID_RE.finditer(clean):
        if _append_entity('uuid', match.group(0), f"uuid:{match.group(0).lower()}"):
            return

    for match in error_code_re.finditer(clean):
        if _append_entity('error_code', match.group(1), f"errcode:{match.group(1).lower()}"):
            return

    for match in http_status_re.finditer(clean):
        if _append_entity('http_status', f"HTTP {match.group(1)}", f"http:{match.group(1)}"):
            return

    for match in base64_like_re.finditer(clean):
        token = match.group(0)
        if _append_entity('token_id', token[:80], f"token:{token[:40]}"):
            return

    for compiled, key_name in schema.get('session_keys', []):
        for match in compiled.finditer(clean):
            value = f"{key_name}={match.group(1)}"
            normalized = f"session:{key_name}:{match.group(1).lower()}"
            if _append_entity('session_id', value, normalized):
                return


def _is_error_bearing(text: str, error_keywords: Optional[list[str]] = None) -> bool:
    """
    Determine whether a chunk contains error-bearing signals.

    Args:
        text: Chunk content

    Returns:
        True if text contains error indicators, else False
    """
    keywords = error_keywords if error_keywords is not None else _DEFAULT_ERROR_KEYWORDS
    diagnostic_text = _extract_diagnostic_text(text)
    return any(keyword in diagnostic_text for keyword in keywords)


def _is_noisy_text(text: str, noise_patterns: list[re.Pattern]) -> bool:
    """
    Determine whether a chunk matches known infrastructure noise patterns.

    Args:
        text: Chunk content
        noise_patterns: Compiled regex list

    Returns:
        True if any noise pattern matches
    """
    return any(pattern.search(text) for pattern in noise_patterns)


def _extract_diagnostic_text(text: str) -> str:
    """
    Collapse a chunk to message-bearing lines and drop stack-frame-only lines.

    Args:
        text: Chunk text

    Returns:
        Diagnostic text used for signal detection
    """
    diagnostic_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if _STACK_TRACE_LINE_RE.match(stripped):
            continue
        diagnostic_lines.append(stripped)

    if diagnostic_lines:
        return '\n'.join(diagnostic_lines)
    return text


def _is_iam_critical_text(text: str, iam_keywords: list[str]) -> bool:
    """
    Determine whether chunk content contains IAM-critical keywords.

    Args:
        text: Chunk text
        iam_keywords: IAM-critical keyword list from config

    Returns:
        True if IAM-critical keyword is present
    """
    diagnostic_text = _extract_diagnostic_text(text)
    return any(keyword in diagnostic_text for keyword in iam_keywords)
