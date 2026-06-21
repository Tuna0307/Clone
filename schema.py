"""
Hybrid schema detection for low-confidence regex cases.

The active pipeline uses fast regex detection in pipeline.parsing first.
This module provides detect_log_structure_hybrid for optional LLM-assisted
fallback when regex confidence is low.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from pipeline.parsing import detect_log_structure

llm = get_llm()

_LINE_PREFIX_RE = re.compile(r'^\[Line \d+\]\s*')


def _generate_schema_llm(sample_lines: list[str]) -> Optional[dict[str, Any]]:
    """Use LLM to infer log schema when regex heuristics fail."""
    from pipeline.prompts_api import build_schema_messages

    system_prompt, human_prompt = build_schema_messages(sample_lines)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        schema = json.loads(content.strip())

        compiled_schema = {
            'timestamp_re': None,
            'timestamp_fmt': schema.get('timestamp_format'),
            'thread_re': None,
            'session_keys': [],
            'stack_trace_re': re.compile(r'^(?:\s+at |\s*Caused by:)'),
        }

        if schema.get('timestamp_regex'):
            compiled_schema['timestamp_re'] = re.compile(f"({schema['timestamp_regex']})")

        if schema.get('thread_regex'):
            compiled_schema['thread_re'] = re.compile(schema['thread_regex'])

        for key in schema.get('session_keys', []):
            compiled_schema['session_keys'].append((
                re.compile(key['regex'], re.IGNORECASE),
                key['name'],
            ))

        return compiled_schema

    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM schema inference failed: {e}")
        return None


def _detect_multiline_properties(sample_lines: list[str], schema: dict) -> None:
    """Analyze samples to determine if logs are multiline and set continuation patterns."""
    cleaned = [_LINE_PREFIX_RE.sub('', l).rstrip('\n') for l in sample_lines]

    stack_trace_re = schema.get('stack_trace_re')
    stack_trace_count = 0
    if stack_trace_re:
        stack_trace_count = sum(1 for l in cleaned if stack_trace_re.match(l))

    timestamp_re = schema.get('timestamp_re')
    no_ts_indices = []

    for i, line in enumerate(cleaned):
        if not line.strip():
            continue
        if timestamp_re:
            match = timestamp_re.search(line)
            if not match or match.start() > 10:
                no_ts_indices.append(i)
        else:
            return

    no_ts_count = len(no_ts_indices)
    ratio_no_ts = no_ts_count / len(cleaned) if cleaned else 0

    whitespace_start_count = sum(1 for l in cleaned if l and l[0].isspace())
    ratio_whitespace = whitespace_start_count / len(cleaned) if cleaned else 0

    is_multiline = (
        stack_trace_count > 0 or
        ratio_no_ts > 0.15 or
        ratio_whitespace > 0.15
    )

    if is_multiline:
        schema['is_multiline'] = True

        if stack_trace_count > 0:
            schema['continuation_re'] = stack_trace_re
        elif ratio_whitespace > 0.1:
            schema['continuation_re'] = re.compile(r'^\s+')
        else:
            schema['continuation_re'] = re.compile(r'^(?!\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{2})')

        print(f"Multiline detected: stack_traces={stack_trace_count}, "
              f"no_ts_ratio={ratio_no_ts:.2f}, whitespace_ratio={ratio_whitespace:.2f}")
    else:
        schema['is_multiline'] = False
        schema['continuation_re'] = None


def _validate_multiline_detection(schema: dict, sample_lines: list[str]) -> None:
    """Sanity-check multiline detection to avoid false positives."""
    if not schema.get('is_multiline'):
        return

    cleaned = [_LINE_PREFIX_RE.sub('', l).rstrip('\n') for l in sample_lines]
    timestamp_re = schema.get('timestamp_re')
    continuation_re = schema.get('continuation_re')

    if not timestamp_re or not continuation_re:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print("Multiline disabled: missing timestamp or continuation regex")
        return

    continuation_count = sum(1 for l in cleaned if continuation_re.match(l))
    ratio_continuation = continuation_count / len(cleaned) if cleaned else 0

    if ratio_continuation > 0.8:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print(f"Multiline disabled: too many continuation matches ({ratio_continuation:.1%})")
        return

    if ratio_continuation < 0.01:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print(f"Multiline disabled: too few continuation matches ({ratio_continuation:.1%})")
        return

    print(f"Multiline validation passed: {ratio_continuation:.1%} continuation lines")


def _calculate_detection_confidence(schema: dict, sample_lines: list[str]) -> float:
    """Calculate confidence score (0.0 - 1.0) for detected schema."""
    score = 0.0
    weights = {'timestamp': 0.5, 'thread': 0.3, 'session': 0.2}

    cleaned = [_LINE_PREFIX_RE.sub('', l) for l in sample_lines[:200]]
    if not cleaned:
        return 0.0

    if schema.get('timestamp_re'):
        hits = sum(1 for l in cleaned if schema['timestamp_re'].search(l))
        score += weights['timestamp'] * (hits / len(cleaned))

    if schema.get('thread_re'):
        hits = sum(1 for l in cleaned if schema['thread_re'].search(l))
        score += weights['thread'] * (hits / len(cleaned))

    if schema.get('session_keys'):
        for compiled, _ in schema['session_keys']:
            hits = sum(1 for l in cleaned if compiled.search(l))
            if hits > 0:
                score += weights['session']
                break

    if schema.get('is_multiline'):
        continuation_re = schema.get('continuation_re')
        if continuation_re:
            cont_hits = sum(1 for l in cleaned if continuation_re.match(l))
            cont_ratio = cont_hits / len(cleaned)
            if 0.05 <= cont_ratio <= 0.4:
                score += 0.05
            elif cont_ratio > 0.6:
                score -= 0.1

    return max(0.0, min(1.0, score))


def detect_log_structure_hybrid(
    sample_lines: list[str],
    use_llm_fallback: bool = True,
    enable_multiline: bool = True,
) -> dict:
    """
    Detect log schema using regex heuristics first, LLM as fallback.

    Args:
        sample_lines: First 800-1200 lines of the log file
        use_llm_fallback: Whether to use LLM if regex confidence is low
        enable_multiline: Whether to detect and enable multiline entry grouping

    Returns:
        Schema dict compatible with pipeline parsing helpers
    """
    schema = detect_log_structure(sample_lines)

    schema['is_multiline'] = False
    schema['continuation_re'] = None

    confidence = _calculate_detection_confidence(schema, sample_lines)

    if enable_multiline:
        _detect_multiline_properties(sample_lines, schema)

    if use_llm_fallback and confidence < 0.5:
        print(f"Regex detection confidence low ({confidence:.2f}), trying LLM...")
        llm_schema = _generate_schema_llm(sample_lines)

        if llm_schema:
            if schema['timestamp_re'] is None and llm_schema.get('timestamp_re'):
                schema['timestamp_re'] = llm_schema['timestamp_re']
                schema['timestamp_fmt'] = llm_schema.get('timestamp_fmt')

            if schema['thread_re'] is None and llm_schema.get('thread_re'):
                schema['thread_re'] = llm_schema['thread_re']

            if not schema['session_keys'] and llm_schema.get('session_keys'):
                schema['session_keys'] = llm_schema['session_keys']

            if enable_multiline:
                if not schema['is_multiline'] and llm_schema.get('is_multiline'):
                    schema['is_multiline'] = llm_schema['is_multiline']
                    schema['continuation_re'] = llm_schema.get('continuation_re')
                elif schema['is_multiline'] and llm_schema.get('continuation_re'):
                    schema['continuation_re'] = llm_schema['continuation_re']

            print("LLM fallback successful!")
        else:
            print("LLM fallback failed, using regex schema")

    if enable_multiline and schema['is_multiline']:
        _validate_multiline_detection(schema, sample_lines)

    return schema