"""Follow-up intent parsing helpers extracted from followup_retrieval.py."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MAX_TOKENS, LLM_MODEL_ID, LLM_TEMPERATURE
from llm_factory import get_llm
from followup.context import AnalysisContext, FollowupIntent, _as_float

_FOLLOWUP_LLM = None

FOLLOWUP_INTENT_MIN_CONFIDENCE = 0.45
FOLLOWUP_CHAT_HISTORY_TURNS = 10
FOLLOWUP_CHAT_HISTORY_CHAR_CAP = 6000
FOLLOWUP_SHORT_QUERY_MAX_CHARS = 48
FOLLOWUP_REPORT_SNIPPET_CHARS = 5000

_SUMMARY_BROAD_MARKERS = {
    "other",
    "others",
    "else",
    "additional",
    "another",
    "remaining",
    "rest",
    "more",
    "all",
    "issues",
    "problem",
    "problems",
    "summary",
    "summarize",
}

_BROAD_ISSUE_TERMS = [
    "error",
    "exception",
    "failed",
    "failure",
    "anomaly",
    "incident",
    "token",
    "timeout",
    "connection",
    "refused",
    "invalid",
    "denied",
]


def _get_followup_llm():
    """
    Lazily initialize follow-up LLM.

    Returns:
        Configured LLM instance (provider-agnostic)
    """
    global _FOLLOWUP_LLM
    if _FOLLOWUP_LLM is None:
        _FOLLOWUP_LLM = get_llm()
    return _FOLLOWUP_LLM


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Extract first JSON object from text.

    Args:
        text: Raw model response content

    Returns:
        Parsed dictionary or None
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = text[start:index + 1]
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
    return None


def _format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
    """
    Build compact string representation of recent chat turns.

    Args:
        chat_history: Streamlit-style messages list

    Returns:
        Formatted chat history
    """
    if not chat_history:
        return "(no prior chat turns)"

    recent_turns = chat_history[-FOLLOWUP_CHAT_HISTORY_TURNS:]
    lines: list[str] = []
    for turn in recent_turns:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:800]}")
    history_text = "\n".join(lines) if lines else "(no prior chat turns)"
    return history_text[:FOLLOWUP_CHAT_HISTORY_CHAR_CAP]


def _fallback_intent_from_query(
    query: str,
    chat_history: list[dict[str, str]] | None,
) -> Optional[FollowupIntent]:
    """
    Build deterministic fallback intent for short/underspecified follow-ups.

    Args:
        query: Current user query
        chat_history: Prior session turns

    Returns:
        FollowupIntent when fallback is possible, else None
    """
    text = query.strip()
    if not text:
        return None

    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_./:-]{2,}", lowered)
    token_set = set(tokens)
    history_text = _format_chat_history(chat_history).lower()
    combined = f"{lowered} {history_text}"

    ask_type = "summary"
    if any(word in combined for word in ["timeline", "when", "time range", "chronology"]):
        ask_type = "timeline"
    elif any(word in combined for word in ["root cause", "cause", "why"]):
        ask_type = "root_cause"
    elif any(word in combined for word in ["thread", "session", "primary key", "key="]):
        ask_type = "thread"
    elif any(word in combined for word in ["anomaly", "outlier", "unusual"]):
        ask_type = "anomalies"
    elif any(word in combined for word in ["error", "exception", "failure", "failed", "issue", "problem"]):
        ask_type = "errors"

    broad_summary = bool(token_set.intersection(_SUMMARY_BROAD_MARKERS))
    entities = [term for term in re.findall(r"[A-Za-z0-9_./:-]{3,}", text)][:10]
    must_include: list[str] = []

    if broad_summary:
        ask_type = "summary"
        must_include = list(_BROAD_ISSUE_TERMS[:6])

    if len(text) <= FOLLOWUP_SHORT_QUERY_MAX_CHARS or broad_summary:
        return FollowupIntent(
            ask_type=ask_type,
            entities=entities,
            primary_keys=[],
            must_include=must_include,
            confidence=0.62 if broad_summary else 0.55,
            notes="deterministic fallback for short/underspecified follow-up",
        )

    return None


def _parse_intent(
    context: AnalysisContext,
    query: str,
    chat_history: list[dict[str, str]] | None,
) -> tuple[Optional[FollowupIntent], Optional[str]]:
    """
    Parse user follow-up intent with LLM.

    Args:
        context: Active analysis context
        query: Current user message
        chat_history: Prior session turns

    Returns:
        Tuple of parsed intent or rephrase message
    """
    llm = _get_followup_llm()
    history_text = _format_chat_history(chat_history)
    fallback_intent = _fallback_intent_from_query(query, chat_history)

    from followup.prompts import build_intent_messages

    system_prompt, human_prompt = build_intent_messages(
        original_query=context.query_text,
        report_excerpt=context.report_text,
        chat_history=history_text,
        query=query,
    )

    try:
        response = llm.invoke([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
    except Exception as error:
        if fallback_intent is not None:
            return fallback_intent, None
        return None, (
            "I couldn't interpret that follow-up right now due to an LLM parsing error. "
            f"Please rephrase your question in one sentence. Details: {error}"
        )

    payload = _extract_first_json_object(str(getattr(response, "content", "")))
    if not payload:
        if fallback_intent is not None:
            return fallback_intent, None
        return None, "I couldn't clearly parse your request. Please rephrase with the exact information you need."

    confidence = _as_float(payload.get("confidence", 0.0))
    intent = FollowupIntent(
        ask_type=str(payload.get("ask_type", "other")).strip().lower() or "other",
        entities=[str(item).strip() for item in payload.get("entities", []) if str(item).strip()],
        primary_keys=[str(item).strip() for item in payload.get("primary_keys", []) if str(item).strip()],
        must_include=[str(item).strip() for item in payload.get("must_include", []) if str(item).strip()],
        confidence=max(0.0, min(confidence, 1.0)),
        notes=str(payload.get("notes", "")).strip(),
    )

    if intent.confidence < FOLLOWUP_INTENT_MIN_CONFIDENCE:
        if fallback_intent is not None:
            return fallback_intent, None
        return None, (
            "I couldn't confidently infer your intent from that follow-up. "
            "Please rephrase with the target detail (for example: exact error, timeline range, or thread/key)."
        )

    return intent, None
