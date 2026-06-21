"""Thin builders for follow-up chat prompts."""

from __future__ import annotations

import json

from followup.intent import FOLLOWUP_REPORT_SNIPPET_CHARS
from pipeline.prompt_loader import load_fragment, load_prompt

PROMPT_REGISTRY: list[str] = [
    "followup.intent.system",
    "followup.intent.user",
    "followup.answer.system",
    "followup.answer.system.api_extension",
    "followup.answer.citation.api",
    "followup.answer.citation.default",
    "followup.answer.user",
    "server_monitoring.followup.sql_retry_nudge",
]


def build_intent_messages(
    *,
    original_query: str,
    report_excerpt: str,
    chat_history: str,
    query: str,
) -> tuple[str, str]:
    """Build follow-up intent parser system and user prompts."""
    return (
        load_fragment("followup.intent.system"),
        load_prompt(
            "followup.intent.user",
            original_query=original_query[:1200],
            report_excerpt=report_excerpt[:FOLLOWUP_REPORT_SNIPPET_CHARS],
            chat_history=chat_history,
            query=query,
        ),
    )


def build_answer_messages(
    *,
    original_query: str,
    ticket_block: str,
    chat_history: str,
    query: str,
    intent_payload: dict,
    evidence_block: str,
    api_followup_mode: bool,
) -> tuple[str, str]:
    """Build follow-up answer system and user prompts."""
    system_prompt = load_fragment("followup.answer.system")
    if api_followup_mode:
        system_prompt += " " + load_fragment("followup.answer.system.api_extension").lstrip()

    citation_key = "followup.answer.citation.api" if api_followup_mode else "followup.answer.citation.default"
    citation_instruction = load_fragment(citation_key)

    user_prompt = load_prompt(
        "followup.answer.user",
        original_query=original_query,
        ticket_block=ticket_block,
        chat_history=chat_history,
        query=query,
        intent_payload_json=json.dumps(intent_payload),
        evidence_block=evidence_block,
        citation_instruction=citation_instruction,
    )
    return system_prompt, user_prompt


def build_server_followup_sql_retry_nudge() -> str:
    """Return the server-monitoring follow-up SQL retry nudge fragment."""
    return load_fragment("server_monitoring.followup.sql_retry_nudge")