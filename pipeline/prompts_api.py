"""Thin builders for API-request, reduce, and schema prompts."""

from __future__ import annotations

import json
from typing import Any

from pipeline.prompt_loader import load_fragment, load_prompt

PROMPT_REGISTRY: list[str] = [
    "api_request.map.system",
    "api_request.map.guardrail",
    "api_request.map.user",
    "api_request.reduce.system",
    "api_request.reduce.guardrail",
    "api_request.reduce.user",
    "server_monitoring.reduce.system",
    "schema.hybrid.system",
    "schema.hybrid.user",
]


def build_api_map_messages(
    *,
    file_name: str,
    category: str,
    subcategory: str,
    evidence_profile: dict[str, Any],
    evidence_text: str,
) -> tuple[str, str]:
    """Build map-phase system and user prompts for the API request path."""
    guardrail = ""
    if category == "api_request":
        guardrail = load_fragment("api_request.map.guardrail")
    system_prompt = load_prompt("api_request.map.system", api_map_guardrail_text=guardrail)
    user_prompt = load_prompt(
        "api_request.map.user",
        file_name=file_name,
        category=category,
        subcategory=subcategory,
        evidence_profile_json=json.dumps(evidence_profile, indent=2),
        evidence_text=evidence_text,
    )
    return system_prompt, user_prompt


def build_reduce_system_prompt(mode: str) -> str:
    """Build the reduce-phase system prompt for the active analysis mode."""
    if mode == "server_monitoring":
        return load_fragment("server_monitoring.reduce.system")
    guardrail = load_fragment("api_request.reduce.guardrail")
    return load_prompt("api_request.reduce.system", reduce_api_guardrail_text=guardrail)


def build_reduce_user_message(compiled_evidence: str) -> str:
    """Build the reduce-phase user message."""
    return load_prompt("api_request.reduce.user", compiled_evidence=compiled_evidence)


def build_schema_messages(sample_lines: list[str]) -> tuple[str, str]:
    """Build hybrid schema fallback system and user prompts."""
    sample_text = "\n".join(sample_lines[:50])
    return (
        load_fragment("schema.hybrid.system"),
        load_prompt("schema.hybrid.user", sample_text=sample_text),
    )