"""Source-candidate extraction helpers for artifact-first follow-up retrieval."""

from __future__ import annotations

import os
import re
from collections import deque
from typing import Any

from followup.context import AnalysisContext, EvidenceItem

FOLLOWUP_EVIDENCE_PREVIEW_CHARS = 320
FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE = 3
FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET = 7
FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS = 900

_REF_ID_RE = re.compile(r"\[(REF_[^\]]+)\]")


def _preview_text(text: str, max_chars: int = FOLLOWUP_EVIDENCE_PREVIEW_CHARS) -> str:
    """Build one-line preview."""
    return text.replace("\n", " ")[:max_chars]


def parse_debug_evidence_file(file_path: str) -> dict[str, str]:
    """Parse debug evidence text into sections."""
    if not os.path.exists(file_path):
        return {"error": f"Debug evidence file not found: {file_path}"}

    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()
    except Exception as error:
        return {"error": f"Failed to read debug evidence: {error}"}

    system_marker = "=== SYSTEM PROMPT ==="
    user_marker = "=== USER PROMPT ==="

    system_prompt = ""
    user_prompt = text

    if system_marker in text and user_marker in text:
        try:
            _, after_system = text.split(system_marker, 1)
            system_prompt, user_prompt = after_system.split(user_marker, 1)
            system_prompt = system_prompt.strip()
            user_prompt = user_prompt.strip()
        except Exception:
            user_prompt = text

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw": text,
    }


def _is_api_followup_mode(context: AnalysisContext) -> bool:
    """True when any entry belongs to API request category."""
    return any(entry.category == "api_request" for entry in context.entries)


def _extract_ref_ids(text: str) -> list[str]:
    """Extract ordered unique REF IDs from evidence text."""
    seen: set[str] = set()
    ids: list[str] = []
    for match in _REF_ID_RE.findall(text):
        ref_id = match.strip()
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        ids.append(ref_id)
    return ids


def _debug_ref_candidates(context: AnalysisContext, terms: list[str]) -> list[EvidenceItem]:
    """Build REF-grounded evidence candidates from debug evidence prompts."""
    items: list[EvidenceItem] = []
    lowered_terms = [term.lower() for term in terms if term.strip()]

    for entry in context.entries:
        if entry.category != "api_request":
            continue

        parsed = parse_debug_evidence_file(entry.debug_evidence_file)
        if "error" in parsed:
            continue

        user_prompt = parsed.get("user_prompt", "")
        if not user_prompt.strip():
            continue

        ref_ids = _extract_ref_ids(user_prompt)
        if not ref_ids:
            continue

        for ref_id in ref_ids:
            marker = f"[{ref_id}]"
            pos = user_prompt.find(marker)
            if pos < 0:
                continue

            snippet = user_prompt[pos:pos + FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS]
            lowered_snippet = snippet.lower()
            score = 0.45
            if lowered_terms:
                for term in lowered_terms:
                    score += lowered_snippet.count(term) * 0.2
            else:
                score += 0.2

            items.append(
                EvidenceItem(
                    evidence_id=ref_id,
                    source="api_ref",
                    file_name=entry.file_name,
                    relevance=min(max(score, 0.0), 1.0),
                    anomaly_score=0.0,
                    excerpt=_preview_text(snippet),
                    raw_text=snippet,
                )
            )

    return items


def build_analysis_results_debug_markdown(context: AnalysisContext, max_chars: int = 7000) -> str:
    """Build markdown summary for debug evidence files."""
    if not context.entries:
        return "No analysis context available for debug evidence."

    lines: list[str] = ["### Debug Evidence"]

    for entry in context.entries:
        parsed = parse_debug_evidence_file(entry.debug_evidence_file)
        if "error" in parsed:
            lines.append(f"- {entry.file_name}: {parsed['error']}")
            continue

        system_prompt = parsed.get("system_prompt", "")
        user_prompt = parsed.get("user_prompt", "")

        if len(system_prompt) > max_chars:
            system_prompt = system_prompt[:max_chars] + "\n...[truncated]"
        if len(user_prompt) > max_chars:
            user_prompt = user_prompt[:max_chars] + "\n...[truncated]"

        lines.append("")
        lines.append(f"#### {entry.file_name}")
        lines.append(f"- Debug file: {entry.debug_evidence_file}")
        lines.append("- System prompt excerpt:")
        lines.append("```")
        lines.append(system_prompt or "(empty)")
        lines.append("```")
        lines.append("- User prompt + evidence excerpt:")
        lines.append("```")
        lines.append(user_prompt or "(empty)")
        lines.append("```")

    return "\n".join(lines)


def _format_numbered_window(numbered_lines: list[tuple[int, str]]) -> str:
    return "\n".join(
        f"L{line_number}: {line.rstrip()}"
        for line_number, line in numbered_lines[:FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET]
    )


def _stream_raw_log_windows(file_handle: Any, lowered_terms: list[str]) -> list[str]:
    """Build raw-log snippets without loading the full source file into memory."""
    windows: list[str] = []
    before: deque[tuple[int, str]] = deque(maxlen=2)
    pending: list[dict[str, Any]] = []
    first_lines: list[tuple[int, str]] = []
    last_lines: deque[tuple[int, str]] = deque(maxlen=FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET)

    for line_number, line in enumerate(file_handle, start=1):
        numbered_line = (line_number, line)
        if len(first_lines) < FOLLOWUP_RAW_LOG_MAX_LINES_PER_SNIPPET:
            first_lines.append(numbered_line)
        last_lines.append(numbered_line)

        next_pending: list[dict[str, Any]] = []
        for item in pending:
            item["lines"].append(numbered_line)
            item["remaining"] -= 1
            if item["remaining"] <= 0:
                windows.append(_format_numbered_window(item["lines"]))
            else:
                next_pending.append(item)
        pending = next_pending

        lowered_line = line.lower()
        has_hit = bool(lowered_terms) and any(term in lowered_line for term in lowered_terms)
        if has_hit and len(windows) + len(pending) < FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            pending.append(
                {
                    "remaining": 2,
                    "lines": [*before, numbered_line],
                }
            )

        before.append(numbered_line)

        if len(windows) >= FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            break

    for item in pending:
        if len(windows) >= FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE:
            break
        windows.append(_format_numbered_window(item["lines"]))

    if windows:
        return windows[:FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE]

    fallback_windows: list[str] = []
    if first_lines:
        fallback_windows.append(_format_numbered_window(first_lines))
    last_window = list(last_lines)
    if last_window and last_window != first_lines:
        fallback_windows.append(_format_numbered_window(last_window))
    return fallback_windows[:FOLLOWUP_RAW_LOG_MAX_SNIPPETS_PER_FILE]


def _raw_log_candidates(context: AnalysisContext, terms: list[str]) -> list[EvidenceItem]:
    """Retrieve relevant snippets from source logs."""
    items: list[EvidenceItem] = []
    counter = 1
    lowered_terms = [term.lower() for term in terms if term.strip()]

    for entry in context.entries:
        if not entry.source_path or not os.path.exists(entry.source_path):
            continue

        try:
            with open(entry.source_path, "r", encoding="utf-8", errors="replace") as file_handle:
                windows = _stream_raw_log_windows(file_handle, lowered_terms)
        except Exception:
            continue

        for window in windows:
            score = 0.3
            lowered_window = window.lower()
            for term in lowered_terms:
                score += lowered_window.count(term) * 0.2

            item_text = f"file={entry.file_name}\n{window}"
            items.append(
                EvidenceItem(
                    evidence_id=f"R{counter}",
                    source="raw_log",
                    file_name=entry.file_name,
                    relevance=min(max(score, 0.0), 1.0),
                    anomaly_score=0.0,
                    excerpt=_preview_text(item_text),
                    raw_text=item_text,
                )
            )
            counter += 1

    return items