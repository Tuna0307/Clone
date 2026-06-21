"""Reusable prompt builders for the structured server_monitoring workflow."""

from __future__ import annotations

import json
from typing import Any

from pipeline.constants import SERVER_LOG_EVENTS_TABLE
from pipeline.prompt_loader import load_fragment, load_prompt
PROMPT_REGISTRY: list[str] = [
    "server_monitoring.synthesis.system",
    "server_monitoring.synthesis.user",
    "server_monitoring.synthesis.ticket_block",
    "server_monitoring.synthesis.archetype_block",
    "server_monitoring.archetype_classification",
    "server_monitoring.onset_analysis",
    "server_monitoring.red_herring_filter",
    "server_monitoring.evidence_gathering",
    "server_monitoring.critic",
    "server_monitoring.ticket_refinement",
    "server_monitoring.ticket_refinement.evidence_block",
    "server_monitoring.followup_sql",
    "server_monitoring.followup_synthesis",
]


def _reference_duckdb_schema() -> str:
    return load_fragment("reference.duckdb_schema")


def _reference_uam5_dictionary() -> str:
    return load_fragment("reference.uam5_dictionary")


def _reference_archetype_taxonomy() -> str:
    return load_fragment("reference.archetype_taxonomy")


def _reference_sql_fence_rules() -> str:
    return load_fragment("reference.sql_fence_rules")


def _archetype_investigation_focus(archetype: str) -> str:
    try:
        return load_fragment(f"reference.archetype.{archetype}.investigation_focus")
    except KeyError:
        return "(see taxonomy)"


def _archetype_red_herrings(archetype: str) -> str:
    try:
        return load_fragment(f"reference.archetype.{archetype}.red_herrings")
    except KeyError:
        return "(see taxonomy)"


def build_server_monitoring_system_prompt(
    seed_facts: str,
    ticket_text: str | None = None,
    archetype_classification: dict[str, Any] | None = None,
) -> str:
    """Core system prompt for server monitoring analysis."""
    ticket_block = ""
    if ticket_text:
        ticket_block = load_prompt(
            "server_monitoring.synthesis.ticket_block",
            ticket_text=ticket_text[:3000],
        )

    archetype_block = ""
    if archetype_classification:
        archetype_block = load_prompt(
            "server_monitoring.synthesis.archetype_block",
            classification_json=json.dumps(archetype_classification, default=str, indent=2),
        )

    return load_prompt(
        "server_monitoring.synthesis.system",
        server_log_events_table=SERVER_LOG_EVENTS_TABLE,
        duckdb_schema=_reference_duckdb_schema(),
        uam5_dictionary=_reference_uam5_dictionary(),
        archetype_taxonomy=_reference_archetype_taxonomy(),
        ticket_block=ticket_block,
        archetype_block=archetype_block,
    )


def build_synthesis_user_message(file_name: str) -> str:
    """User message for report synthesis."""
    return load_prompt("server_monitoring.synthesis.user", file_name=file_name)


def build_archetype_classification_prompt(
    structural_signals: list[dict[str, Any]],
    pre_scores: dict[str, float],
    pre_scan_summary: str,
) -> str:
    """Prompt for LLM synthesis of scored archetype classification."""
    return load_prompt(
        "server_monitoring.archetype_classification",
        archetype_taxonomy=_reference_archetype_taxonomy(),
        structural_signals_json=json.dumps(structural_signals[:12], default=str, indent=2),
        pre_scores_json=json.dumps(pre_scores, indent=2),
        pre_scan_summary=pre_scan_summary,
    )


def build_onset_analysis_prompt(
    classification: dict[str, Any],
    structural_signals: list[dict[str, Any]],
    metric_onsets: list[dict[str, Any]],
) -> str:
    """Prompt for onset timing and symptom vs cause discrimination."""
    return load_prompt(
        "server_monitoring.onset_analysis",
        classification_json=json.dumps(classification, default=str, indent=2),
        structural_signals_json=json.dumps(structural_signals[:10], default=str, indent=2),
        metric_onsets_json=json.dumps(metric_onsets[:15], default=str, indent=2),
    )


def build_red_herring_filter_prompt(
    classification: dict[str, Any] | None,
    onset_analysis: dict[str, Any] | None,
    recurring_operations: list[dict[str, Any]],
    structural_signals: list[dict[str, Any]],
) -> str:
    """LLM pass for accurate red herring identification."""
    primary_arch = (classification or {}).get("primary", {}).get("archetype", "unknown")
    red_herring_hints = _archetype_red_herrings(primary_arch) if primary_arch != "unknown" else "(see taxonomy)"

    return load_prompt(
        "server_monitoring.red_herring_filter",
        primary_archetype=primary_arch,
        red_herring_hints=red_herring_hints,
        onset_analysis_json=(
            json.dumps(onset_analysis, default=str, indent=2)
            if onset_analysis else "(not yet available)"
        ),
        recurring_operations_json=json.dumps(recurring_operations[:8], default=str, indent=2),
        structural_signals_json=json.dumps(structural_signals[:6], default=str, indent=2),
    )


def _format_onset_window_hint(
    onset_analysis: dict[str, Any] | None,
    critical_windows: list[dict[str, Any]],
) -> str:
    deg = (onset_analysis or {}).get("degradation_start")
    if deg:
        return (
            f"**Onset anchor (mandatory):** degradation_start = `{deg}`. "
            "Scope metrics to ±3 minutes and log_events to ±10 minutes around this timestamp."
        )
    if critical_windows:
        first = critical_windows[0]
        return (
            f"**Onset anchor:** use critical window `{first.get('start_time')}` → "
            f"`{first.get('end_time')}` from the windows listed below."
        )
    return "**Onset anchor:** query MIN/MAX(timestamp) first if onset time is unknown."


def build_archetype_evidence_instruction(
    classification: dict[str, Any] | None,
    competing_hypotheses: list[dict[str, Any]],
    onset_analysis: dict[str, Any] | None,
    structural_signals: list[dict[str, Any]],
    high_volume_signals: list[dict[str, Any]],
    critical_windows: list[dict[str, Any]],
    current_phase: str = "evidence_gathering",
    *,
    turn_number: int = 1,
    max_turns: int = 3,
    prior_observations: list[str] | None = None,
) -> str:
    """Archetype-aware evidence gathering instruction."""
    primary = (classification or {}).get("primary", {})
    primary_arch = primary.get("archetype", "unknown")

    competing = "\n".join(
        f"- {h.get('archetype')} (confidence={h.get('confidence')})"
        for h in competing_hypotheses[:3]
    ) or "(none — derive from taxonomy competing_archetypes)"

    hv = "\n".join(
        f"- {s.get('signal_type')}: {s.get('snippet', '')[:120]}"
        for s in high_volume_signals[:5]
    ) or "(none)"
    struct = "\n".join(
        f"- [{s.get('signal_family')}] {s.get('summary', '')[:100]}"
        for s in structural_signals[:6]
    ) or "(none)"
    wins = "\n".join(
        f"- {w.get('label')} ({w.get('start_time')} → {w.get('end_time')})"
        for w in critical_windows[:3]
    ) or "(none identified)"

    return load_prompt(
        "server_monitoring.evidence_gathering",
        current_phase=current_phase,
        turn_line=f"**Evidence turn:** {turn_number} of {max_turns}",
        primary_archetype=primary_arch,
        primary_confidence=str(primary.get("confidence", "n/a")),
        investigation_focus=_archetype_investigation_focus(primary_arch),
        competing_hypotheses=competing,
        onset_analysis_json=(
            json.dumps(onset_analysis, default=str, indent=2)[:2000]
            if onset_analysis else "(pending)"
        ),
        structural_signals_summary=struct,
        high_volume_signals_summary=hv,
        critical_windows_summary=wins,
        onset_hint=_format_onset_window_hint(onset_analysis, critical_windows),
        duckdb_schema=_reference_duckdb_schema(),
        server_log_events_table=SERVER_LOG_EVENTS_TABLE,
        prior_observations="\n\n".join(prior_observations) if prior_observations else "(none yet this visit)",
        sql_fence_rules=_reference_sql_fence_rules(),
    )

def build_archetype_critic_prompt(
    state_summary: str,
    phases_completed: set[str],
    classification: dict[str, Any] | None,
    competing_hypotheses: list[dict[str, Any]],
    evidence_packages: dict[str, Any],
    onset_analysis: dict[str, Any] | None,
    red_herring_count: int,
) -> str:
    """Archetype-aware critic with reclassification support."""
    return load_prompt(
        "server_monitoring.critic",
        state_summary=state_summary,
        phases_completed=str(sorted(phases_completed)),
        classification_json=(
            json.dumps(classification, default=str, indent=2)
            if classification else "MISSING"
        ),
        competing_hypotheses_json=json.dumps(competing_hypotheses, default=str),
        evidence_package_keys=str(list(evidence_packages.keys()) if evidence_packages else "(none)"),
        onset_analysis_present=str(bool(onset_analysis)),
        red_herring_count=str(red_herring_count),
    )


def build_ticket_refinement_prompt(
    ticket_text: str,
    current_findings: str,
    existing_evidence: str = "",
) -> str:
    """Ticket-guided refinement prompt."""
    evidence_block = ""
    if existing_evidence.strip():
        evidence_block = load_prompt(
            "server_monitoring.ticket_refinement.evidence_block",
            existing_evidence=existing_evidence[:3000],
        )

    return load_prompt(
        "server_monitoring.ticket_refinement",
        current_findings=current_findings[:4000],
        existing_evidence_block=evidence_block,
        ticket_text=ticket_text[:2500],
    )


def build_followup_sql_instruction(
    *,
    user_query: str,
    file_name: str,
    metric_row_count: int,
    log_event_row_count: int,
    report_excerpt: str,
    original_query: str,
    start_time: str,
    end_time: str,
    ticket_excerpt: str,
    chat_history: str,
    prior_observations: list[str],
    available_files: list[str] | None = None,
    force_synthesis: bool = False,
    observation_bounds_text: str = "",
) -> str:
    """Prompt for server_monitoring follow-up SQL."""
    files_block = ""
    if available_files and len(available_files) > 1:
        files_block = (
            "\n**Files in this analysis session:**\n"
            + "\n".join(f"- {name}" for name in available_files)
            + f"\nYou are currently querying: **{file_name}**\n"
        )

    ticket_block = ""
    if ticket_excerpt.strip():
        ticket_block = f"\n**Support ticket excerpt:**\n{ticket_excerpt}\n"

    force_note = ""
    if force_synthesis:
        force_note = (
            "**IMPORTANT:** This is the final step. Do NOT emit SQL. "
            "Use the observations above and emit **FINAL_ANSWER:** only."
        )

    return load_prompt(
        "server_monitoring.followup_sql",
        file_name=file_name,
        metric_row_count=f"{metric_row_count:,}",
        log_event_row_count=f"{log_event_row_count:,}",
        duckdb_schema=_reference_duckdb_schema(),
        uam5_dictionary=_reference_uam5_dictionary(),
        files_block=files_block,
        original_query=original_query or "(not provided)",
        start_time=start_time or "unspecified",
        end_time=end_time or "unspecified",
        observation_bounds_text=observation_bounds_text,
        ticket_block=ticket_block,
        report_excerpt=report_excerpt[:12000],
        chat_history=chat_history.strip() or "(no prior chat)",
        user_query=user_query,
        prior_observations="\n\n".join(prior_observations) if prior_observations else "(none yet)",
        server_log_events_table=SERVER_LOG_EVENTS_TABLE,
        sql_fence_rules=_reference_sql_fence_rules(),
        force_synthesis_note=force_note,
    )


def build_followup_synthesis_instruction(
    *,
    user_query: str,
    report_excerpt: str,
    prior_observations: list[str],
) -> str:
    """Force a grounded final answer from accumulated SQL observations."""
    return load_prompt(
        "server_monitoring.followup_synthesis",
        user_query=user_query,
        report_excerpt=report_excerpt[:8000],
        prior_observations="\n\n".join(prior_observations) if prior_observations else "(none)",
    )