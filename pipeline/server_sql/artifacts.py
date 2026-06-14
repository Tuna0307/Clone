"""Artifact writing helpers for the structured server monitoring workflow.

These functions must produce byte-identical (or structurally equivalent) output
to the previous inline writing logic in analysis.py so that:
- Existing debug_evidence_*.txt consumers continue to work
- The .sql_trace.jsonl sidecar format remains compatible
- Follow-up retrieval and UI panels see no change

This is a critical piece of the safe migration strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifact_paths import ensure_parent_dir
# (SERVER_LOG_EVENTS_TABLE import removed — it was unused after consolidation)


def write_server_debug_artifacts(
    *,
    file_name: str,
    state: Any,                    # ServerMonitoringState (or legacy dict during transition)
    findings: str,
    refinement_performed: bool = False,
    ticket_text: str | None = None,
    system_prompt: str | None = None,
    trace_steps: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """
    Writes the two canonical artifacts for a server_monitoring analysis:

    - debug_evidence_<file>.txt   (human-readable, with full transcript)
    - debug_evidence_<file>.sql_trace.jsonl  (structured trace)

    Returns (debug_txt_path, trace_jsonl_path).

    This function is intentionally written to be a faithful port of the previous
    inline writing logic so that downstream consumers (followup, UI, manual
    inspection) see no behavioral difference.
    """
    from artifact_paths import debug_evidence_path

    debug_txt_path = debug_evidence_path(file_name)
    trace_jsonl_path = debug_txt_path.replace(".txt", ".sql_trace.jsonl")

    ensure_parent_dir(debug_txt_path)

    # --- Human-readable debug_evidence_*.txt ---
    with open(debug_txt_path, "w", encoding="utf-8") as f:
        f.write("=== SERVER MONITORING (DuckDB + Agentic SQL) ===\n")
        # Try to pull row counts from state if available (supports canonical fields + legacy names during transition)
        metric_rows = (
            getattr(state, "metric_row_count", None)
            or getattr(state, "duckdb_row_count", None)
            or (state.get("metric_row_count") if isinstance(state, dict) else None)
            or 0
        )
        log_rows = (
            getattr(state, "log_event_row_count", None)
            or getattr(state, "log_event_count", None)
            or (state.get("log_event_row_count") if isinstance(state, dict) else None)
            or 0
        )
        f.write(f"File: {file_name}\nMetric rows: {metric_rows} | Log event rows: {log_rows}\n\n")

        if ticket_text:
            f.write("=== TICKET CONTEXT (used for post-report refinement) ===\n")
            f.write(ticket_text[:2000] + ("...\n" if len(ticket_text) > 2000 else "\n"))
            f.write("\n")

        if system_prompt:
            f.write("=== SYSTEM PROMPT ===\n")
            f.write(system_prompt)
            f.write("\n\n")

        archetype = getattr(state, "archetype_classification", None)
        if archetype is not None:
            f.write("=== ARCHETYPE CLASSIFICATION ===\n")
            if hasattr(archetype, "model_dump"):
                f.write(json.dumps(archetype.model_dump(mode="json"), indent=2, default=str))
            else:
                f.write(json.dumps(archetype, indent=2, default=str))
            f.write("\n\n")

        onset = getattr(state, "onset_analysis", None)
        if onset is not None:
            f.write("=== ONSET ANALYSIS ===\n")
            if hasattr(onset, "model_dump"):
                f.write(json.dumps(onset.model_dump(mode="json"), indent=2, default=str))
            else:
                f.write(json.dumps(onset, indent=2, default=str))
            f.write("\n\n")

        # Full transcript section (the valuable new observability)
        f.write("=== AGENTIC SQL FULL CONVERSATION TRANSCRIPT (step-by-step) ===\n")
        f.write("This is the complete reasoning trace the agent actually executed.\n\n")

        steps_to_write = trace_steps or getattr(state, "trace", []) or []
        for ts in steps_to_write:
            step_id = ts.get("step", "?")
            phase = ts.get("phase", ts.get("current_phase", "unknown"))
            f.write(f"--- Step {step_id} | Phase: {phase} ---\n")

            if ts.get("llm_output"):
                f.write("LLM Output:\n")
                f.write(str(ts["llm_output"]))
                f.write("\n")

            if ts.get("sql_blocks"):
                for i, sql in enumerate(ts["sql_blocks"]):
                    f.write(f"SQL[{i}]:\n{sql}\n")

            if ts.get("observations"):
                f.write("Observations:\n")
                obs = ts["observations"]
                if isinstance(obs, list):
                    f.write("\n\n".join(str(o) for o in obs))
                else:
                    f.write(str(obs))
                f.write("\n")
            f.write("\n")

        f.write("=== END OF AGENTIC SQL TRANSCRIPT ===\n\n")

        f.write("=== FINAL LLM OUTPUT (structured report) ===\n")
        f.write(findings)

        if refinement_performed:
            f.write("\n\n[Refinement iteration with ticket context was performed after the initial report.]\n")

    # --- Structured trace sidecar ---
    try:
        with open(trace_jsonl_path, "w", encoding="utf-8") as tf:
            for ts in steps_to_write:
                tf.write(json.dumps(ts, default=str) + "\n")
    except Exception:
        # Non-fatal — the human-readable file is the primary artifact
        pass

    return debug_txt_path, trace_jsonl_path
