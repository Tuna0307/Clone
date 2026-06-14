"""Reusable prompt builders for the structured server_monitoring workflow.

Centralizes archetype-aware diagnostic protocol, 3-section contract, critic
disciplines, and UAM dictionary usage so all nodes stay consistent.
"""

from __future__ import annotations

import json
from typing import Any

from pipeline.server_metrics import DUCKDB_TABLE_SCHEMA_TEXT, UAM5_SERVER_MONITORING_DICTIONARY_TEXT
from pipeline.constants import SERVER_LOG_EVENTS_TABLE
from pipeline.server_sql.archetypes import (
    format_archetype_taxonomy_for_prompt,
    get_common_red_herrings,
    get_investigation_focus,
    IncidentArchetype,
)

_SQL_FENCE_FORMAT_RULES = """
SQL FORMAT RULES (required for automatic execution):
- Put each query in its own fenced ```sql block.
- Start each block directly with WITH or SELECT. Do NOT prefix blocks with -- or /* comment lines.
- Put brief purpose/labels outside the fence, not inside it.
- Only read-only SELECT/WITH queries are permitted.
"""


def build_server_monitoring_system_prompt(
    seed_facts: str,
    ticket_text: str | None = None,
    archetype_classification: dict[str, Any] | None = None,
) -> str:
    """Core system prompt for server monitoring analysis."""
    ticket_block = ""
    if ticket_text:
        ticket_block = (
            "\n**SUPPORT TICKET CONTEXT (user-provided symptoms):**\n"
            f"{ticket_text[:3000]}\n"
            "After producing the standard report, you will be asked to do a targeted refinement pass "
            "against this ticket. Prepare evidence that directly addresses the symptoms described.\n"
        )

    archetype_block = ""
    if archetype_classification:
        archetype_block = (
            "\n**CLASSIFIED INCIDENT ARCHETYPE (from prior diagnostic phases):**\n"
            f"{json.dumps(archetype_classification, default=str, indent=2)}\n"
        )

    taxonomy = format_archetype_taxonomy_for_prompt()

    return f"""You are a senior IAM Forensic Evidence Analyst specializing in UAM server monitoring and time-series resource behavior analysis.

You have access to a DuckDB database with these query surfaces:
1. `server_metrics` — raw EAV metric rows (`metric_name`, `metric_value` per snapshot).
2. `server_metrics_wide` — **preferred** pivoted one-row-per-snapshot view (`thread_count`, `dbcp_active_connections`, `response_time_ms`, etc.).
3. `{SERVER_LOG_EVENTS_TABLE}` — **every parsed log line** (`timestamp`, `thread`, `raw_line`) in the observation window.

{DUCKDB_TABLE_SCHEMA_TEXT}

**OFFICIAL UAM5 SERVER MONITORING DATA DICTIONARY** (use these exact metric names):
{UAM5_SERVER_MONITORING_DICTIONARY_TEXT}

{taxonomy}

**Application Event Investigation (use {SERVER_LOG_EVENTS_TABLE})**:
Search the full log text for causes that may not appear in periodic metric snapshots:
- Large result/row/return counts, method bursts, extreme latencies, rate spikes.
- Log output gaps, process-wide vs narrow endpoint patterns.
- Correlations between JVM threads, Tomcat pools, DBCP, and Hibernate sessions.

{ticket_block}{archetype_block}

**DIAGNOSTIC PROTOCOL (decision-tree — no single archetype is preferred by default)**:
1. **Classify the incident archetype** using balanced structural signals (log gaps, metric correlations, endpoint breadth, high-volume indicators, runtime stall indicators).
2. **Establish onset** — when degradation began (not just peak), abrupt vs gradual, and classify each major signal as likely cause, confirmed effect, or ambiguous.
3. **Investigate per the classified archetype(s)** while maintaining at least one competing hypothesis until evidence reasonably rejects it.
4. **Filter red herrings** — scheduled jobs, post-onset symptoms, steady-state metrics — with explicit reasoning.

You will be critiqued on balanced archetype handling, competing hypothesis testing, and symptom vs cause discrimination.

STRICT RULES:
- Always use SQL to gather evidence before drawing conclusions.
- Minimum of 3 successful SQL queries before a final report (the system will nudge you).
- Do NOT use placeholder language ("not yet evidenced", etc.). Only report facts you have verified.
- Use exact metric names from the dictionary.
- When you have sufficient evidence, output ONLY the final report using the exact headers below. No extra sections.

OUTPUT FORMAT (Markdown – follow exactly):

## 1. File-Wide Evidence Summary

### File Metadata & Observation Window
- File name, total metric snapshots, observed time range, sampling frequency.

### Time-Series Snapshot — Start / End / Extremes
- Key metrics: start, end, min, max, net delta. Highlight significant movement.

### Resource Utilization Trends
- JVM & Memory, Tomcat Thread Pool, Event & Delivery Manager, Hibernate, DBCP.

### Thread Pool & Queue Health Analysis
- eventManager, deliveryManager, Tomcat pools. Note queue buildup, rejected tasks, saturation.

### Database & Persistence Layer Activity
- Hibernate sessionCount, cache ratios, DBCP connections.

### Configuration & Environment Context
- am.serverName, connector, jvm.maxMemory, etc.

### Supporting Log Evidence (verbatim)
- For critical windows, query {SERVER_LOG_EVENTS_TABLE} for verbatim raw_line excerpts. Include timestamps.

### Potential Bottleneck & Degradation Indicators
- Threshold crossings and application-level events discovered in raw log lines.

## 2. Analysis Boundaries & Uncertainty
- What the logs do NOT contain. Granularity notes.

## 3. Time-Series Observations & Key Patterns
- Most important behavioral patterns across the window, with evidence.

### Incident Archetype Assessment
- Primary archetype and confidence.
- Secondary archetype (if relevant) and confidence.
- Strongest rejected alternative and why it was rejected.
- For mixed cases: state "mixed, with dominant archetype X".
"""


def build_archetype_classification_prompt(
    structural_signals: list[dict[str, Any]],
    pre_scores: dict[str, float],
    pre_scan_summary: str,
) -> str:
    """Prompt for LLM synthesis of scored archetype classification."""
    signals_json = json.dumps(structural_signals[:12], default=str, indent=2)
    scores_json = json.dumps(pre_scores, indent=2)
    taxonomy = format_archetype_taxonomy_for_prompt()

    return f"""You are performing **broad diagnostic and archetype classification** for a UAM server slowness incident.

{taxonomy}

**Deterministic structural signals (from balanced SQL pre-screening):**
{signals_json}

**Deterministic archetype pre-scores (0–1, anchors only — you may adjust):**
{scores_json}

**Application outlier pre-scan hits:**
{pre_scan_summary}

Synthesize a **scored multi-label classification**. Do not default to high_volume_cardinality unless signals support it.
Consider global_runtime_stall when log gaps or metric-without-log divergence are present.
Consider mixed_compound when two archetypes have comparable support.

Reply with ONLY a JSON object (no markdown outside the JSON block):
```json
{{
  "primary": {{
    "archetype": "<one of: global_runtime_stall | high_volume_cardinality | thread_pool_pressure | db_connection_pressure | mixed_compound>",
    "confidence": 0.0,
    "supporting_signals": ["signal_id or summary", "..."]
  }},
  "secondary": {{
    "archetype": "...",
    "confidence": 0.0,
    "supporting_signals": ["..."]
  }},
  "rejected_hypotheses": [
    {{
      "archetype": "...",
      "confidence": 0.0,
      "supporting_signals": [],
      "rejection_reason": "why this archetype is unlikely"
    }}
  ],
  "rationale": "brief synthesis of supporting signals and rejected alternatives"
}}
```

Set secondary to null if no relevant secondary archetype. Include at least one rejected hypothesis.
"""


def build_onset_analysis_prompt(
    classification: dict[str, Any],
    structural_signals: list[dict[str, Any]],
    metric_onsets: list[dict[str, Any]],
) -> str:
    """Prompt for onset timing and symptom vs cause discrimination."""
    return f"""You are performing **onset analysis and symptom discrimination**.

**Archetype classification:**
{json.dumps(classification, default=str, indent=2)}

**Structural signals:**
{json.dumps(structural_signals[:10], default=str, indent=2)}

**Deterministic metric onset anchors (first threshold crossings):**
{json.dumps(metric_onsets[:15], default=str, indent=2)}

Answer:
- When did meaningful degradation begin (not just peak)?
- Was onset abrupt or gradual?
- For each major signal (connection pool saturation, thread growth, response time, log bursts, log gaps), classify as likely_cause, confirmed_effect, or ambiguous.

Reply with ONLY a JSON object:
```json
{{
  "degradation_start": "ISO timestamp or null",
  "onset_shape_overall": "abrupt | gradual | unknown",
  "signal_records": [
    {{
      "signal_name": "e.g. dbcp.ActiveConnections saturation",
      "onset_time": "ISO timestamp or null",
      "onset_shape": "abrupt | gradual | unknown",
      "role": "likely_cause | confirmed_effect | ambiguous",
      "evidence": ["brief fact from signals or metrics"]
    }}
  ]
}}
```
"""


def build_red_herring_filter_prompt(
    classification: dict[str, Any] | None,
    onset_analysis: dict[str, Any] | None,
    recurring_operations: list[dict[str, Any]],
    structural_signals: list[dict[str, Any]],
) -> str:
    """LLM pass for accurate red herring identification (always run after deterministic checks)."""
    primary_arch = (classification or {}).get("primary", {}).get("archetype", "unknown")
    red_herring_hints = ""
    if primary_arch and primary_arch != "unknown":
        try:
            red_herring_hints = "\n".join(f"- {h}" for h in get_common_red_herrings(primary_arch))
        except (KeyError, TypeError):
            pass

    return f"""You are performing **red herring filtering** for a UAM server slowness incident.

**Primary archetype:** {primary_arch}

**Common red herrings for this archetype:**
{red_herring_hints or "(see taxonomy)"}

**Onset analysis:**
{json.dumps(onset_analysis, default=str, indent=2) if onset_analysis else "(not yet available)"}

**Deterministic recurring-operation candidates (fixed cadence):**
{json.dumps(recurring_operations[:8], default=str, indent=2)}

**Structural signals for context:**
{json.dumps(structural_signals[:6], default=str, indent=2)}

Identify signals that are NOT causal root causes:
- cadence_scheduled: recurring at fixed intervals regardless of incident
- post_onset_symptom: appears only after degradation_start
- steady_state: constant cost outside the incident window
- background_polling, low_impact, out_of_scope

Reply with ONLY a JSON array of rejections:
```json
[
  {{
    "signal_description": "what was observed",
    "rejection_category": "cadence_scheduled | post_onset_symptom | steady_state | out_of_scope | low_impact | background_polling | other",
    "rejection_reason": "why not causal",
    "evidence": ["supporting fact"],
    "confidence": "CERTAIN | STRONG | INFERRED | WEAK"
  }}
]
```

Be conservative: only reject when evidence supports it. Return [] if no red herrings are confident.
"""


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
    focus = []
    if primary_arch and primary_arch != "unknown":
        try:
            focus = get_investigation_focus(primary_arch)
        except KeyError:
            pass

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
    onset_hint = _format_onset_window_hint(onset_analysis, critical_windows)
    obs_block = "\n\n".join(prior_observations) if prior_observations else "(none yet this visit)"
    turn_line = f"**Evidence turn:** {turn_number} of {max_turns}"

    return f"""You are in the **{current_phase}** phase of the archetype-aware server monitoring workflow.
{turn_line}

**Primary archetype:** {primary_arch} (confidence={primary.get('confidence', 'n/a')})
**Investigation focus for primary:**
{chr(10).join(f'- {f}' for f in focus) or '(see taxonomy)'}

**Competing hypotheses to test (mandatory — at least 1 SQL query must target one):**
{competing}

**Onset analysis:**
{json.dumps(onset_analysis, default=str, indent=2)[:2000] if onset_analysis else "(pending)"}

**Structural signals:**
{struct}

**Application outlier signals:**
{hv}

**Critical windows:**
{wins}

{onset_hint}

{DUCKDB_TABLE_SCHEMA_TEXT}

Drive **targeted SQL exploration** on `server_metrics_wide` + `{SERVER_LOG_EVENTS_TABLE}`:
- **Mandatory:** at least one query MUST hit `{SERVER_LOG_EVENTS_TABLE}` for verbatim `raw_line` proof (REST traces, lapse(ms), DB/LDAP/Hibernate keywords).
- **Mandatory:** at least one query MUST hit `server_metrics_wide` around the onset anchor (not the whole file).
- On `{SERVER_LOG_EVENTS_TABLE}`, filter with `timestamp` (not `ts`).
- For metrics, avoid sparse NaN rows: filter `WHERE thread_count IS NOT NULL` or aggregate with `time_bucket(INTERVAL 1 MINUTE, timestamp)`.
- When parsing numbers from `raw_line` with `regexp_extract`, use `TRY_CAST(... AS BIGINT)` (never plain `CAST`) because non-matching lines return empty strings.
- Run at least one query that could confirm or reject the top competing hypothesis.
- Respect onset timing: distinguish causes from post-onset symptoms.

**SQL observations from prior turns this visit:**
{obs_block}

Example dense metric query (preferred over raw sparse timelines):
```sql
SELECT
  time_bucket(INTERVAL 1 MINUTE, timestamp) AS minute,
  MAX(thread_count) AS max_threads,
  MAX(dbcp_active_connections) AS max_dbcp,
  MAX(response_time_ms) AS max_response_ms
FROM server_metrics_wide
WHERE timestamp BETWEEN TIMESTAMP '2026-02-25 15:50:00' AND TIMESTAMP '2026-02-25 15:54:00'
  AND thread_count IS NOT NULL
GROUP BY 1
ORDER BY 1;
```

Example log evidence query:
```sql
SELECT timestamp, raw_line
FROM {SERVER_LOG_EVENTS_TABLE}
WHERE timestamp BETWEEN TIMESTAMP '2026-02-25 15:51:00' AND TIMESTAMP '2026-02-25 15:53:00'
  AND (raw_line LIKE '%lapse(ms)%' OR raw_line LIKE '%REST:%')
ORDER BY timestamp
LIMIT 15;
```

Output 1–3 safe, read-only SQL queries in fenced blocks.
{_SQL_FENCE_FORMAT_RULES}
When you have enough evidence, say: "READY FOR SYNTHESIS" (no SQL in that turn).
On later turns, refine based on the SQL observations above — do not repeat failed or empty query shapes.
"""


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
    return f"""CRITIC REVIEW — archetype-aware server monitoring analysis.

Current state summary:
{state_summary}

Phases completed: {sorted(phases_completed)}
Archetype classification: {json.dumps(classification, default=str, indent=2) if classification else "MISSING"}
Competing hypotheses: {json.dumps(competing_hypotheses, default=str)}
Evidence packages: {list(evidence_packages.keys()) if evidence_packages else "(none)"}
Onset analysis present: {bool(onset_analysis)}
Red herrings rejected: {red_herring_count}

Evaluate against these disciplines:
1. Were broad diagnostic + onset analysis completed before deep archetype-specific dives?
2. Does gathered evidence match the primary archetype proportionally?
3. Was at least one competing hypothesis tested with actual SQL results?
4. Are symptom vs cause distinctions respected (e.g. pool saturation under global stall)?
5. Were red herrings explicitly identified where applicable?
6. **Archetype mismatch**: if evidence strongly supports a different archetype than classified, say RECLASSIFY.

Reply with one of:
- PASS: Ready for synthesis.
- FAIL: <violations>. Must gather more evidence on <X>.
- RETRY: Minor issues; synthesis can proceed carefully.
- RECLASSIFY: <reason>. Evidence supports <archetype> over current primary <current>.

Be evidence-based. Do not require high-volume/Count evidence unless that is the classified archetype.
"""


def build_critic_prompt(
    state_summary: str,
    phases_completed: set[str],
    high_volume_signal_count: int,
    red_herring_count: int,
) -> str:
    """Legacy critic wrapper — prefer build_archetype_critic_prompt."""
    return build_archetype_critic_prompt(
        state_summary=state_summary,
        phases_completed=phases_completed,
        classification=None,
        competing_hypotheses=[],
        evidence_packages={},
        onset_analysis=None,
        red_herring_count=red_herring_count,
    )


def build_evidence_gathering_instruction(
    current_phase: str,
    high_volume_signals: list[dict],
    critical_windows: list[dict],
) -> str:
    """Legacy evidence instruction — prefer build_archetype_evidence_instruction."""
    return build_archetype_evidence_instruction(
        classification=None,
        competing_hypotheses=[],
        onset_analysis=None,
        structural_signals=[],
        high_volume_signals=high_volume_signals,
        critical_windows=critical_windows,
        current_phase=current_phase,
    )


def build_ticket_refinement_prompt(ticket_text: str, current_findings: str) -> str:
    """Prompt for the post-report ticket-guided refinement pass."""
    return f"""You previously produced this server monitoring report:

{current_findings[:4000]}

The user has now supplied this support ticket describing the observed symptoms:

{ticket_text[:2500]}

Perform a **targeted refinement pass**:
- Re-examine the DuckDB data (especially around critical windows already identified).
- Run any additional SQL or raw_line queries needed to surface the root cause matching ticket symptoms.
- Output a concise "Ticket-Guided Root Cause Addendum" citing specific raw log lines or metric values.
- Reference the incident archetype assessment if present in the report.

Only emit the addendum (no need to repeat the full original report).
"""


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
    """Prompt for server_monitoring follow-up: answer a specific user question via SQL."""
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

    history_block = chat_history.strip() or "(no prior chat)"
    obs_block = "\n\n".join(prior_observations) if prior_observations else "(none yet)"

    return f"""You are a senior IAM forensic analyst answering a **follow-up question** about a completed UAM server monitoring investigation.

You have a live DuckDB database for file `{file_name}` with:
- `server_metrics` — numeric monitoring snapshots ({metric_row_count:,} rows)
- `{SERVER_LOG_EVENTS_TABLE}` — every parsed log line ({log_event_row_count:,} rows)

{DUCKDB_TABLE_SCHEMA_TEXT}

**UAM5 metric dictionary (use exact metric names):**
{UAM5_SERVER_MONITORING_DICTIONARY_TEXT}

{files_block}
**Original analysis query:** {original_query or "(not provided)"}
**Analysis time window (UI filter):** {start_time or "unspecified"} to {end_time or "unspecified"}
{observation_bounds_text}
{ticket_block}
**Incident report excerpt:**
{report_excerpt[:12000]}

**Recent chat:**
{history_block}

**Follow-up question:** {user_query}

**SQL observations already gathered this turn:**
{obs_block}

TASK:
- Answer the follow-up using SQL evidence from `server_metrics` and `{SERVER_LOG_EVENTS_TABLE}`.
- Typical questions: timeframes, affected users/accounts, thread saturation windows, endpoint bursts, metric correlations.
- If the user references **"cause 1"**, **"cause 2"**, **"primary cause"**, or similar numbered causes, map that label to the corresponding cause/root-cause bullet or pattern described in the incident report excerpt above, then query DuckDB for the precise timestamps/windows for that cause.
- Use the **Loaded DuckDB observation bounds** for calendar dates. Ticket clock times (e.g. 15:59:01) must use the same date as the loaded log data, not today's date and not a placeholder year.
- Use exact metric names from the dictionary.
- Cite concrete timestamps and `raw_line` snippets from query results. Do not invent facts.
- Prefer answering after 1-3 targeted SQL queries when the question is narrow (e.g. timeframes for one cause).

EXECUTION RULES (critical):
- The system **automatically executes** any SQL you emit. The user cannot run SQL.
- NEVER ask the user to run queries. NEVER say "run these queries", "I'll give you the timeframes after...", or similar.
- ALWAYS wrap SQL in fenced ```sql blocks. Unfenced SQL will not run.
{_SQL_FENCE_FORMAT_RULES}
- After SQL results appear in observations, emit **FINAL_ANSWER:** with the direct answer (timeframes, counts, etc.).

OUTPUT CONTRACT (choose one):
1. If you need more data, emit up to **2** fenced ```sql blocks with read-only SELECT/WITH queries only.
2. If you have enough evidence, emit **FINAL_ANSWER:** followed by concise conversational markdown grounded in the SQL results.

Do not output both new SQL and FINAL_ANSWER in the same response.
{f"**IMPORTANT:** This is the final step. Do NOT emit SQL. Use the observations above and emit **FINAL_ANSWER:** only." if force_synthesis else ""}
"""


def build_followup_synthesis_instruction(
    *,
    user_query: str,
    report_excerpt: str,
    prior_observations: list[str],
) -> str:
    """Force a grounded final answer from accumulated SQL observations."""
    obs_block = "\n\n".join(prior_observations) if prior_observations else "(none)"
    return f"""You are finishing a server monitoring follow-up answer.

**Follow-up question:** {user_query}

**Incident report context:**
{report_excerpt[:8000]}

**SQL observations gathered:**
{obs_block}

Emit **FINAL_ANSWER:** followed by concise markdown.
- Ground every timeframe in the SQL observations or report context.
- If the user asked about "cause 1" / a numbered cause, name which cause you mapped it to.
- Do not emit SQL.
"""