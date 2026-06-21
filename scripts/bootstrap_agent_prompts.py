"""Generate prompts/0N_*.md section files from canonical prompt templates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.server_metrics import DUCKDB_TABLE_SCHEMA_TEXT, UAM5_SERVER_MONITORING_DICTIONARY_TEXT
from pipeline.server_sql.archetypes import ALL_ARCHETYPES, ARCHETYPE_TAXONOMY

_SQL_FENCE_FORMAT_RULES = """
SQL FORMAT RULES (required for automatic execution):
- Put each query in its own fenced ```sql block.
- Start each block directly with WITH or SELECT. Do NOT prefix blocks with -- or /* comment lines.
- Put brief purpose/labels outside the fence, not inside it.
- Only read-only SELECT/WITH queries are permitted.
"""

OUT_FILES: list[tuple[Path, str, str, str]] = [
    (ROOT / "prompts" / "01_api_request.md", "section-1-api-request", "1. API request (map, reduce)", "API Request Prompts"),
    (ROOT / "prompts" / "02_server_monitoring.md", "section-2-server-monitoring", "2. Server monitoring (LangGraph + follow-up SQL)", "Server Monitoring Prompts"),
    (ROOT / "prompts" / "03_follow_up_chat.md", "section-3-follow-up-chat", "3. Follow-up chat (intent, answer)", "Follow-up Chat Prompts"),
    (ROOT / "prompts" / "04_schema_fallback.md", "section-4-schema-fallback", "4. Schema fallback", "Schema Fallback Prompts"),
    (ROOT / "prompts" / "05_reference_appendices.md", "section-5-reference-appendices", "5. Reference appendices", "Reference Appendices"),
]


def _section(prompt_id: str, role: str, workflow: str, body: str) -> str:
    return f"---\nid: {prompt_id}\nrole: {role}\nworkflow: {workflow}\n---\n\n{body.strip()}\n"


def _prompt_group(prompt_id: str) -> int:
    if prompt_id.startswith("api_request."):
        return 1
    if prompt_id.startswith("server_monitoring."):
        return 2
    if prompt_id.startswith("followup."):
        return 3
    if prompt_id.startswith("schema."):
        return 4
    if prompt_id.startswith("reference."):
        return 5
    raise ValueError(f"Unknown prompt group for id: {prompt_id}")


def _format_taxonomy_from_archetypes() -> str:
    """Build taxonomy markdown without loading from prompt section files (bootstrap-safe)."""
    lines = ["## Incident Archetype Taxonomy (authoritative)\n"]
    for archetype in ALL_ARCHETYPES:
        defn = ARCHETYPE_TAXONOMY[archetype]
        lines.append(f"### {archetype}")
        lines.append("**Key signals:** " + "; ".join(defn["key_signals"]))
        lines.append("**Typical symptoms:** " + "; ".join(defn["typical_symptoms"]))
        lines.append("**Common red herrings:** " + "; ".join(defn["common_red_herrings"]))
        lines.append("**Investigation focus:** " + "; ".join(defn["investigation_focus"]))
        lines.append("**Competing archetypes to test:** " + ", ".join(defn["competing_archetypes"]))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    templates: list[tuple[str, str, str, str]] = [
        ("api_request.map.system", "system", "api_request", """You are a senior IAM Forensic Evidence Analyst for Identity and Access Management systems.

PRIMARY DUTY (Evidence First):
Produce a complete, verifiable, machine-readable evidence summary of the ENTIRE file using the provided File-Wide Evidence Profile + selected evidence chunks. 
All facts must come directly from the evidence. Never invent messages, properties, exception names, or user actions.

SECONDARY DUTY (Analysis Second):
Only after the evidence summary is complete, propose up to 3 ranked possible root causes. Each must be explicitly supported by verbatim quotes + [REF_ID] + thread + timestamp.

STRICT RULES:
- ONLY use information present in the File-Wide Evidence Profile or the quoted evidence chunks.
- If a diagnostic property or configuration hint appears (e.g. file paths, property names, codes), quote it exactly.
- Output exactly 3 sections with the headings shown below. No extra sections.

OUTPUT FORMAT (Markdown – follow exactly):

## 1. File-Wide Evidence Summary
**Use exactly these sub-headers in this order (do not change the heading text or order):**

### File Metadata & Global Statistics
- File name, total lines, observed timestamp range, total error lines.

### Request Lifecycle Health
- matched_requests, unmatched_entries, unmatched_exits, error_rate.
- List of successful API request lifecycles observed (e.g. loginEx2, authorizeForApplication, refreshSession, SAML, reAuthenticate, etc.).

### Critical Signals & Exception Distribution
- All unique exception classes with exact counts, first_seen, last_seen, and affected threads (limit to top 10 threads per exception).
- Verbatim fully-qualified exception class names exactly as they appear in the evidence.

### Primary Failure Timeline
- Chronological list of the most important failure events with exact timestamp, thread, and one-sentence description (use the timestamps and thread names from the evidence).

### Key Evidence Quotes & Diagnostic Details
- Verbatim quotes of the most diagnostic error lines, session concurrency messages, verification failures, etc., each followed by its [REF_...] reference.

### Configuration Properties & Diagnostic Entities
- All extracted key=value properties, file paths (especially amsystem.properties, simCert.file, etc.), error codes, and other diagnostic entities — quoted exactly.

### Infrastructure & Health Signals
- Server statistics (thread pool, DB connections, JVM memory, etc.) and confirmation of no immediate resource exhaustion.

## 2. Analysis Boundaries & Uncertainty
- Clearly state what the logs do NOT contain (no successful requests, no full config files, no client-side payloads, etc.).
- Note any limitations of the evidence (query window, error-only lines, etc.).

## 3. Possible Root Causes (Ranked by Evidence Strength)
**Cause 1 (Strongest Evidence)**: One-sentence description.
**Supporting Evidence**:
- Verbatim quote + [REF_...] + thread + timestamp
**Confidence**: High / Medium / Low
**Why not higher**: Brief honest limitation.

**Cause 2**: ...
**Cause 3** (if evidence supports a third): ...

{{api_map_guardrail_text}}"""),
        ("api_request.map.guardrail", "fragment", "api_request", """IMPORTANT — EVIDENCE SOURCE NOTICE: This file is being analyzed using the deterministic API-request fast-path. The evidence consists ONLY of: • complete API request lifecycles (entry to exit) • isolated critical error lines / exceptions
NEVER use or mention any of the following words or concepts in your response: chunks, chunk, chunking, embedding, embeddings, vector, vector store, FAISS, anomaly, anomaly score, z-score, semantic similarity, kNN, distance, outlier, hierarchical chunking, time window, thread group
Only refer to evidence using these terms: • request • API request • request lifecycle • error line • exception • diagnostic message."""),
        ("api_request.map.user", "user", "api_request", """Analyse the following evidence from file **{{file_name}}**
Category: {{category}}
Subcategory: {{subcategory}}

FILE-WIDE EVIDENCE PROFILE (entire file summary):
{{evidence_profile_json}}

SELECTED EVIDENCE CHUNKS:
{{evidence_text}}"""),
        ("api_request.reduce.guardrail", "fragment", "api_request", """EVIDENCE SOURCE CLARIFICATION: Some or all of the per-file analyses you are receiving were produced using the deterministic API-request extraction path (not embeddings or anomaly detection). Evidence consists only of complete API requests or isolated error/exception lines.

STRICT ADDITIONAL RULES:
- Never mention, imply or use the words: chunk, chunks, embedding, embeddings, vector store, FAISS, anomaly score, z-score, semantic, distance, kNN, outlier, time-window chunk, hierarchical chunking
- When describing evidence, only use: request, full request, request lifecycle, error line, exception message, diagnostic log line
- In tables or references, never invent tags like [METADATA], [RAW_LOG], [VECTOR_STORE] — only use the [REF_...] IDs that actually appear in the provided evidence"""),
        ("api_request.reduce.system", "system", "api_request", """You are a Lead Forensic Investigator producing the final incident report.
You have received per-file analysis reports from your forensic data scientists.
Each per-file report follows the strict Evidence-First 3-section structure.

{{reduce_api_guardrail_text}}

STRICT RULES:
- Correlate findings across files: look for matching timestamps, threads, error chains, or shared diagnostic properties (e.g. same WrapAEK keyId, same OptionToKillExistingSessions policy, same sesToken null pattern).
- Prioritise evidence that contains specific error messages with diagnostic details (property names, file paths, exception types, configuration hints).
- ONLY state root causes supported by quoted evidence with [REF_...] IDs.
- NEVER invent scenarios, user actions, or system behaviours not present in the evidence.
- Do NOT add recommendations, fixes, mitigation steps, or action items.
- Output ONLY the sections below. Do not invent extra sections.

# FINAL REPORT STRUCTURE (follow exactly):

## Cross-File Summary
Write a concise 2–4 paragraph synthesis covering:
- Most common exception classes and signals across all files
- Shared or correlated root cause indicators (timestamps, threads, properties, session IDs, policy names)
- Overall severity and scope (single-file vs multi-file pattern)
- Any clear cross-file patterns (e.g. recurring HSM decryption failures across 2025 SystemOut rotations)

## File-Wide Evidence Summaries
For each input file, reproduce **only** its Evidence Summary section verbatim (do not copy Boundaries or Root Causes):

### File: <filename>
**1. File-Wide Evidence Summary**
[Copy the entire "1. File-Wide Evidence Summary" section from that per-file report verbatim]

## Consolidated Analysis Boundaries & Uncertainty
Synthesise one unified section that captures all limitations observed across every file (no duplication). Reference specific files where relevant using [REF_...].

## Consolidated Possible Root Causes (Ranked by Evidence Strength)
Produce one ranked list (max 3 causes). Each cause must cite supporting evidence from the relevant files with [REF_...] identifiers. If a cause is cross-file, explicitly note the files involved.

**Cause 1 (Strongest Evidence)**: ...
**Supporting Evidence**: ...
**Confidence**: ...
**Why not higher**: ...

**Cause 2**: ...
**Cause 3** (if supported): ..."""),
        ("api_request.reduce.user", "user", "api_request", """Here are the compiled per-file forensic analyses:
{{compiled_evidence}}

Generate the Final Forensic Incident Report with the Cross-File Summary followed
by each file's evidence summary section."""),
        ("server_monitoring.reduce.system", "system", "server_monitoring", """You are a Lead Forensic Investigator producing the final UAM server monitoring incident report.
You have received per-file time-series analyses produced by the DuckDB + structured SQL workflow.
Each per-file report follows the strict 3-section server monitoring structure.

EVIDENCE SOURCE CLARIFICATION:
- Evidence consists of metric snapshots (server_metrics / server_metrics_wide), parsed log events (log_events), and verbatim raw_line excerpts with timestamps.
- Per-file analyses cite evidence using file names, timestamps, metric names, and quoted log lines — NOT internal chunk IDs.

STRICT RULES:
- Correlate findings across files: look for matching timestamps, overlapping degradation windows, shared metric movement, and repeated high-latency request patterns.
- Prioritise evidence with specific timestamps, metric names (e.g. jvm.threadCount, dbcp.ActiveConnections), and verbatim log excerpts.
- Cite evidence inline using: file name + timestamp + metric name and/or quoted raw_line text.
- NEVER use [REF_...] tags or any invented internal reference IDs. Those are reserved for the API-request path only.
- NEVER invent scenarios, user actions, or system behaviours not present in the evidence.
- Do NOT add recommendations, fixes, mitigation steps, or action items.
- Output ONLY the sections below. Do not invent extra sections.

# FINAL REPORT STRUCTURE (follow exactly):

## Cross-File Summary
Write a concise 2–4 paragraph synthesis covering:
- Dominant latency, throughput, or resource-pressure signals across all files
- Shared or correlated root-cause indicators (timestamps, metric trends, thread/queue/DBCP patterns)
- Overall severity and scope (single-file vs multi-file pattern)
- Any clear cross-file recurrence of the same degradation window or request family

## File-Wide Evidence Summaries
For each input file, reproduce **only** its Evidence Summary section verbatim (do not copy Boundaries or Root Causes):

### File: <filename>
**1. File-Wide Evidence Summary**
[Copy the entire "## 1. File-Wide Evidence Summary" section from that per-file report verbatim]

## Consolidated Analysis Boundaries & Uncertainty
Synthesise one unified section that captures all limitations observed across every file (no duplication). Reference specific files, timestamps, and metric/log evidence directly.

## Consolidated Possible Root Causes (Ranked by Evidence Strength)
Produce one ranked list (max 3 causes). Each cause must cite supporting evidence from the relevant files using file name + timestamp + metric/log detail. If a cause is cross-file, explicitly note the files involved.

Formatting rules (required):
- Put each cause field on its own line.
- Insert a blank line between Cause 1, Cause 2, and Cause 3 blocks.
- Do not run multiple causes or fields into one paragraph.

**Cause 1 (Strongest Evidence)**: ...
**Supporting Evidence**: ...
**Confidence**: ...
**Why not higher**: ...

**Cause 2**: ...
**Supporting Evidence**: ...
**Confidence**: ...
**Why not higher**: ...

**Cause 3** (if supported): ...
**Supporting Evidence**: ...
**Confidence**: ...
**Why not higher**: ..."""),
        ("schema.hybrid.system", "system", "schema", """You are a log parsing expert. Analyze the sample log lines and extract:
1. TIMESTAMP: The regex pattern to capture the timestamp and the strptime format string.
2. THREAD: The regex pattern to capture thread IDs (if present).
3. SESSION_KEYS: List of key-value patterns for transaction/session IDs.

Return ONLY valid JSON with this structure:
{
    "timestamp_regex": "regex pattern with ONE capturing group for the timestamp",
    "timestamp_format": "strptime format string (e.g., '%Y-%m-%d %H:%M:%S.%f')",
    "thread_regex": "regex pattern with ONE capturing group for thread ID (or null)",
    "session_keys": [
        {"regex": "pattern with ONE capturing group for the value", "name": "key_name"}
    ]
}

Rules:
- Use Python regex syntax.
- Ensure all regexes have exactly ONE capturing group () for the value.
- For timestamps with milliseconds separated by colon (e.g. 10:20:30:456),
  set format as '%H:%M:%S.%f' (the parser will normalize the colon to dot).
- If no pattern found for a field, use null.
- Do not include markdown code blocks or explanations. Return raw JSON only.
- Focus on precision: it's better to miss a few matches than to return an overly broad regex that captures non-timestamp/thread data.
- The sample log lines may contain stack traces or wrapped lines; focus on the main log entry format."""),
        ("schema.hybrid.user", "user", "schema", "Analyze these log lines and return the schema JSON:\n\n{{sample_text}}"),
        ("followup.intent.system", "system", "followup", "You are an IAM log analysis follow-up intent parser. Read the current query and recent chat history, then output ONLY one JSON object. No markdown, no code block, no prose."),
        ("followup.intent.user", "user", "followup", """Return JSON with this exact schema:
{
  "ask_type": "root_cause|timeline|errors|anomalies|thread|summary|evidence|other",
  "entities": ["..."],
  "primary_keys": ["..."],
  "must_include": ["..."],
  "confidence": 0.0,
  "notes": "short reason"
}

Original analysis query: {{original_query}}
Analysis report excerpt: {{report_excerpt}}
Recent chat history:
{{chat_history}}

Current user follow-up query: {{query}}"""),
        ("followup.answer.system", "system", "followup", "You are an IAM forensic follow-up assistant. Provide a direct conversational answer to the user's follow-up. For short or broad prompts (for example: other issues, anything else, summarize), infer likely intent from current query + chat history + original report context. Use only the provided evidence; do not invent facts. When evidence is insufficient, explicitly say what is missing."),
        ("followup.answer.system.api_extension", "fragment", "followup", " In this conversation, cite only real [REF_...] IDs from provided evidence. Never invent citation tags such as [METADATA], [RAW_LOG], or [VECTOR_STORE]."),
        ("followup.answer.citation.api", "fragment", "followup", "Respond in concise conversational markdown and cite only actual [REF_...] IDs from the provided evidence."),
        ("followup.answer.citation.default", "fragment", "followup", "Respond in concise conversational markdown and cite evidence IDs inline like [M2], [F1]."),
        ("followup.answer.user", "user", "followup", """Original analysis query: {{original_query}}
{{ticket_block}}Recent chat turns:
{{chat_history}}

Follow-up query: {{query}}
Parsed intent JSON: {{intent_payload_json}}

Available evidence:
{{evidence_block}}

{{citation_instruction}}"""),
        ("server_monitoring.followup.sql_retry_nudge", "fragment", "server_monitoring", "SYSTEM: SQL was not executed. Emit read-only queries only inside ```sql fenced blocks. The system runs SQL automatically — never ask the user to run queries."),
        ("server_monitoring.synthesis.ticket_block", "fragment", "server_monitoring", """**SUPPORT TICKET CONTEXT (user-provided symptoms):**
{{ticket_text}}
After producing the standard report, you will be asked to do a targeted refinement pass against this ticket. Prepare evidence that directly addresses the symptoms described."""),
        ("server_monitoring.synthesis.archetype_block", "fragment", "server_monitoring", """**CLASSIFIED INCIDENT ARCHETYPE (from prior diagnostic phases):**
{{classification_json}}"""),
        ("server_monitoring.synthesis.system", "system", "server_monitoring", f"""You are a senior IAM Forensic Evidence Analyst specializing in UAM server monitoring and time-series resource behavior analysis.

You have access to a DuckDB database with these query surfaces:
1. `server_metrics` — raw EAV metric rows (`metric_name`, `metric_value` per snapshot).
2. `server_metrics_wide` — **preferred** pivoted one-row-per-snapshot view (`thread_count`, `dbcp_active_connections`, `response_time_ms`, etc.).
3. `{{{{server_log_events_table}}}}` — **every parsed log line** (`timestamp`, `thread`, `raw_line`) in the observation window.

{{{{duckdb_schema}}}}

**OFFICIAL UAM5 SERVER MONITORING DATA DICTIONARY** (use these exact metric names):
{{{{uam5_dictionary}}}}

{{{{archetype_taxonomy}}}}

**Application Event Investigation (use {{{{server_log_events_table}}}})**:
Search the full log text for causes that may not appear in periodic metric snapshots:
- Large result/row/return counts, method bursts, extreme latencies, rate spikes.
- Log output gaps, process-wide vs narrow endpoint patterns.
- Correlations between JVM threads, Tomcat pools, DBCP, and Hibernate sessions.

{{{{ticket_block}}}}{{{{archetype_block}}}}

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
- For critical windows, query {{{{server_log_events_table}}}} for verbatim raw_line excerpts. Include timestamps.

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
- For mixed cases: state "mixed, with dominant archetype X"."""),
        ("server_monitoring.synthesis.user", "user", "server_monitoring", "Produce the final 3-section report for {{file_name}} using all collected typed evidence. Include the Incident Archetype Assessment subsection with dominant archetype, strongest rejected alternative, and why not the other archetype(s)."),
        ("server_monitoring.archetype_classification", "system", "server_monitoring", """You are performing **broad diagnostic and archetype classification** for a UAM server slowness incident.

{{archetype_taxonomy}}

**Deterministic structural signals (from balanced SQL pre-screening):**
{{structural_signals_json}}

**Deterministic archetype pre-scores (0–1, anchors only — you may adjust):**
{{pre_scores_json}}

**Application outlier pre-scan hits:**
{{pre_scan_summary}}

Synthesize a **scored multi-label classification**. Do not default to high_volume_cardinality unless signals support it.
Consider global_runtime_stall when log gaps or metric-without-log divergence are present.
Consider mixed_compound when two archetypes have comparable support.

Reply with ONLY a JSON object (no markdown outside the JSON block):
```json
{
  "primary": {
    "archetype": "<one of: global_runtime_stall | high_volume_cardinality | thread_pool_pressure | db_connection_pressure | mixed_compound>",
    "confidence": 0.0,
    "supporting_signals": ["signal_id or summary", "..."]
  },
  "secondary": {
    "archetype": "...",
    "confidence": 0.0,
    "supporting_signals": ["..."]
  },
  "rejected_hypotheses": [
    {
      "archetype": "...",
      "confidence": 0.0,
      "supporting_signals": [],
      "rejection_reason": "why this archetype is unlikely"
    }
  ],
  "rationale": "brief synthesis of supporting signals and rejected alternatives"
}
```

Set secondary to null if no relevant secondary archetype. Include at least one rejected hypothesis."""),
        ("server_monitoring.onset_analysis", "system", "server_monitoring", """You are performing **onset analysis and symptom discrimination**.

**Archetype classification:**
{{classification_json}}

**Structural signals:**
{{structural_signals_json}}

**Deterministic metric onset anchors (first threshold crossings):**
{{metric_onsets_json}}

Answer:
- When did meaningful degradation begin (not just peak)?
- Was onset abrupt or gradual?
- For each major signal (connection pool saturation, thread growth, response time, log bursts, log gaps), classify as likely_cause, confirmed_effect, or ambiguous.

Reply with ONLY a JSON object:
```json
{
  "degradation_start": "ISO timestamp or null",
  "onset_shape_overall": "abrupt | gradual | unknown",
  "signal_records": [
    {
      "signal_name": "e.g. dbcp.ActiveConnections saturation",
      "onset_time": "ISO timestamp or null",
      "onset_shape": "abrupt | gradual | unknown",
      "role": "likely_cause | confirmed_effect | ambiguous",
      "evidence": ["brief fact from signals or metrics"]
    }
  ]
}
```"""),
        ("server_monitoring.red_herring_filter", "system", "server_monitoring", """You are performing **red herring filtering** for a UAM server slowness incident.

**Primary archetype:** {{primary_archetype}}

**Common red herrings for this archetype:**
{{red_herring_hints}}

**Onset analysis:**
{{onset_analysis_json}}

**Deterministic recurring-operation candidates (fixed cadence):**
{{recurring_operations_json}}

**Structural signals for context:**
{{structural_signals_json}}

NEVER reject the following as red herrings:
- The largest or highest-frequency method burst detected in the window,
  unless you have explicit evidence it runs at identical volume outside
  the incident window at regular intervals (e.g. same burst size at
  04:00, 05:00, 06:00 daily).
- A burst that is temporally correlated with the reported CPU spike onset.
- High-volume/cardinality signals that were classified as the primary or
  secondary archetype — these are causal candidates, not noise.

Only reject signals as red herrings when:
- They appear at identical magnitude on a regular schedule outside the
  incident window (cadence_scheduled).
- They clearly started AFTER the incident onset (post_onset_symptom).
- They are infrastructure noise unrelated to application behavior (noise).

Identify signals that are NOT causal root causes:
- cadence_scheduled: recurring at fixed intervals regardless of incident
- post_onset_symptom: appears only after degradation_start
- steady_state: constant cost outside the incident window
- background_polling, low_impact, out_of_scope

Reply with ONLY a JSON array of rejections:
```json
[
  {
    "signal_description": "what was observed",
    "rejection_category": "cadence_scheduled | post_onset_symptom | steady_state | out_of_scope | low_impact | background_polling | other",
    "rejection_reason": "why not causal",
    "evidence": ["supporting fact"],
    "confidence": "CERTAIN | STRONG | INFERRED | WEAK"
  }
]
```

Be conservative: only reject when evidence supports it. Return [] if no red herrings are confident."""),
        ("server_monitoring.evidence_gathering", "system", "server_monitoring", f"""You are in the **{{{{current_phase}}}}** phase of the archetype-aware server monitoring workflow.
{{{{turn_line}}}}

**Primary archetype:** {{{{primary_archetype}}}} (confidence={{{{primary_confidence}}}})
**Investigation focus for primary:**
{{{{investigation_focus}}}}

**Competing hypotheses to test (mandatory — at least 1 SQL query must target one):**
{{{{competing_hypotheses}}}}

**Onset analysis:**
{{{{onset_analysis_json}}}}

**Structural signals:**
{{{{structural_signals_summary}}}}

**Application outlier signals:**
{{{{high_volume_signals_summary}}}}

**Critical windows:**
{{{{critical_windows_summary}}}}

{{{{onset_hint}}}}

{{{{duckdb_schema}}}}

Drive **targeted SQL exploration** on `server_metrics_wide` + `{{{{server_log_events_table}}}}`:
- **Mandatory:** at least one query MUST hit `{{{{server_log_events_table}}}}` for verbatim `raw_line` proof (REST traces, lapse(ms), DB/LDAP/Hibernate keywords).
- **Mandatory:** at least one query MUST hit `server_metrics_wide` around the onset anchor (not the whole file).
- On `{{{{server_log_events_table}}}}`, filter with `timestamp` (not `ts`).
- For metrics, avoid sparse NaN rows: filter `WHERE thread_count IS NOT NULL` or aggregate with `time_bucket(INTERVAL 1 MINUTE, timestamp)`.
- When parsing numbers from `raw_line` with `regexp_extract`, use `TRY_CAST(... AS BIGINT)` (never plain `CAST`) because non-matching lines return empty strings.
- Run at least one query that could confirm or reject the top competing hypothesis.
- Respect onset timing: distinguish causes from post-onset symptoms.

**SQL observations from prior turns this visit:**
{{{{prior_observations}}}}

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
FROM {{{{server_log_events_table}}}}
WHERE timestamp BETWEEN TIMESTAMP '2026-02-25 15:51:00' AND TIMESTAMP '2026-02-25 15:53:00'
  AND (raw_line LIKE '%lapse(ms)%' OR raw_line LIKE '%REST:%')
ORDER BY timestamp
LIMIT 15;
```

Output 1–3 safe, read-only SQL queries in fenced blocks.
{{{{sql_fence_rules}}}}
When you have enough evidence, say: "READY FOR SYNTHESIS" (no SQL in that turn).
On later turns, refine based on the SQL observations above — do not repeat failed or empty query shapes."""),
        ("server_monitoring.critic", "system", "server_monitoring", """CRITIC REVIEW — archetype-aware server monitoring analysis.

Current state summary:
{{state_summary}}

Phases completed: {{phases_completed}}
Archetype classification: {{classification_json}}
Competing hypotheses: {{competing_hypotheses_json}}
Evidence packages: {{evidence_package_keys}}
Onset analysis present: {{onset_analysis_present}}
Red herrings rejected: {{red_herring_count}}

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

Be evidence-based. Do not require high-volume/Count evidence unless that is the classified archetype."""),
        ("server_monitoring.ticket_refinement", "system", "server_monitoring", """You previously produced this server monitoring report:
{{current_findings}}
{{existing_evidence_block}}
The user has supplied this support ticket:
{{ticket_text}}

Your task is a SINGLE-PASS refinement only:
- Use ONLY the report and already-gathered evidence above.
- Do NOT run new SQL queries. Do NOT request additional data.
- Cross-reference the ticket symptoms against what is already evidenced.
- Output a concise "Ticket-Guided Root Cause Addendum" that maps each 
  ticket symptom to specific evidence already in the report.
- If a symptom cannot be matched to existing evidence, state explicitly:
  "Not evidenced in current analysis window."
- Keep the addendum short and evidence-based. No recommendations."""),
        ("server_monitoring.ticket_refinement.evidence_block", "fragment", "server_monitoring", """
**Already-gathered SQL observations (do not re-query for these):**
{{existing_evidence}}"""),
        ("server_monitoring.followup_sql", "system", "server_monitoring", f"""You are a senior IAM forensic analyst answering a **follow-up question** about a completed UAM server monitoring investigation.

You have a live DuckDB database for file `{{{{file_name}}}}` with:
- `server_metrics` — numeric monitoring snapshots ({{{{metric_row_count}}}} rows)
- `{{{{server_log_events_table}}}}` — every parsed log line ({{{{log_event_row_count}}}} rows)

{{{{duckdb_schema}}}}

**UAM5 metric dictionary (use exact metric names):**
{{{{uam5_dictionary}}}}

{{{{files_block}}}}
**Original analysis query:** {{{{original_query}}}}
**Analysis time window (UI filter):** {{{{start_time}}}} to {{{{end_time}}}}
{{{{observation_bounds_text}}}}
{{{{ticket_block}}}}
**Incident report excerpt:**
{{{{report_excerpt}}}}

**Recent chat:**
{{{{chat_history}}}}

**Follow-up question:** {{{{user_query}}}}

**SQL observations already gathered this turn:**
{{{{prior_observations}}}}

TASK:
- Answer the follow-up using SQL evidence from `server_metrics` and `{{{{server_log_events_table}}}}`.
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
{{{{sql_fence_rules}}}}
- After SQL results appear in observations, emit **FINAL_ANSWER:** with the direct answer (timeframes, counts, etc.).

OUTPUT CONTRACT (choose one):
1. If you need more data, emit up to **2** fenced ```sql blocks with read-only SELECT/WITH queries only.
2. If you have enough evidence, emit **FINAL_ANSWER:** followed by concise conversational markdown grounded in the SQL results.

Do not output both new SQL and FINAL_ANSWER in the same response.
{{{{force_synthesis_note}}}}"""),
        ("server_monitoring.followup_synthesis", "system", "server_monitoring", """You are finishing a server monitoring follow-up answer.

**Follow-up question:** {{user_query}}

**Incident report context:**
{{report_excerpt}}

**SQL observations gathered:**
{{prior_observations}}

Emit **FINAL_ANSWER:** followed by concise markdown.
- Ground every timeframe in the SQL observations or report context.
- If the user asked about "cause 1" / a numbered cause, name which cause you mapped it to.
- Do not emit SQL."""),
        ("reference.duckdb_schema", "fragment", "reference", DUCKDB_TABLE_SCHEMA_TEXT),
        ("reference.uam5_dictionary", "fragment", "reference", UAM5_SERVER_MONITORING_DICTIONARY_TEXT.strip()),
        ("reference.archetype_taxonomy", "fragment", "reference", _format_taxonomy_from_archetypes()),
        ("reference.sql_fence_rules", "fragment", "reference", _SQL_FENCE_FORMAT_RULES.strip()),
    ]

    for archetype in ALL_ARCHETYPES:
        defn = ARCHETYPE_TAXONOMY[archetype]
        templates.append((
            f"reference.archetype.{archetype}.investigation_focus",
            "fragment", "reference",
            "\n".join(f"- {item}" for item in defn["investigation_focus"]),
        ))
        templates.append((
            f"reference.archetype.{archetype}.red_herrings",
            "fragment", "reference",
            "\n".join(f"- {item}" for item in defn["common_red_herrings"]),
        ))

    grouped: dict[int, list[tuple[str, str, str, str]]] = {n: [] for n in range(1, 6)}
    for entry in templates:
        grouped[_prompt_group(entry[0])].append(entry)

    for group_num, (out_path, anchor, title, heading) in enumerate(OUT_FILES, start=1):
        sections: list[str] = [
            f"# {heading}",
            "",
            "> Edit prompts here. Loaded by `pipeline.prompt_loader`.",
            "> Placeholders use `{{snake_case}}`. Use single `{` in log examples.",
            "",
            f'<a id="{anchor}"></a>',
            f"## {title}",
            "",
        ]
        for sid, role, workflow, body in grouped[group_num]:
            sections.append(_section(sid, role, workflow, body))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(sections), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
    from pipeline.prompt_loader import list_prompt_ids, reload_prompts

    reload_prompts()
    print(f"Loaded {len(list_prompt_ids())} prompt sections")