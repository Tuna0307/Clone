"""Agentic SQL follow-up for server_monitoring analysis sessions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from followup.context import AnalysisContext, ArtifactEntry, _markdown_table_cell
from followup.intent import _format_chat_history, _get_followup_llm
from pipeline.constants import SERVER_SQL_FOLLOWUP_MAX_STEPS, SERVER_SQL_RESULT_TRUNCATE
from pipeline.server_metrics import (
    copy_duckdb_file_to_memory,
    format_duckdb_observation_bounds,
    format_query_dataframe,
    get_duckdb_observation_bounds,
    get_sql_safety_rejection_reason,
    is_safe_select,
    normalize_llm_sql,
)
from pipeline.progress import ProgressCallback, emit_ui_progress, progress_callback_scope
from pipeline.server_sql.prompts import (
    build_followup_sql_instruction,
    build_followup_synthesis_instruction,
)

_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.*)", re.DOTALL | re.IGNORECASE)
_DELEGATION_RE = re.compile(
    r"(?:run\s+(?:these|the\s+following)\s+(?:read-?only\s+)?quer(?:y|ies)|"
    r"i(?:'ll| will)\s+(?:give|provide|pull|fetch)|"
    r"please\s+run|execute\s+these\s+quer(?:y|ies)|"
    r"you\s+(?:can|should|need\s+to)\s+run)",
    re.IGNORECASE,
)


@dataclass
class _SqlExecutionRecord:
    file_name: str
    sql: str
    row_count: int
    observation: str


def is_server_monitoring_followup_mode(context: AnalysisContext) -> bool:
    """Return True when every analyzed file used server_monitoring mode."""
    if not context.entries:
        return False
    return all(entry.category == "server_monitoring" for entry in context.entries)


def close_server_monitoring_connections(duckdb_conns: dict[str, Any] | None) -> None:
    """Close in-memory DuckDB connections held for follow-up."""
    if not duckdb_conns:
        return
    for conn in duckdb_conns.values():
        try:
            conn.close()
        except Exception:
            pass


def load_temp_duckdb_into_session(
    per_file_reports: list[dict[str, Any]],
    session_store: Any,
) -> dict[str, Any]:
    """Copy temp workflow DuckDB files into in-memory connections for follow-up.

    Stores connections on ``session_store.server_monitoring_conns`` keyed by file name.
    Deletes the temp file after a successful in-memory copy.
    """
    close_server_monitoring_connections(getattr(session_store, "server_monitoring_conns", None))
    conns: dict[str, Any] = {}

    for report in per_file_reports:
        if str(report.get("category", "")).strip() != "server_monitoring":
            continue
        file_name = str(report.get("file", "")).strip()
        temp_path = str(report.get("duckdb_temp_path", "")).strip()
        if not file_name or not temp_path or not os.path.exists(temp_path):
            continue
        try:
            conns[file_name] = copy_duckdb_file_to_memory(temp_path)
        except Exception as exc:
            print(f"  [Follow-up] Warning: could not load in-memory DuckDB for {file_name}: {exc}")
            continue
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    session_store.server_monitoring_conns = conns
    return conns


def _extract_final_answer(text: str) -> str | None:
    match = _FINAL_ANSWER_RE.search(text)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer or None


def _is_delegating_to_user(text: str) -> bool:
    """True when the model asks the user to run SQL instead of emitting executable blocks."""
    return bool(_DELEGATION_RE.search(text))


def _extract_sql_queries(text: str) -> list[str]:
    """Extract up to two safe read-only SQL statements from an LLM turn."""
    fenced = [block.strip() for block in _SQL_BLOCK_RE.findall(text) if block.strip()]
    if fenced:
        return fenced[:2]

    cleaned = _FINAL_ANSWER_RE.sub("", text)
    start_match = re.search(r"(?is)\b(with|select)\b", cleaned)
    if not start_match:
        return []

    remainder = cleaned[start_match.start():]
    queries: list[str] = []
    for part in re.split(r";\s*", remainder):
        candidate = re.sub(r"^(?:and\s+)", "", part.strip(), flags=re.IGNORECASE)
        if not candidate:
            continue
        if not re.match(r"(?is)^(with|select)\b", candidate):
            continue
        sql = candidate.rstrip(";").strip()
        if is_safe_select(sql):
            queries.append(sql)
        if len(queries) >= 2:
            break
    return queries


def _emit_followup_sql_progress(
    sql: str,
    row_count: int,
    *,
    step_number: int,
    error: str | None = None,
    rejection_reason: str | None = None,
) -> None:
    preview = sql.replace("\n", " ").strip()
    if len(preview) > 240:
        preview = preview[:240] + "..."
    emit_ui_progress(f"[Follow-up SQL] Step {step_number}")
    emit_ui_progress(preview)
    if error:
        emit_ui_progress(f"→ SQL error: {error}")
    elif rejection_reason:
        emit_ui_progress(f"→ Rejected by SQL guard: {rejection_reason}")
    elif row_count < 0:
        emit_ui_progress("→ Rejected by SQL guard")
    else:
        emit_ui_progress(f"→ {row_count} rows")


def _truncate_observation(observation: str, limit: int = SERVER_SQL_RESULT_TRUNCATE) -> str:
    if len(observation) <= limit:
        return observation
    return observation[:limit] + " ... (truncated)"


def _execute_safe_sql(conn: Any, sql: str) -> tuple[int, str]:
    rejection_reason = get_sql_safety_rejection_reason(sql)
    if rejection_reason:
        return -1, f"Rejected by SQL guard: {rejection_reason}"
    try:
        df = conn.execute(normalize_llm_sql(sql)).fetchdf()
        if df.empty:
            return 0, "No rows returned."
        obs = format_query_dataframe(df)
        return len(df), _truncate_observation(obs)
    except Exception as exc:
        return -1, f"SQL error: {exc}"


def _build_sql_queries_table(records: list[_SqlExecutionRecord]) -> str:
    if not records:
        return ""
    header = "| File | Rows | SQL |"
    sep = "| --- | ---: | --- |"
    rows = [
        "| "
        + " | ".join(
            [
                _markdown_table_cell(rec.file_name),
                _markdown_table_cell(
                    "error" if rec.row_count < 0 else str(rec.row_count)
                ),
                _markdown_table_cell(rec.sql.replace("\n", " ")[:240]),
            ]
        )
        + " |"
        for rec in records
    ]
    return "\n".join(["**SQL Queries Executed**", "", header, sep, *rows])


def _run_followup_sql_loop(
    *,
    conn: Any,
    entry: ArtifactEntry,
    context: AnalysisContext,
    query: str,
    chat_history: list[dict[str, str]] | None,
    available_files: list[str],
) -> tuple[str, list[_SqlExecutionRecord]]:
    llm = _get_followup_llm()
    prior_observations: list[str] = []
    execution_records: list[_SqlExecutionRecord] = []
    final_answer = ""

    ticket_excerpt = ""
    if context.ticket_text:
        ticket_excerpt = context.ticket_text[:1200]

    observation_bounds_text = format_duckdb_observation_bounds(
        get_duckdb_observation_bounds(conn)
    )

    for step_idx in range(SERVER_SQL_FOLLOWUP_MAX_STEPS):
        is_last_step = step_idx >= SERVER_SQL_FOLLOWUP_MAX_STEPS - 1
        if is_last_step and prior_observations:
            emit_ui_progress("[Follow-up SQL] Synthesizing answer...")
            instruction = build_followup_synthesis_instruction(
                user_query=query,
                report_excerpt=context.report_text,
                prior_observations=prior_observations,
            )
        else:
            instruction = build_followup_sql_instruction(
                user_query=query,
                file_name=entry.file_name,
                metric_row_count=entry.duckdb_row_count,
                log_event_row_count=entry.log_event_row_count,
                report_excerpt=context.report_text,
                original_query=context.query_text,
                start_time=context.start_time,
                end_time=context.end_time,
                ticket_excerpt=ticket_excerpt,
                chat_history=_format_chat_history(chat_history),
                prior_observations=prior_observations,
                available_files=available_files,
                force_synthesis=is_last_step,
                observation_bounds_text=observation_bounds_text,
            )

        try:
            response = llm.invoke([{"role": "user", "content": instruction}])
            text = str(getattr(response, "content", response)).strip()
        except Exception as exc:
            return (
                "I couldn't run the follow-up SQL agent right now due to an LLM error. "
                f"Please try again. Details: {exc}",
                execution_records,
            )

        sql_blocks = _extract_sql_queries(text)
        maybe_final = _extract_final_answer(text)
        if maybe_final and not sql_blocks:
            final_answer = maybe_final
            break

        if not sql_blocks:
            if _is_delegating_to_user(text) or re.search(r"(?is)\b(with|select)\b", text):
                from followup.prompts import build_server_followup_sql_retry_nudge

                prior_observations.append(build_server_followup_sql_retry_nudge())
                continue
            final_answer = text.strip() or (
                "I could not gather enough SQL evidence to answer that follow-up."
            )
            break

        for sql in sql_blocks[:2]:
            sql_text = sql.strip()
            row_count, observation = _execute_safe_sql(conn, sql_text)
            if row_count < 0 and observation.startswith("Rejected by SQL guard:"):
                _emit_followup_sql_progress(
                    sql_text,
                    row_count,
                    step_number=step_idx + 1,
                    rejection_reason=observation.removeprefix("Rejected by SQL guard: ").strip(),
                )
            elif row_count < 0:
                _emit_followup_sql_progress(
                    sql_text,
                    row_count,
                    step_number=step_idx + 1,
                    error=observation.removeprefix("SQL error: "),
                )
            else:
                _emit_followup_sql_progress(sql_text, row_count, step_number=step_idx + 1)
            prior_observations.append(
                f"SQL:\n{sql.strip()}\n\nResult ({row_count} rows):\n{observation}"
            )
            execution_records.append(
                _SqlExecutionRecord(
                    file_name=entry.file_name,
                    sql=sql.strip(),
                    row_count=row_count,
                    observation=observation,
                )
            )

    if not final_answer and prior_observations:
        try:
            synth = llm.invoke([
                {
                    "role": "user",
                    "content": build_followup_synthesis_instruction(
                        user_query=query,
                        report_excerpt=context.report_text,
                        prior_observations=prior_observations,
                    ),
                }
            ])
            synth_text = str(getattr(synth, "content", synth)).strip()
            final_answer = _extract_final_answer(synth_text) or synth_text
        except Exception:
            final_answer = ""

    if not final_answer:
        final_answer = (
            "I reached the SQL step limit before producing a final answer. "
            "Please narrow your question or ask about a specific timeframe, user, or metric."
        )

    return final_answer, execution_records


def answer_server_monitoring_followup(
    context: AnalysisContext,
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    duckdb_conns: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Answer a follow-up question using agentic SQL against in-memory DuckDB tables."""
    if not query.strip():
        return "Please provide a follow-up question."

    server_entries = [entry for entry in context.entries if entry.category == "server_monitoring"]
    if not server_entries:
        return (
            "No in-memory DuckDB tables are available for server monitoring follow-up. "
            "Run a Server Monitoring analysis first."
        )

    if not duckdb_conns:
        return (
            "The in-memory DuckDB for this session is no longer available. "
            "Please rerun the Server Monitoring analysis to reload the tables."
        )

    available_files = [entry.file_name for entry in server_entries]

    with progress_callback_scope(progress_callback):
        return _answer_server_monitoring_followup_impl(
            context=context,
            query=query,
            chat_history=chat_history,
            duckdb_conns=duckdb_conns,
            server_entries=server_entries,
            available_files=available_files,
        )


def _answer_server_monitoring_followup_impl(
    *,
    context: AnalysisContext,
    query: str,
    chat_history: list[dict[str, str]] | None,
    duckdb_conns: dict[str, Any] | None,
    server_entries: list[ArtifactEntry],
    available_files: list[str],
) -> str:
    answers: list[str] = []
    all_records: list[_SqlExecutionRecord] = []

    for entry in server_entries:
        conn = duckdb_conns.get(entry.file_name)
        if conn is None:
            answers.append(
                f"**{entry.file_name}**: in-memory DuckDB unavailable for this file."
            )
            continue

        answer, records = _run_followup_sql_loop(
            conn=conn,
            entry=entry,
            context=context,
            query=query,
            chat_history=chat_history,
            available_files=available_files,
        )
        all_records.extend(records)
        if len(server_entries) == 1:
            answers.append(answer)
        else:
            answers.append(f"### {entry.file_name}\n\n{answer}")

    combined = "\n\n".join(answers).strip()
    sql_table = _build_sql_queries_table(all_records)
    if sql_table:
        return f"{combined}\n\n{sql_table}"
    return combined