"""Tests for server_monitoring agentic SQL follow-up."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import followup.answer as fa
import followup.server_sql as fss
from followup.context import AnalysisContext, ArtifactEntry
from pipeline.server_metrics import copy_duckdb_file_to_memory, load_server_metrics_into_duckdb


def _make_schema():
    import re

    return {
        "timestamp_re": re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"),
        "timestamp_fmt": "%Y-%m-%d %H:%M:%S.%f",
        "thread_re": re.compile(r"\[(\S+)\]"),
        "session_re": None,
        "stack_trace_re": re.compile(r"^\s*(at\s+|\.\.\.\s+\d+\s+more)"),
        "timestamp_group": 1,
        "thread_group": 1,
        "session_group": None,
        "has_timestamp": True,
        "has_thread": True,
        "has_session": False,
        "is_api_request_log": False,
        "is_server_monitoring": True,
    }


def _seed_duckdb_file(tmpdir: str) -> tuple[str, str]:
    lines = [
        "2024-01-15 09:00:00.000 [main] INFO Server statistics={am.activeUsers=12}",
        "2024-01-15 09:00:01.000 [main] INFO user alice logged in",
        "2024-01-15 09:00:02.000 [main] INFO user bob logged in",
    ]
    log_path = os.path.join(tmpdir, "server.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    db_path = os.path.join(tmpdir, "server.duckdb")
    conn = load_server_metrics_into_duckdb(log_path, _make_schema(), db_path=db_path)
    conn.close()
    return log_path, db_path


def test_copy_duckdb_file_to_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        mem_conn = copy_duckdb_file_to_memory(db_path)
        try:
            log_cnt = mem_conn.execute("SELECT COUNT(*) FROM log_events").fetchone()[0]
            assert log_cnt == 3
        finally:
            mem_conn.close()


def test_load_temp_duckdb_into_session_deletes_temp_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        session = SimpleNamespace(server_monitoring_conns={})
        reports = [
            {
                "file": "server.log",
                "category": "server_monitoring",
                "duckdb_temp_path": db_path,
            }
        ]
        conns = fss.load_temp_duckdb_into_session(reports, session)
        assert "server.log" in conns
        assert not os.path.exists(db_path)
        fss.close_server_monitoring_connections(conns)


def test_answer_server_monitoring_followup_executes_sql_then_final_answer(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        mem_conn = copy_duckdb_file_to_memory(db_path)
        os.unlink(db_path)

        responses = [
            "```sql\nSELECT MIN(timestamp), MAX(timestamp) FROM log_events\n```",
            "FINAL_ANSWER: Degradation occurred between 2024-01-15 09:00:00 and 09:00:02.",
        ]
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = [MagicMock(content=text) for text in responses]
        monkeypatch.setattr("followup.intent._FOLLOWUP_LLM", fake_llm)

        context = AnalysisContext(
            query_text="analyze slowness",
            log_path="/tmp",
            start_time="",
            end_time="",
            report_text="Incident report body",
            entries=[
                ArtifactEntry(
                    file_name="server.log",
                    source_path=os.path.join(tmpdir, "server.log"),
                    faiss_index_dir="",
                    debug_evidence_file="",
                    metadata_rows=[],
                    selected_row_ids_for_reduce=[],
                    category="server_monitoring",
                    subcategory="server_monitoring",
                    duckdb_row_count=1,
                    log_event_row_count=3,
                )
            ],
            created_at="2024-01-15T00:00:00",
        )

        answer = fss.answer_server_monitoring_followup(
            context,
            "what timeframes did this happen?",
            duckdb_conns={"server.log": mem_conn},
        )

        assert "2024-01-15" in answer
        assert "SQL Queries Executed" in answer
        mem_conn.close()


def test_answer_server_monitoring_followup_emits_progress_callback(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        mem_conn = copy_duckdb_file_to_memory(db_path)
        os.unlink(db_path)

        responses = [
            "```sql\nSELECT COUNT(*) FROM log_events\n```",
            "FINAL_ANSWER: There are 3 log events in the selected window.",
        ]
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = [MagicMock(content=text) for text in responses]
        monkeypatch.setattr("followup.intent._FOLLOWUP_LLM", fake_llm)

        context = AnalysisContext(
            query_text="analyze slowness",
            log_path="/tmp",
            start_time="",
            end_time="",
            report_text="Incident report body",
            entries=[
                ArtifactEntry(
                    file_name="server.log",
                    source_path=os.path.join(tmpdir, "server.log"),
                    faiss_index_dir="",
                    debug_evidence_file="",
                    metadata_rows=[],
                    selected_row_ids_for_reduce=[],
                    category="server_monitoring",
                    subcategory="server_monitoring",
                    duckdb_row_count=1,
                    log_event_row_count=3,
                )
            ],
            created_at="2024-01-15T00:00:00",
        )

        progress_lines: list[str] = []
        answer = fss.answer_server_monitoring_followup(
            context,
            "how many log events are there?",
            duckdb_conns={"server.log": mem_conn},
            progress_callback=progress_lines.append,
        )

        assert "3 log events" in answer
        assert any("[Follow-up SQL] Step" in line for line in progress_lines)
        assert any("SELECT COUNT(*) FROM log_events" in line for line in progress_lines)
        assert any(line.startswith("→ ") for line in progress_lines)
        mem_conn.close()


def test_extract_sql_queries_from_unfenced_with_statement():
    text = (
        "Run these read-only queries and I'll give you the timeframes: "
        "and WITH stale_events AS (SELECT ts FROM log_events) "
        "SELECT MIN(ts), MAX(ts) FROM stale_events;"
    )
    queries = fss._extract_sql_queries(text)
    assert len(queries) == 1
    assert queries[0].lower().startswith("with stale_events")


def test_answer_server_monitoring_followup_executes_unfenced_sql(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        mem_conn = copy_duckdb_file_to_memory(db_path)
        os.unlink(db_path)

        responses = [
            (
                "I need to pull the windows. Run these queries: "
                "and WITH bounds AS (SELECT MIN(timestamp) AS start_ts, MAX(timestamp) AS end_ts "
                "FROM log_events) SELECT * FROM bounds;"
            ),
            "FINAL_ANSWER: Cause 1 occurred between 2024-01-15 09:00:00 and 09:00:02.",
        ]
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = [MagicMock(content=text) for text in responses]
        monkeypatch.setattr("followup.intent._FOLLOWUP_LLM", fake_llm)

        context = AnalysisContext(
            query_text="analyze slowness",
            log_path="/tmp",
            start_time="",
            end_time="",
            report_text="Cause 1: Hibernate stale object exceptions.",
            entries=[
                ArtifactEntry(
                    file_name="server.log",
                    source_path=os.path.join(tmpdir, "server.log"),
                    faiss_index_dir="",
                    debug_evidence_file="",
                    metadata_rows=[],
                    selected_row_ids_for_reduce=[],
                    category="server_monitoring",
                    subcategory="server_monitoring",
                    duckdb_row_count=1,
                    log_event_row_count=3,
                )
            ],
            created_at="2024-01-15T00:00:00",
        )

        answer = fss.answer_server_monitoring_followup(
            context,
            "For cause 1, what are the timeframes it happened?",
            duckdb_conns={"server.log": mem_conn},
        )

        assert "2024-01-15" in answer
        assert "run these" not in answer.lower()
        assert "SQL Queries Executed" in answer
        mem_conn.close()


def test_answer_server_monitoring_followup_retries_when_model_delegates_to_user(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, db_path = _seed_duckdb_file(tmpdir)
        mem_conn = copy_duckdb_file_to_memory(db_path)
        os.unlink(db_path)

        responses = [
            "Run these two read-only queries and I'll give you the precise timeframes.",
            "```sql\nSELECT MIN(timestamp), MAX(timestamp) FROM log_events\n```",
            "FINAL_ANSWER: Cause 1 happened from 2024-01-15 09:00:00 to 09:00:02.",
        ]
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = [MagicMock(content=text) for text in responses]
        monkeypatch.setattr("followup.intent._FOLLOWUP_LLM", fake_llm)

        context = AnalysisContext(
            query_text="analyze slowness",
            log_path="/tmp",
            start_time="",
            end_time="",
            report_text="Cause 1: Hibernate stale object exceptions.",
            entries=[
                ArtifactEntry(
                    file_name="server.log",
                    source_path=os.path.join(tmpdir, "server.log"),
                    faiss_index_dir="",
                    debug_evidence_file="",
                    metadata_rows=[],
                    selected_row_ids_for_reduce=[],
                    category="server_monitoring",
                    subcategory="server_monitoring",
                    duckdb_row_count=1,
                    log_event_row_count=3,
                )
            ],
            created_at="2024-01-15T00:00:00",
        )

        answer = fss.answer_server_monitoring_followup(
            context,
            "For cause 1, what are the timeframes it happened?",
            duckdb_conns={"server.log": mem_conn},
        )

        assert fake_llm.invoke.call_count == 3
        assert "2024-01-15" in answer
        assert "run these" not in answer.lower()
        mem_conn.close()


def test_execute_safe_sql_allows_hibernate_literal_patterns():
    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [
            "2024-01-15 09:00:00.000 [main] ERROR org.hibernate.StaleObjectStateException: Row was updated or deleted by another transaction",
            "2024-01-15 09:05:00.000 [main] ERROR Could not synchronize database state with session",
        ]
        log_path = os.path.join(tmpdir, "server.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        db_path = os.path.join(tmpdir, "server.duckdb")
        conn = load_server_metrics_into_duckdb(log_path, _make_schema(), db_path=db_path)
        conn.close()
        mem_conn = copy_duckdb_file_to_memory(db_path)

        sql = """
        WITH stale AS (
          SELECT timestamp, raw_line
          FROM log_events
          WHERE raw_line ILIKE '%StaleObjectStateException%'
             OR raw_line ILIKE '%Could not synchronize database state with session%'
             OR raw_line ILIKE '%Row was updated or deleted by another transaction%'
        )
        SELECT min(timestamp) AS first_seen, max(timestamp) AS last_seen, count(*) AS total
        FROM stale
        """
        row_count, observation = fss._execute_safe_sql(mem_conn, sql)
        assert row_count == 1
        assert "2024-01-15" in observation
        mem_conn.close()


def test_execute_safe_sql_uses_timestamp_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [
            "2024-01-15 09:00:00.000 [main] ERROR org.hibernate.StaleObjectStateException: Row was updated",
            "2024-01-15 09:05:00.000 [main] ERROR Could not synchronize database state with session",
        ]
        log_path = os.path.join(tmpdir, "server.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        db_path = os.path.join(tmpdir, "server.duckdb")
        conn = load_server_metrics_into_duckdb(log_path, _make_schema(), db_path=db_path)
        conn.close()
        mem_conn = copy_duckdb_file_to_memory(db_path)

        sql = """
        SELECT min(timestamp) AS first_seen, max(timestamp) AS last_seen, count(*) AS total
        FROM log_events
        WHERE lower(raw_line) LIKE '%staleobjectstateexception%'
           OR lower(raw_line) LIKE '%could not synchronize database state with session%'
        """
        row_count, observation = fss._execute_safe_sql(mem_conn, sql)
        assert row_count == 1
        assert "2024-01-15" in observation
        mem_conn.close()


def test_answer_analysis_results_query_routes_to_server_sql(monkeypatch):
    monkeypatch.setattr(
        "followup.server_sql.answer_server_monitoring_followup",
        lambda **kwargs: "SQL follow-up answer",
    )

    context = AnalysisContext(
        query_text="analyze",
        log_path="/tmp",
        start_time="",
        end_time="",
        report_text="report",
        entries=[
            ArtifactEntry(
                file_name="server.log",
                source_path="/tmp/server.log",
                faiss_index_dir="",
                debug_evidence_file="",
                metadata_rows=[],
                selected_row_ids_for_reduce=[],
                category="server_monitoring",
                subcategory="server_monitoring",
            )
        ],
        created_at="2024-01-15T00:00:00",
    )

    result = fa.answer_analysis_results_query(context, "which users were affected?")
    assert result == "SQL follow-up answer"