"""Tests for archetype-aware server_sql_graph routing and parsing."""

import os
import tempfile
from unittest.mock import MagicMock

from pipeline.server_metrics import load_server_metrics_into_duckdb
from pipeline.server_sql.state import ServerMonitoringState
from pipeline.server_sql_graph import (
    _parse_classification_from_llm,
    _parse_red_herrings_from_llm,
    _produce_partial_report_from_typed_state,
    _should_continue_after_critic,
    critic_node,
    evidence_gathering_node,
)


def test_parse_classification_from_llm():
    text = '''```json
    {"primary": {"archetype": "global_runtime_stall", "confidence": 0.8, "supporting_signals": ["s1"]},
     "secondary": null,
     "rejected_hypotheses": [],
     "rationale": "log gaps"}
    ```'''
    fallback = {"primary": {"archetype": "high_volume_cardinality", "confidence": 0.5, "supporting_signals": []}}
    result = _parse_classification_from_llm(text, fallback)
    assert result["primary"]["archetype"] == "global_runtime_stall"


def test_parse_red_herrings_from_llm():
    text = '''[{"signal_description": "hourly job", "rejection_category": "cadence_scheduled",
               "rejection_reason": "fixed cadence", "evidence": ["e1"], "confidence": "STRONG"}]'''
    result = _parse_red_herrings_from_llm(text)
    assert len(result) == 1
    assert result[0]["rejection_category"] == "cadence_scheduled"


def test_should_continue_after_critic_reclassify():
    state = ServerMonitoringState(file_name="t.log", file_path="/t.log")
    state.critic_feedback_history.append({"verdict": "RECLASSIFY"})
    # With 1 RECLASSIFY in history, reclassify_count = 1, so 1 < 1 is False;
    # the function routes to report_synthesis instead of broad_diagnostic.
    assert _should_continue_after_critic(state) == "report_synthesis"


def test_critic_retry_routes_to_evidence_gathering():
    state = ServerMonitoringState(file_name="t.log", file_path="/t.log")
    state.max_evidence_critic_retry_loops = 1
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="RETRY: need more log_events evidence")
    object.__setattr__(state, "_llm", mock_llm)

    result = critic_node(state)
    # critic_node itself still sets current_phase = "evidence_gathering" for RETRY
    assert result.current_phase == "evidence_gathering"
    assert result.evidence_critic_retry_loops == 1
    # But the LangGraph routing function now sends everything to report_synthesis
    assert _should_continue_after_critic(result) == "report_synthesis"


def test_critic_retry_exhausted_routes_to_synthesis():
    state = ServerMonitoringState(file_name="t.log", file_path="/t.log")
    state.max_evidence_critic_retry_loops = 1
    state.evidence_critic_retry_loops = 1
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="RETRY: still weak")
    object.__setattr__(state, "_llm", mock_llm)

    result = critic_node(state)
    assert result.current_phase == "report_synthesis"


def test_critic_reclassify_routes_to_broad_diagnostic():
    state = ServerMonitoringState(file_name="t.log", file_path="/t.log")
    state.max_reclassifications = 1
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="RECLASSIFY: evidence supports global_runtime_stall")
    object.__setattr__(state, "_llm", mock_llm)

    result = critic_node(state)
    assert result.current_phase == "broad_diagnostic_and_archetype_classification"
    assert result.reclassification_count == 1


def _make_schema():
    import re

    return {
        "timestamp_re": re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"),
        "timestamp_fmt": "%Y-%m-%d %H:%M:%S.%f",
        "thread_re": re.compile(r"\[(\S+)\]"),
        "session_re": None,
        "stack_trace_re": None,
        "timestamp_group": 1,
        "thread_group": 1,
        "session_group": None,
        "has_timestamp": True,
        "has_thread": True,
        "has_session": False,
    }


def test_evidence_gathering_runs_supplements_and_multi_turn(monkeypatch):
    lines = [
        "2026-02-25 15:51:35.286000 [main] INFO Server statistics={jvm.threadCount=344}",
        "2026-02-25 15:52:13.036000 [exec-1] DEBUG REST:authn/login - exit,lapse(ms)=97550",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "server.log")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        conn = load_server_metrics_into_duckdb(
            path, _make_schema(), db_path=os.path.join(tmpdir, "test.duckdb")
        )

        responses = [
            (
                "```sql\nSELECT timestamp, raw_line FROM log_events "
                "WHERE raw_line LIKE '%lapse(ms)%' LIMIT 5\n```"
            ),
            "READY FOR SYNTHESIS",
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [MagicMock(content=text) for text in responses]

        state = ServerMonitoringState(file_name="server.log", file_path=path)
        state.set_archetype_classification({
            "primary": {"archetype": "mixed_compound", "confidence": 0.9, "supporting_signals": []},
            "rejected_hypotheses": [],
            "rationale": "test",
        })
        state.update_onset_analysis({
            "degradation_start": "2026-02-25T15:51:35.286000",
            "onset_shape_overall": "abrupt",
            "signal_records": [],
        })
        object.__setattr__(state, "_duckdb_conn", conn)
        object.__setattr__(state, "_llm", mock_llm)

        result = evidence_gathering_node(state)
        try:
            assert result.current_phase == "critic"
            assert result.queries_executed >= 4
            assert mock_llm.invoke.call_count == 2
            assert "pkg_mixed_compound" in result.evidence_packages
            supplement_nodes = [
                step.node for step in result.trace if str(step.node).startswith("evidence_supplement_")
            ]
            assert len(supplement_nodes) == 3
        finally:
            conn.close()


def test_partial_report_includes_archetype():
    state = ServerMonitoringState(file_name="t.log", file_path="/t.log")
    state.set_archetype_classification({
        "primary": {"archetype": "global_runtime_stall", "confidence": 0.75, "supporting_signals": []},
        "rejected_hypotheses": [],
        "rationale": "log gaps",
    })
    report = _produce_partial_report_from_typed_state(state)
    assert "global_runtime_stall" in report
    assert "Incident Archetype Assessment" in report