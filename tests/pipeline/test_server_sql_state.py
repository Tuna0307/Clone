"""Tests for archetype-aware ServerMonitoringState models."""

from pipeline.server_sql.state import (
    ArchetypeClassification,
    ArchetypeHypothesis,
    OnsetAnalysis,
    OnsetRecord,
    ServerMonitoringState,
    StructuralSignal,
)


def test_structural_signal_round_trip():
    sig = StructuralSignal(
        signal_id="sig_test",
        signal_family="log_gap",
        summary="90s gap detected",
        strength=0.7,
    )
    state = ServerMonitoringState(file_name="test.log", file_path="/tmp/test.log")
    state.add_structural_signal(sig)
    assert len(state.structural_signals) == 1
    assert state.structural_signals[0].signal_family == "log_gap"


def test_archetype_classification_serialization():
    classification = ArchetypeClassification(
        primary=ArchetypeHypothesis(
            archetype="global_runtime_stall",
            confidence=0.8,
            supporting_signals=["sig_1"],
        ),
        secondary=ArchetypeHypothesis(
            archetype="db_connection_pressure",
            confidence=0.4,
            supporting_signals=["sig_2"],
        ),
        rejected_hypotheses=[
            ArchetypeHypothesis(
                archetype="high_volume_cardinality",
                confidence=0.2,
                rejection_reason="No large counts found",
            ),
        ],
        rationale="Log gaps dominate",
    )
    state = ServerMonitoringState(file_name="test.log", file_path="/tmp/test.log")
    state.set_archetype_classification(classification)
    dumped = state.to_serializable_dict()
    assert dumped["archetype_classification"]["primary"]["archetype"] == "global_runtime_stall"


def test_onset_analysis_update():
    analysis = OnsetAnalysis(
        onset_shape_overall="abrupt",
        signal_records=[
            OnsetRecord(
                signal_name="dbcp.ActiveConnections",
                role="confirmed_effect",
                onset_shape="gradual",
            ),
        ],
    )
    state = ServerMonitoringState(file_name="test.log", file_path="/tmp/test.log")
    state.update_onset_analysis(analysis)
    assert state.onset_analysis.onset_shape_overall == "abrupt"
    assert state.onset_analysis.signal_records[0].role == "confirmed_effect"