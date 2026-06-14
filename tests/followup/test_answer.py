"""Unit tests for followup.answer module."""

import followup.answer as fa
from followup.context import EvidenceItem, FollowupIntent


def test_preview_text():
    assert fa._preview_text("short", 100) == "short"
    long_text = "a" * 500
    result = fa._preview_text(long_text, 100)
    assert len(result) <= 100
    assert result.strip("a") == ""


def test_rank_and_select_evidence():
    intent = FollowupIntent(
        ask_type="summary", entities=[], primary_keys=[],
        must_include=[], confidence=0.8, notes="test",
    )
    candidates = [
        EvidenceItem(evidence_id="R1", source="faiss", file_name="a.log",
                     relevance=1.0, anomaly_score=0.0, excerpt="error A", raw_text="error A"),
        EvidenceItem(evidence_id="R2", source="debug", file_name="b.log",
                     relevance=0.9, anomaly_score=0.0, excerpt="error B", raw_text="error B"),
    ]
    selected = fa._rank_and_select_evidence(intent, candidates, top_k=2)
    assert len(selected) <= 2


def test_evidence_table_separator_matches_header():
    selected = [
        EvidenceItem(
            evidence_id="R1",
            source="faiss",
            file_name="a.log",
            relevance=1.0,
            anomaly_score=2.7,
            excerpt="ERROR failed auth",
            raw_text="ERROR failed auth",
        )
    ]

    lines = fa._build_evidence_table(selected).splitlines()

    assert lines[1].count("|") == lines[2].count("|")


def test_evidence_table_escapes_pipe_characters_in_cells():
    selected = [
        EvidenceItem(
            evidence_id="R|1",
            source="raw_log",
            file_name="a|b.log",
            relevance=1.0,
            anomaly_score=2.7,
            excerpt="ERROR auth | failed",
            raw_text="ERROR auth | failed",
        )
    ]

    row = fa._build_evidence_table(selected).splitlines()[3]

    assert "\\|" in row
