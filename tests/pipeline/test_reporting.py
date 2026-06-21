"""Tests for pipeline.reporting module."""

import os
import tempfile
from pathlib import Path

import pytest

import pipeline.reporting as rp


def test_markdown_links_to_reportlab():
    md = "[link](http://example.com)"
    result = rp._markdown_links_to_reportlab(md)
    assert '<a href="http://example.com" color="blue"><u>link</u></a>' in result


def test_consolidate_reports_strips_ref_markers_for_server_monitoring(mock_llm):
    mock_llm.invoke = lambda messages, **kwargs: type(
        "FakeMsg",
        (),
        {
            "content": (
                "## Cross-File Summary\n"
                "Latency spike in server.log [REF_1][REF_2].\n\n"
                "## Consolidated Possible Root Causes (Ranked by Evidence Strength)\n"
                "Cause 1 (Strongest Evidence): Stall. Supporting Evidence: server.log. "
                "Confidence: High. Why not higher: Limited telemetry. "
                "Cause 2: Contention. Confidence: Moderate."
            ),
        },
    )()
    findings = [
        {
            "file": "server.log",
            "findings": "## 1. File-Wide Evidence Summary\nRuntime stall observed.",
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "category": "server_monitoring",
            "subcategory": "server_monitoring",
            "status": "ok",
            "query_valid": True,
            "source_path": "/tmp/server.log",
        },
    ]
    report = rp.consolidate_reports(findings, mode="server_monitoring")
    assert "REF_" not in report
    assert "Latency spike in server.log." in report
    assert "\n\nCause 2:" in report
    assert "\nSupporting Evidence:" in report


def test_consolidate_reports_emits_progress_for_server_monitoring(mock_llm, monkeypatch):
    collected: list[str] = []
    monkeypatch.setattr("pipeline.reporting.emit_ui_progress", collected.append)
    mock_llm.invoke = lambda messages, **kwargs: type(
        "FakeMsg", (), {"content": "Consolidated server monitoring report."}
    )()
    findings = [
        {
            "file": "server.log",
            "findings": "Runtime stall observed.",
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "category": "server_monitoring",
            "subcategory": "server_monitoring",
            "status": "ok",
            "query_valid": True,
            "source_path": "/tmp/server.log",
        },
    ]
    rp.consolidate_reports(findings, mode="server_monitoring")
    assert any("[REDUCE] Consolidating findings" in line for line in collected)


def test_consolidate_reports_basic(mock_llm):
    # The production mock in conftest truncates LLM prompts to 200 chars,
    # which cuts off the findings text. Override invoke so the assertion
    # below is meaningful.
    mock_llm.invoke = lambda messages, **kwargs: type(
        "FakeMsg", (), {"content": "Mock report mentions a.log and Found error."}
    )()

    findings = [
        {
            "file": "a.log",
            "findings": "Found error",
            "chunk_count": 5,
            "high_anomaly_count": 2,
            "category": "server_monitoring",
            "subcategory": "timeout",
            "status": "ok",
            "query_valid": True,
            "source_path": "/tmp/a.log",
        },
    ]
    report = rp.consolidate_reports(findings)
    assert "a.log" in report
    assert "Found error" in report


def test_export_to_pdf(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        report_dir = Path(d) / "reports"
        monkeypatch.setattr("artifact_paths.REPORT_DIR", report_dir)
        path = rp.export_to_pdf("# Test Report\n\nHello world.", "Test_Report.pdf")
        assert path is not None
        assert os.path.exists(path)
