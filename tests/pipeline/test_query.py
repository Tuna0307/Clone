# tests/pipeline/test_query.py
from datetime import datetime

from pipeline.parsing import parse_query_datetime
import pipeline.query as q


def test_parse_query_datetime_iso():
    result = parse_query_datetime("2024-01-15T09:30:00")
    assert result == datetime(2024, 1, 15, 9, 30, 0)


def test_parse_query_datetime_date_only_end_of_day():
    result = parse_query_datetime("2024-01-15", use_end_of_day_for_date_only=True)
    assert result == datetime(2024, 1, 15, 23, 59, 59, 999999)


def test_build_query_context():
    ctx = q.build_query_context("login timeout", "2024-01-15", "2024-01-16")
    assert ctx["query_text"] == "login timeout"
    assert ctx["start_time"] == datetime(2024, 1, 15, 0, 0, 0)
    assert ctx["end_time"] == datetime(2024, 1, 16, 23, 59, 59, 999999)


def test_load_search_config_defaults():
    config = q.load_search_config("nonexistent.json")
    assert config["search_strategy"] == "signal_first_anomaly_ranking"
    assert "iam_critical_keywords" in config


def test_line_overlaps_query_window():
    ctx = {"start_time": datetime(2024, 1, 1), "end_time": datetime(2024, 1, 31)}
    assert q._line_overlaps_query_window(datetime(2024, 1, 15), ctx) is True
    assert q._line_overlaps_query_window(datetime(2024, 2, 1), ctx) is False
    assert q._line_overlaps_query_window(None, ctx) is True
