import os
import re

import pipeline.parsing as p


def test_detect_log_structure_on_sample_log(sample_log_file):
    with open(sample_log_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    schema = p.detect_log_structure(lines)
    assert schema["timestamp_re"] is not None
    assert schema["thread_re"] is not None
    assert schema["timestamp_fmt"] == "%Y-%m-%d %H:%M:%S.%f"


def test_parse_line(sample_schema):
    line = "2024-01-15 09:23:45.123 [main] INFO Server started"
    ts, pk, clean = p._parse_line(line, sample_schema)
    assert ts is not None
    assert pk == "main"
    assert "Server started" in clean


def test_parse_line_no_timestamp():
    schema = {
        "timestamp_re": re.compile(r"(NOTIMESTAMP)"),
        "timestamp_fmt": "",
        "thread_re": None,
        "session_keys": [],
        "stack_trace_re": re.compile(
            r"^(?:\s+at |\s*Caused by:|\s*\.\.\. \d+ more)"
        ),
    }
    ts, pk, clean = p._parse_line("hello world", schema)
    assert ts is None
    assert pk is None
    assert clean == "hello world"
