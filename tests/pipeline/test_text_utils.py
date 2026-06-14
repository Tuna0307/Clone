"""Tests for pipeline.text_utils."""

from __future__ import annotations

import re

import pipeline.text_utils as tu


def test_is_error_bearing():
    assert tu._is_error_bearing("ERROR: connection failed") is True
    assert tu._is_error_bearing("INFO: all good") is False


def test_is_iam_critical_text():
    assert tu._is_iam_critical_text("CryptoService failure", ["CryptoService"]) is True
    assert tu._is_iam_critical_text("random text", ["CryptoService"]) is False


def test_is_noisy_text():
    patterns = [re.compile(r"Audit took \d+")]
    assert tu._is_noisy_text("Audit took 123ms", patterns) is True
    assert tu._is_noisy_text("Something else", patterns) is False


def test_contains_any_marker():
    assert tu._contains_any_marker("hello world", ["world"]) is True
    assert tu._contains_any_marker("hello world", ["foo"]) is False
