"""Helpers for rendering small source-log windows in the Streamlit app."""

from __future__ import annotations

import re
import hashlib
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class LogWindowLine:
    """One source-log line in a displayed context window."""

    line_number: int
    text: str
    is_target: bool


@dataclass(frozen=True)
class LogLineWindow:
    """A bounded source-log window around a cited line."""

    target_line: int
    lines: list[LogWindowLine]


def parse_line_reference_start(line_reference: str) -> int | None:
    """
    Return the first line number from citation text.

    Examples:
        "line 1152" -> 1152
        "lines 1152-1158" -> 1152
        "SystemOut.log, lines 1152" -> 1152
    """
    match = re.search(r"\blines?\s+(\d+)", line_reference, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def build_log_reference_key(
    source_path: str,
    target_line: int,
    key_prefix: str,
    line_index: int,
) -> str:
    """
    Build a stable UI key for one rendered log reference.

    The key includes the rendered message scope so the same file/line can appear
    in multiple chat messages without Streamlit widget collisions.
    """
    seed = f"{key_prefix}|{source_path}|{target_line}|{line_index}"
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"view_log_ref_{digest}"


def read_log_line_window(
    file_path: str,
    target_line: int,
    context_radius: int = 5,
) -> LogLineWindow:
    """
    Stream a source log and return only lines around target_line.

    This avoids loading very large logs into memory.
    """
    if target_line < 1:
        raise ValueError("target_line must be >= 1")
    if context_radius < 0:
        raise ValueError("context_radius must be >= 0")

    before_lines: deque[LogWindowLine] = deque(maxlen=context_radius)
    window_lines: list[LogWindowLine] = []
    found_target = False
    last_needed_line = target_line + context_radius

    with open(file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        for line_number, raw_line in enumerate(file_handle, start=1):
            text = raw_line.rstrip("\r\n")

            if line_number < target_line:
                before_lines.append(LogWindowLine(line_number, text, False))
                continue

            if line_number == target_line:
                window_lines.extend(before_lines)
                window_lines.append(LogWindowLine(line_number, text, True))
                found_target = True
                continue

            if found_target and line_number <= last_needed_line:
                window_lines.append(LogWindowLine(line_number, text, False))
                continue

            if found_target and line_number > last_needed_line:
                break

    return LogLineWindow(target_line=target_line, lines=window_lines)
