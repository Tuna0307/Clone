"""File discovery and streaming utilities for the IAM pipeline."""

import os
from typing import Iterator, Union

from pipeline.constants import MAX_LOG_FILE_SIZE_BYTES


def get_log_files_from_path(path: str) -> list[str]:
    """
    Recursively find all log files in a directory or return the single file.

    Args:
        path: Path to a file or directory

    Returns:
        List of absolute file paths
    """
    if os.path.isfile(path):
        return [os.path.abspath(path)]

    log_files: list[str] = []
    valid_extensions = {'.log', '.txt', '.out', '.err', '.msg'}

    print(f"-> Scanning directory: {path}")
    for root, _, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, file)
                log_files.append(os.path.abspath(full_path))

    print(f"   Found {len(log_files)} log files.")
    return log_files


def format_file_size(size_bytes: int) -> str:
    """
    Return human-readable file size string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like '524.6 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def stream_file_lines(file_path: str) -> Iterator[str]:
    """
    Generator that yields lines from a file one at a time.
    Memory-efficient for extremely large files.

    Args:
        file_path: Absolute path to the file

    Yields:
        Individual lines (with newline chars)
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            yield line


# =============================================================================
# Ticket / incident context file helpers (server_monitoring only)
# Used by CLI (--ticket-file) and app.py (dedicated uploader) to load
# ticket.md / *-ticket.* files and pass their text for the post-report
# refinement iteration of the agentic SQL loop. These are NEVER used to
# auto-scan directories or alter log discovery for API mode.
# =============================================================================

def is_ticket_file(filename: str) -> bool:
    """
    Return True if the filename (not full path) looks like a support ticket.
    Rule: stem contains "ticket" (case-insensitive) and extension is .md/.txt/.markdown.
    This is the single source of truth for the "word ticket in it" detection.
    """
    name = filename.lower()
    stem, ext = os.path.splitext(name)
    if "ticket" not in stem:
        return False
    return ext in {".md", ".txt", ".markdown"}


def find_ticket_files(paths: Union[str, list[str]]) -> list[str]:
    """
    Find ticket files (by is_ticket_file rule) under the given path(s).
    Mirrors get_log_files_from_path style but collects only matching ticket files.
    Supports single file or directory (recursive). Returns absolute paths.
    Intended for explicit --ticket-file or uploader flows, not auto-discovery.

    Note: this function is provided for symmetry and future use; current
    implementation (per user decision) does not auto-invoke it on log input paths.
    """
    if isinstance(paths, str):
        paths_list = [paths]
    else:
        paths_list = paths

    ticket_files: list[str] = []
    ticket_exts = {".md", ".txt", ".markdown"}

    for path_input in paths_list:
        if not os.path.exists(path_input):
            continue
        if os.path.isfile(path_input):
            if is_ticket_file(os.path.basename(path_input)):
                ticket_files.append(os.path.abspath(path_input))
            continue

        # Directory: walk and collect only ticket-named files
        for root, _, files in os.walk(path_input):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ticket_exts and "ticket" in file.lower():
                    full = os.path.join(root, file)
                    ticket_files.append(os.path.abspath(full))

    # Dedup while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in ticket_files:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def load_and_format_ticket_text(
    ticket_paths: Union[str, list[str]], max_chars: int = 12000
) -> str:
    """
    Read one or more ticket files and return a single formatted string suitable
    for injection into the LLM (post-report refinement pass for server_monitoring).

    Each file is prefixed with a clear header so the LLM knows the source.
    Total output is truncated at max_chars (from the end) to protect context length.
    Uses the same utf-8 + errors=replace policy as the rest of the pipeline.
    """
    if not ticket_paths:
        return ""

    if isinstance(ticket_paths, str):
        paths_list = [ticket_paths]
    else:
        paths_list = ticket_paths

    parts: list[str] = []
    for p in paths_list:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            header = f"=== TICKET FILE: {os.path.basename(p)} ===\n"
            parts.append(header + content + "\n")
        except Exception as e:
            parts.append(f"=== TICKET FILE: {os.path.basename(p)} (read error: {e}) ===\n")

    combined = "\n".join(parts).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n... [ticket truncated]"
    return combined
