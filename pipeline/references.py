"""Reference-formatting helpers for citations, line ranges, and file URIs."""

import os
import re
import urllib.parse
from typing import Any, Optional

from langchain_core.documents import Document

from pipeline.text_utils import _is_error_bearing


def _compact_line_ranges(line_numbers: list[int], max_ranges: int = 8) -> str:
    """
    Compact original log line numbers into readable ranges.

    Args:
        line_numbers: Original 1-based line numbers
        max_ranges: Maximum ranges to display before summarising

    Returns:
        Compact range string such as "10-14, 20, 25-28"
    """
    unique_numbers = sorted({int(n) for n in line_numbers if int(n) > 0})
    if not unique_numbers:
        return ''

    ranges: list[tuple[int, int]] = []
    start = unique_numbers[0]
    previous = unique_numbers[0]

    for number in unique_numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = number
        previous = number
    ranges.append((start, previous))

    visible_ranges = ranges[:max_ranges]
    parts = [
        str(begin) if begin == end else f"{begin}-{end}"
        for begin, end in visible_ranges
    ]
    if len(ranges) > max_ranges:
        parts.append(f"... +{len(ranges) - max_ranges} ranges")
    return ', '.join(parts)


def _error_line_ranges_from_numbered_lines(
    numbered_lines: list[tuple[int, str]],
    error_keywords: Optional[list[str]] = None,
) -> str:
    """
    Return compact line ranges for error-bearing lines only.

    Args:
        numbered_lines: Original line number and text pairs
        error_keywords: Optional error keyword override

    Returns:
        Compact range string for matching error lines
    """
    severity_line_numbers = [
        line_number
        for line_number, text in numbered_lines
        if any(marker in text for marker in ("ERROR", "FATAL", "CRITICAL"))
    ]
    if severity_line_numbers:
        return _compact_line_ranges(severity_line_numbers)

    error_line_numbers = [
        line_number
        for line_number, text in numbered_lines
        if _is_error_bearing(text, error_keywords)
    ]
    return _compact_line_ranges(error_line_numbers)


def _line_reference_from_metadata(metadata: dict[str, Any]) -> str:
    """
    Build a human-readable original line reference from chunk metadata.

    Args:
        metadata: Document metadata

    Returns:
        Human-readable line reference
    """
    error_ranges = str(metadata.get('error_line_ranges', '')).strip()
    if error_ranges:
        return f"lines {error_ranges}"

    compact_ranges = str(metadata.get('line_ranges', '')).strip()
    if compact_ranges:
        return f"lines {compact_ranges}"

    start_line = metadata.get('start_line')
    end_line = metadata.get('end_line')
    if isinstance(start_line, int) and isinstance(end_line, int):
        return f"line {start_line}" if start_line == end_line else f"lines {start_line}-{end_line}"
    return "lines unavailable"


def _build_vscode_file_uri(file_path: str, start_line: Optional[int] = None) -> str:
    """
    Build a VS Code URI that opens a local file at a line when supported.

    Args:
        file_path: Local file path
        start_line: Optional 1-based line number

    Returns:
        vscode:// URI
    """
    normalized = os.path.abspath(file_path).replace('\\', '/')
    encoded_path = urllib.parse.quote(normalized, safe='/:')
    suffix = f":{start_line}" if isinstance(start_line, int) and start_line > 0 else ''
    return f"vscode://file/{encoded_path}{suffix}"


def _build_file_uri(file_path: str) -> str:
    """
    Build a generic local file URI.

    Args:
        file_path: Local file path

    Returns:
        file:// URI
    """
    normalized = os.path.abspath(file_path).replace('\\', '/')
    encoded_path = urllib.parse.quote(normalized, safe='/:')
    return f"file:///{encoded_path}"


def _source_reference_from_doc(doc: Document, ref_id: str) -> dict[str, Any]:
    """
    Build a source-reference row for one selected evidence document.

    Args:
        doc: Selected evidence document
        ref_id: Chunk citation ID shown to the LLM

    Returns:
        JSON-safe source-reference row
    """
    metadata = doc.metadata
    source_file = str(metadata.get('source_file', 'unknown'))
    source_path = str(metadata.get('source_path', source_file))
    start_line = metadata.get('start_line')
    end_line = metadata.get('end_line')
    line_reference = _line_reference_from_metadata(metadata)
    vscode_uri = _build_vscode_file_uri(source_path, start_line if isinstance(start_line, int) else None)
    file_uri = _build_file_uri(source_path)

    return {
        'ref_id': ref_id,
        'chunk_reference': ref_id,
        'source_file': source_file,
        'source_path': source_path,
        'start_line': start_line,
        'end_line': end_line,
        'line_ranges': str(metadata.get('line_ranges', '')).strip(),
        'error_line_ranges': str(metadata.get('error_line_ranges', '')).strip(),
        'line_reference': line_reference,
        'vscode_uri': vscode_uri,
        'file_uri': file_uri,
    }


def _format_original_reference_markdown(ref: dict[str, Any]) -> list[str]:
    """
    Format a source-reference row as audience-facing Markdown lines.

    Args:
        ref: Source-reference row

    Returns:
        Markdown bullet lines with original log location only
    """
    source_file = str(ref.get('source_file', 'unknown')).strip()
    line_reference = str(ref.get('line_reference', 'lines unavailable')).strip()
    source_path = str(ref.get('source_path', '')).strip()
    vscode_uri = str(ref.get('vscode_uri', '')).strip()
    file_uri = str(ref.get('file_uri', '')).strip()
    link_uri = vscode_uri or file_uri
    label = f"{source_file}, {line_reference}"

    if link_uri:
        original_reference = f"[{label}]({link_uri})"
    else:
        original_reference = label

    path_toggle = f" [📄]({file_uri})" if file_uri and source_path else ""

    return [
        f"- Original Log Reference: {original_reference}{path_toggle}",
    ]


def _replace_chunk_refs_with_original_references(report_text: str, all_findings: list[dict]) -> str:
    """
    Replace visible chunk REF IDs with original log references.

    Args:
        report_text: Final report text from the reduce LLM
        all_findings: Per-file map findings with source reference rows

    Returns:
        Audience-facing report text with original log references inline
    """
    refs_by_id: dict[str, dict[str, Any]] = {}
    for finding in all_findings:
        for ref in finding.get('source_reference_map', []):
            ref_id = str(ref.get('ref_id', '')).strip()
            if ref_id and ref_id not in refs_by_id:
                refs_by_id[ref_id] = ref

    if not refs_by_id:
        return report_text

    citation_re = re.compile(r'\[(REF_[^\]]+)\]')
    output_lines: list[str] = []

    for raw_line in report_text.splitlines():
        matched_ref_ids = [
            match.group(1)
            for match in citation_re.finditer(raw_line)
            if match.group(1) in refs_by_id
        ]
        if not matched_ref_ids:
            output_lines.append(raw_line)
            continue

        ordered_matches: list[str] = []
        for ref_id in matched_ref_ids:
            if ref_id not in ordered_matches:
                ordered_matches.append(ref_id)

        cleaned_line = citation_re.sub(
            lambda match: "" if match.group(1) in refs_by_id else match.group(0),
            raw_line,
        )

        cleaned_line = re.sub(r'\s+([,.;:])', r'\1', cleaned_line)
        cleaned_line = re.sub(r'\s{2,}', ' ', cleaned_line).strip()
        if cleaned_line:
            output_lines.append(cleaned_line)

        seen_sources: set[tuple[str, str]] = set()
        for ref_id in ordered_matches:
            ref = refs_by_id[ref_id]
            source_key = (
                str(ref.get('source_path', '')),
                str(ref.get('line_reference', '')),
            )
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            output_lines.extend(_format_original_reference_markdown(ref))

    return '\n'.join(output_lines).strip() + '\n'
