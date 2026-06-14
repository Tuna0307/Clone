"""Dataclasses and context-builders for artifact-first follow-up interactions.

This module holds the core data structures and metadata helpers extracted from
`followup_retrieval.py` so that the retrieval orchestration layer stays thin.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from artifact_paths import debug_evidence_path, faiss_index_dir


FOLLOWUP_HIGH_ANOMALY_THRESHOLD = 2.5


@dataclass
class ArtifactEntry:
    """
    Per-file artifact references produced by the pipeline.

    Args:
        file_name: Base file name
        source_path: Absolute source log file path
        faiss_index_dir: Directory path to FAISS artifacts
        debug_evidence_file: Path to debug evidence text file
        sql_trace_file: Path to structured agentic SQL trace (JSONL) for server_monitoring runs — contains the full step-by-step reasoning + every SQL executed + observations
        metadata_rows: Chunk-level metadata rows from the pipeline
        selected_row_ids_for_reduce: IDs passed to reduce phase
        category: Detected log category
        subcategory: Detected log subcategory
        evidence_profile: File-wide deterministic profile from the pipeline
    """

    file_name: str
    source_path: str
    faiss_index_dir: str
    debug_evidence_file: str
    metadata_rows: list[dict[str, Any]]
    selected_row_ids_for_reduce: list[str]
    category: str
    subcategory: str
    evidence_profile: dict[str, Any] = field(default_factory=dict)

    # Optional server_monitoring fields — MUST come after all non-default fields
    sql_trace_file: Optional[str] = None
    duckdb_row_count: int = 0
    log_event_row_count: int = 0

    @property
    def metadata_json_path(self) -> str:
        """
        Return metadata JSON path for this artifact.

        Returns:
            Path to metadata.json
        """
        return os.path.join(self.faiss_index_dir, "metadata.json")


@dataclass
class AnalysisContext:
    """
    Session-scoped context for artifact-first follow-up interactions.

    Args:
        query_text: Original analysis query
        log_path: Input path used for analysis
        start_time: Optional start time input text
        end_time: Optional end time input text
        report_text: Final report text
        entries: Artifact entries for each processed file
        created_at: UTC timestamp string
        ticket_text: Optional support ticket text (used for server_monitoring follow-ups)
    """

    query_text: str
    log_path: str
    start_time: str
    end_time: str
    report_text: str
    entries: list[ArtifactEntry]
    created_at: str
    ticket_text: str = ""


@dataclass
class FollowupIntent:
    """
    Structured intent extracted by the follow-up LLM parser.

    Args:
        ask_type: High-level intent class
        entities: Important terms/entities referenced by user
        primary_keys: Optional primary key constraints
        must_include: Required concepts or keywords
        confidence: Intent extraction confidence in [0,1]
        notes: Additional parser rationale
    """

    ask_type: str
    entities: list[str]
    primary_keys: list[str]
    must_include: list[str]
    confidence: float
    notes: str


@dataclass
class EvidenceItem:
    """
    Normalized evidence item across all retrieval sources.

    Args:
        evidence_id: Stable evidence identifier
        source: Evidence source name
        file_name: Source file name
        relevance: Retrieval relevance score
        anomaly_score: Optional anomaly score
        excerpt: Human-readable snippet
        raw_text: Full text sent to LLM
    """

    evidence_id: str
    source: str
    file_name: str
    relevance: float
    anomaly_score: float
    excerpt: str
    raw_text: str


def _safe_abspath(path: str) -> str:
    """
    Resolve to absolute path when possible.

    Args:
        path: Candidate path

    Returns:
        Absolute path string
    """
    if not path:
        return ""
    return os.path.abspath(path)


def build_analysis_context(
    query_text: str,
    log_path: str,
    start_time: str,
    end_time: str,
    report_text: str,
    per_file_reports: list[dict[str, Any]],
    ticket_text: str = "",
) -> AnalysisContext:
    """
    Build session follow-up context from pipeline metadata payload.

    Args:
        query_text: User query text
        log_path: Analysis input path
        start_time: Start time text
        end_time: End time text
        report_text: Final report text
        per_file_reports: Per-file report payload from run_pipeline metadata
        ticket_text: Optional support ticket text (for server_monitoring refinement + follow-ups)

    Returns:
        AnalysisContext object
    """
    entries: list[ArtifactEntry] = []
    for report in per_file_reports:
        file_name = str(report.get("file", "")).strip()
        if not file_name:
            continue

        source_path = _safe_abspath(str(report.get("source_path", "")).strip())
        raw_faiss_dir = str(report.get("faiss_index_dir", "")).strip()
        faiss_dir = "" if raw_faiss_dir.lower() in {"", "none", "null"} else _safe_abspath(raw_faiss_dir)
        debug_file = _safe_abspath(str(report.get("debug_evidence_file", "")).strip())

        if not source_path:
            source_path = _safe_abspath(file_name)
        if not faiss_dir:
            faiss_dir = _safe_abspath(faiss_index_dir(file_name))
        if not debug_file:
            debug_file = _safe_abspath(debug_evidence_path(file_name))

        report_rows = report.get("metadata_rows", [])
        metadata_rows = [row for row in report_rows if isinstance(row, dict)] if isinstance(report_rows, list) else []
        selected_row_ids_for_reduce = report.get("selected_row_ids_for_reduce", [])
        selected_row_ids = [
            str(row_id) for row_id in selected_row_ids_for_reduce
            if isinstance(row_id, (str, int, float)) and str(row_id).strip()
        ] if isinstance(selected_row_ids_for_reduce, list) else []
        raw_profile = report.get("evidence_profile", {})
        evidence_profile = raw_profile if isinstance(raw_profile, dict) else {}

        entries.append(
            ArtifactEntry(
                file_name=file_name,
                source_path=source_path,
                faiss_index_dir=faiss_dir,
                debug_evidence_file=debug_file,
                sql_trace_file=report.get("sql_trace_file"),
                duckdb_row_count=int(report.get("duckdb_row_count") or 0),
                log_event_row_count=int(report.get("log_event_row_count") or 0),
                metadata_rows=metadata_rows,
                selected_row_ids_for_reduce=selected_row_ids,
                category=str(report.get("category", "")).strip(),
                subcategory=str(report.get("subcategory", "")).strip(),
                evidence_profile=evidence_profile,
            )
        )

    return AnalysisContext(
        query_text=query_text,
        log_path=log_path,
        start_time=start_time,
        end_time=end_time,
        report_text=report_text,
        ticket_text=ticket_text or "",
        entries=entries,
        created_at=datetime.utcnow().isoformat(),
    )


def _try_parse_datetime(value: str) -> Optional[datetime]:
    """
    Parse timestamp text from metadata/query.

    Args:
        value: Datetime text

    Returns:
        Datetime or None
    """
    text = value.strip()
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        pass

    datetime_formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]
    for fmt in datetime_formats:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _load_metadata_rows(entry: ArtifactEntry) -> list[dict[str, Any]]:
    """
    Load metadata chunk rows for one entry.

    Args:
        entry: Artifact entry

    Returns:
        Metadata chunk list
    """
    return [row for row in entry.metadata_rows if isinstance(row, dict)]


def _as_float(value: Any) -> float:
    """
    Convert value to float with safe fallback.

    Args:
        value: Candidate numeric value

    Returns:
        Parsed float value or 0.0
    """
    try:
        return float(value)
    except Exception:
        return 0.0


def _markdown_table_cell(value: Any) -> str:
    """Escape content that would break a Markdown table cell."""
    return str(value).replace("\n", " ").replace("|", r"\|")


def build_retrieved_chunks_table_data(
    context: AnalysisContext,
    reduce_only: bool = False,
) -> dict[str, Any]:
    """
    Build full metadata table payload for retrieved chunks UI.

    Args:
        context: Analysis context
        reduce_only: If True, include only rows selected for map evidence

    Returns:
        Dict containing rows, summary stats, and ordered columns
    """
    all_rows: list[dict[str, Any]] = []
    ordered_columns: list[str] = []
    seen_columns: set[str] = set()
    total_rows_all = 0

    for entry in context.entries:
        rows = _load_metadata_rows(entry)
        total_rows_all += len(rows)
        selected_ids = set(entry.selected_row_ids_for_reduce)
        for row in rows:
            row_id = str(row.get("row_id", "")).strip()
            if reduce_only and (not row_id or row_id not in selected_ids):
                continue
            row_copy = dict(row)

            for key in row_copy:
                if key not in seen_columns:
                    seen_columns.add(key)
                    ordered_columns.append(key)

            all_rows.append(row_copy)

    all_rows.sort(key=lambda row: _as_float(row.get("anomaly_score", 0.0)), reverse=True)

    summary = {
        "total_chunks": len(all_rows),
        "total_rows_all": total_rows_all,
        "reduce_only": reduce_only,
    }

    return {
        "rows": all_rows,
        "summary": summary,
        "columns": ordered_columns,
    }


def _as_int(value: Any) -> Optional[int]:
    """
    Convert value to int with safe fallback.
    """
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _line_range_bounds(value: Any) -> tuple[Optional[int], Optional[int]]:
    """
    Parse compact line-range text into first/last line numbers.
    """
    text = str(value or "")
    if not text.strip():
        return None, None

    numbers: list[int] = []
    for start_text, end_text in re.findall(r"(\d+)(?:\s*-\s*(\d+))?", text):
        start = int(start_text)
        end = int(end_text) if end_text else start
        numbers.extend([start, end])

    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def _selected_row_line_bounds(row: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """
    Return the best line bounds for a selected evidence row.
    """
    for key in ("error_line_ranges", "line_ranges"):
        start, end = _line_range_bounds(row.get(key))
        if start is not None and end is not None:
            return start, end

    start_line = _as_int(row.get("start_line"))
    end_line = _as_int(row.get("end_line"))
    if start_line is not None and end_line is not None:
        return start_line, end_line
    if start_line is not None:
        return start_line, start_line
    if end_line is not None:
        return end_line, end_line
    return None, None


def _coverage_bucket_ranges(total_lines: int, bucket_count: int = 4) -> list[tuple[int, int]]:
    """
    Build even line-number buckets for coverage display.
    """
    if total_lines <= 0:
        return []

    bucket_count = max(1, min(bucket_count, total_lines))
    ranges: list[tuple[int, int]] = []
    for index in range(bucket_count):
        start = (index * total_lines) // bucket_count + 1
        end = ((index + 1) * total_lines) // bucket_count
        ranges.append((start, end))
    return ranges


def build_coverage_summary_table_data(context: AnalysisContext) -> dict[str, Any]:
    """
    Build per-file coverage summary data for Streamlit display.
    """
    file_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []

    for entry in context.entries:
        profile = entry.evidence_profile or {}
        profile_time_range = profile.get("time_range", {})
        time_range = profile_time_range if isinstance(profile_time_range, dict) else {}
        total_lines = _as_int(profile.get("total_lines"))
        selected_ids = set(entry.selected_row_ids_for_reduce)
        all_rows = _load_metadata_rows(entry)
        selected_rows = [
            row for row in all_rows
            if str(row.get("row_id", "")).strip() in selected_ids
        ]

        selected_bounds = [
            bounds for bounds in (_selected_row_line_bounds(row) for row in selected_rows)
            if bounds[0] is not None and bounds[1] is not None
        ]
        earliest = min((int(start) for start, _ in selected_bounds), default=None)
        latest = max((int(end) for _, end in selected_bounds), default=None)
        if total_lines is None:
            total_lines = max((int(end) for _, end in selected_bounds), default=0)

        file_rows.append(
            {
                "file_name": entry.file_name,
                "total_lines_scanned": total_lines,
                "timestamp_start": str(time_range.get("start", "")),
                "timestamp_end": str(time_range.get("end", "")),
                "metadata_rows_available": len(all_rows),
                "selected_evidence_items": len(selected_rows),
                "earliest_selected_line": earliest or "",
                "latest_selected_line": latest or "",
            }
        )

        buckets = _coverage_bucket_ranges(total_lines)
        counts = [0 for _ in buckets]
        for start, _ in selected_bounds:
            line_start = int(start)
            for index, (bucket_start, bucket_end) in enumerate(buckets):
                if bucket_start <= line_start <= bucket_end:
                    counts[index] += 1
                    break

        for (bucket_start, bucket_end), count in zip(buckets, counts):
            bucket_rows.append(
                {
                    "file_name": entry.file_name,
                    "line_range": f"{bucket_start}-{bucket_end}",
                    "selected_evidence_items": count,
                }
            )

    return {
        "files": file_rows,
        "buckets": bucket_rows,
    }


def build_analysis_results_metadata_markdown(context: AnalysisContext, top_k: int = 8) -> str:
    """
    Build markdown summary of metadata artifacts.

    Args:
        context: Analysis context
        top_k: Number of rows for top anomaly table

    Returns:
        Markdown summary
    """
    all_rows: list[dict[str, Any]] = []
    for entry in context.entries:
        rows = _load_metadata_rows(entry)
        for index, row in enumerate(rows):
            row_copy = dict(row)
            row_copy["_artifact_file"] = entry.file_name
            row_copy["_chunk_index"] = index
            all_rows.append(row_copy)

    if not all_rows:
        return "No metadata rows found in current analysis context."

    total = len(all_rows)
    high = [row for row in all_rows if _as_float(row.get("anomaly_score", 0.0)) > FOLLOWUP_HIGH_ANOMALY_THRESHOLD]
    iam = [row for row in all_rows if bool(row.get("iam_critical", False))]

    ranked = sorted(all_rows, key=lambda row: _as_float(row.get("anomaly_score", 0.0)), reverse=True)[:max(1, top_k)]

    lines = [
        "### Metadata Summary",
        f"- Total chunks: {total}",
        f"- High anomaly chunks (>{FOLLOWUP_HIGH_ANOMALY_THRESHOLD}): {len(high)}",
        f"- IAM-critical chunks: {len(iam)}",
        "",
        "### Top Anomalies",
        "| File | Score | Key | Time | Preview |",
        "|---|---:|---|---|---|",
    ]

    for row in ranked:
        preview = _markdown_table_cell(str(row.get("content", ""))[:120])
        score = _as_float(row.get("anomaly_score", 0.0))
        key = _markdown_table_cell(str(row.get("primary_key", "n/a"))[:40])
        time_range = _markdown_table_cell(
            f"{row.get('start_time', '')} -> {row.get('end_time', '')}".strip(" ->")[:32]
        )
        file_name = _markdown_table_cell(row.get("_artifact_file", ""))
        lines.append(f"| {file_name} | {score:.3f} | {key} | {time_range} | {preview} |")

    return "\n".join(lines)
