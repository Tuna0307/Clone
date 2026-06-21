"""
Streamlit front end for the IAM log analysis pipeline.

The app accepts either uploaded log files or a local file/folder path, converts
optional calendar/time controls into pipeline query bounds, runs
hybridChunking2, and keeps the generated artifacts available for follow-up chat.
"""

from __future__ import annotations

import html
import os
import re
import urllib.parse
import uuid
from datetime import date, time

import streamlit as st

from artifact_paths import upload_session_dir

from followup.server_sql import close_server_monitoring_connections, load_temp_duckdb_into_session
from followup_retrieval import (
    answer_analysis_results_query,
    build_analysis_context,
    build_coverage_summary_table_data,
    build_retrieved_chunks_table_data,
)
from iam_log_intelligence_agent_hybridChunking2 import build_query_context, run_pipeline
from pipeline.progress import format_progress_details_block, progress_callback_scope
from log_viewer import build_log_reference_key, parse_line_reference_start, read_log_line_window
from ui_time_utils import format_optional_datetime
from upload_utils import save_uploaded_files


st.set_page_config(page_title="IAM Log Analysis", layout="wide")
st.title("IAM Log Analysis")
st.markdown(
    """
    <style>
    .log-ref-line {
        line-height: 1.55;
        margin: 0.25rem 0;
    }
    .log-path-toggle {
        display: inline-block;
        margin-left: 0.25rem;
        position: relative;
        vertical-align: middle;
    }
    .log-path-toggle > summary {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 4px;
        cursor: pointer;
        display: inline-flex;
        line-height: 1.2rem;
        list-style: none;
        padding: 0 0.25rem;
        user-select: none;
    }
    .log-path-toggle > summary::-webkit-details-marker {
        display: none;
    }
    .log-path-popover {
        background: rgb(255, 255, 255);
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        color: rgb(49, 51, 63);
        left: 0;
        max-width: min(72vw, 46rem);
        overflow-wrap: anywhere;
        padding: 0.5rem 0.65rem;
        position: absolute;
        top: 1.6rem;
        white-space: normal;
        z-index: 1000;
    }
    .pipeline-progress-details {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 6px;
        margin: 0.35rem 0 0.85rem 0;
        padding: 0.35rem 0.55rem;
    }
    .pipeline-progress-details > summary {
        cursor: pointer;
        font-weight: 600;
        user-select: none;
    }
    .pipeline-progress-details > pre {
        background: rgba(49, 51, 63, 0.04);
        border-radius: 4px;
        font-size: 0.82rem;
        line-height: 1.45;
        margin: 0.55rem 0 0.15rem 0;
        max-height: 18rem;
        overflow: auto;
        padding: 0.55rem 0.65rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None

if "server_monitoring_conns" not in st.session_state:
    st.session_state.server_monitoring_conns = {}

if "pending_ui_command" not in st.session_state:
    st.session_state.pending_ui_command = ""

if "show_retrieved_chunks" not in st.session_state:
    st.session_state.show_retrieved_chunks = False

if "show_coverage_summary" not in st.session_state:
    st.session_state.show_coverage_summary = False

if "active_log_reference" not in st.session_state:
    st.session_state.active_log_reference = None

if "active_log_reference_key" not in st.session_state:
    st.session_state.active_log_reference_key = ""

if "upload_session_id" not in st.session_state:
    st.session_state.upload_session_id = uuid.uuid4().hex

if "evidence_summary_expanded" not in st.session_state:
    st.session_state.evidence_summary_expanded = {}

with st.sidebar:
    st.header("Analysis Input")
    input_source = st.radio(
        "Input source",
        ["Upload files", "Local path"],
        horizontal=True,
    )
    uploaded_files = []
    log_path = ""
    analysis_scope_label = ""

    if input_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload log files",
            type=["log", "txt", "out", "err", "msg"],
            accept_multiple_files=True,
        )
        uploaded_names = [uploaded_file.name for uploaded_file in uploaded_files]
        analysis_scope_label = "uploaded:" + "|".join(uploaded_names)
        if uploaded_names:
            st.caption(f"{len(uploaded_names)} file(s) selected.")
    else:
        log_path = st.text_input("Log file or folder path", value="")
        analysis_scope_label = log_path.strip()

    st.subheader("Incident Window")
    use_start_time = st.checkbox("Use start time", value=False)
    start_date = st.date_input("Start date", value=date.today(), disabled=not use_start_time)
    start_clock = st.time_input("Start time", value=time(0, 0), disabled=not use_start_time)
    start_time_text = format_optional_datetime(use_start_time, start_date, start_clock)

    use_end_time = st.checkbox("Use end time", value=False)
    end_date = st.date_input("End date", value=date.today(), disabled=not use_end_time)
    end_clock = st.time_input("End time", value=time(23, 59), disabled=not use_end_time)
    end_time_text = format_optional_datetime(use_end_time, end_date, end_clock)

    st.subheader("Analysis Mode")
    analysis_mode = st.radio(
        "Choose analysis path",
        ["API Request (default)", "Server Monitoring (DuckDB + iterative SQL)"],
        horizontal=False,
        help="Server Monitoring mode loads metrics into DuckDB and lets the analyst run iterative SQL queries. Recommended for UAM server statistics / resource logs.",
    )
    mode_value = "server_monitoring" if "Server Monitoring" in analysis_mode else "api_request"
    if mode_value == "server_monitoring":
        st.caption("Follow-up uses agentic SQL against in-memory DuckDB tables for this session.")

    # Dedicated ticket uploader (only its content is used when mode == server_monitoring).
    # No automatic scanning of local paths or upload directories is performed.
    ticket_up = st.file_uploader(
        "Attach support ticket (optional — .md/.txt; sent to the agent for post-report SQL refinement in Server Monitoring mode only)",
        type=["md", "txt", "markdown"],
        accept_multiple_files=False,
        key="ticket_uploader",
    )
    if ticket_up is not None:
        try:
            st.session_state.ticket_text = ticket_up.getvalue().decode("utf-8", "replace")
            st.caption(f"Ticket attached: {ticket_up.name} ({len(st.session_state.ticket_text)} chars)")
        except Exception as te:
            st.session_state.ticket_text = None
            st.warning(f"Could not read ticket file: {te}")
    else:
        # Keep previous value if user is just re-running without re-selecting the file
        if "ticket_text" not in st.session_state:
            st.session_state.ticket_text = None

    if mode_value != "server_monitoring":
        st.divider()
        st.header("Analysis Results")
        active_context = st.session_state.analysis_context
        is_api_context = bool(
            active_context is not None
            and any(getattr(entry, "category", "") == "api_request" for entry in active_context.entries)
        )
        if active_context is None:
            st.caption("No completed analysis context yet.")
        else:
            st.caption(
                "Active context loaded from latest analysis. "
                "Follow-up prompts use artifacts and do not rerun pipeline by default."
            )
            view_button_label = "View Retrieved Rows" if is_api_context else "View Retrieved Chunks"
            if st.button(view_button_label):
                st.session_state.pending_ui_command = "/show metadata"
            if st.button("View Coverage Summary"):
                st.session_state.pending_ui_command = "/show coverage"

st.caption("Use the chat box to submit incident queries. The app validates time windows and runs analysis using the selected Analysis Mode.")


_REFERENCE_LINE_RE = re.compile(
    r"^(?P<prefix>\s*-\s*Original Log Reference:\s*)"
    r"(?P<reference>\[[^\]]+\]\([^)]+\)|.+?)"
    r"(?:\s+\[📄\]\((?P<file_uri>[^)]+)\))?\s*$"
)
_MARKDOWN_LINK_RE = re.compile(r"^\[(?P<label>[^\]]+)\]\((?P<href>[^)]+)\)$")
_PATH_LINE_RE = re.compile(r"^\s*-?\s*Path:\s*`?(?P<path>.+?)`?\s*$")
_EVIDENCE_SUMMARIES_HEADER = "## File-Wide Evidence Summaries"
_EVIDENCE_SUMMARIES_END_HEADER = "## Consolidated Analysis Boundaries & Uncertainty"
_PER_FILE_EVIDENCE_SUMMARY_HEADER = "## 1. File-Wide Evidence Summary"
_PER_FILE_EVIDENCE_SUMMARY_END_HEADER = "## 2. Analysis Boundaries & Uncertainty"


def _normalize_report_heading(line: str) -> str:
    """Normalize a markdown heading line for exact comparisons."""
    return line.strip()


def _heading_label(line: str) -> str:
    """Strip leading markdown heading markers from a line."""
    return _normalize_report_heading(line).lstrip("#").strip()


def _match_evidence_summary_section(line: str) -> tuple[str, str] | None:
    """
    Match a collapsible evidence-summary header and its section end marker.

    Supports API consolidated reports, API map per-file sections, and server
    monitoring outputs (including ``# 1. File-Wide Evidence Summary`` variants).

    Returns:
        (canonical_start_header, end_header) when matched, else None
    """
    label = _heading_label(line)
    if label == "File-Wide Evidence Summaries":
        return _EVIDENCE_SUMMARIES_HEADER, _EVIDENCE_SUMMARIES_END_HEADER
    if label.startswith("1. File-Wide Evidence Summary"):
        return _PER_FILE_EVIDENCE_SUMMARY_HEADER, _PER_FILE_EVIDENCE_SUMMARY_END_HEADER
    return None


def _is_evidence_summary_end(line: str, end_header: str) -> bool:
    """Return True when the line closes a collapsible evidence-summary section."""
    return _heading_label(line) == _heading_label(end_header)


def _find_evidence_summary_end_index(
    lines: list[str],
    start_index: int,
    end_header: str,
) -> int:
    """
    Locate the first line index after the evidence summary that begins the next section.

    Args:
        lines: Full assistant message split into lines
        start_index: Index of the evidence-summary header line
        end_header: Heading text that ends the collapsible block

    Returns:
        Index of the next-section header, or len(lines) when it never appears
    """
    for idx in range(start_index + 1, len(lines)):
        if _is_evidence_summary_end(lines[idx], end_header):
            return idx
    return len(lines)


def _evidence_summary_state_key(key_prefix: str, occurrence_index: int) -> str:
    """Build a stable session-state key for an evidence-summary collapse toggle."""
    return f"{key_prefix}_evidence_summary_{occurrence_index}"


def _is_evidence_summary_expanded(state_key: str) -> bool:
    """Return whether an evidence-summary block is expanded (default collapsed)."""
    expanded = st.session_state.get("evidence_summary_expanded", {})
    if not isinstance(expanded, dict):
        return False
    return bool(expanded.get(state_key, False))


def _set_evidence_summary_expanded(state_key: str, value: bool) -> None:
    """Persist expand/collapse state for an evidence-summary block."""
    expanded = st.session_state.get("evidence_summary_expanded")
    if not isinstance(expanded, dict):
        expanded = {}
        st.session_state.evidence_summary_expanded = expanded
    expanded[state_key] = value


def _path_from_file_uri(file_uri: str) -> str:
    """
    Convert a file:// URI back to a readable local path for the path toggle.

    Args:
        file_uri: Encoded local file URI

    Returns:
        Display-ready local path, or an empty string when unavailable
    """
    parsed = urllib.parse.urlparse(file_uri)
    if parsed.scheme != "file":
        return ""

    path = urllib.parse.unquote(parsed.path)
    if re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    if re.match(r"^[A-Za-z]:/", path):
        path = path.replace("/", "\\")
    return path


def _reference_line_html(line: str, hidden_path: str = "") -> str | None:
    """
    Render an Original Log Reference line with an inline path toggle.

    Args:
        line: Markdown report line
        hidden_path: Optional source path from a legacy following Path line

    Returns:
        HTML for Streamlit rendering, or None if the line is not a reference line
    """
    match = _REFERENCE_LINE_RE.match(line)
    if not match:
        return None

    prefix = html.escape(match.group("prefix"))
    reference = match.group("reference").strip()
    link_match = _MARKDOWN_LINK_RE.match(reference)
    if link_match:
        label = html.escape(link_match.group("label"))
        href = html.escape(link_match.group("href"), quote=True)
        reference_html = f'<a href="{href}">{label}</a>'
    else:
        reference_html = html.escape(reference)

    file_uri = (match.group("file_uri") or "").strip()
    display_path = hidden_path or _path_from_file_uri(file_uri)
    toggle_html = ""
    if display_path:
        escaped_path = html.escape(display_path)
        toggle_html = (
            '<details class="log-path-toggle">'
            '<summary title="Show path">&#128196;</summary>'
            f'<div class="log-path-popover">Path: <code>{escaped_path}</code></div>'
            '</details>'
        )
    elif file_uri:
        escaped_file_uri = html.escape(file_uri, quote=True)
        toggle_html = f' <a href="{escaped_file_uri}" title="Open file">&#128196;</a>'

    return f'<div class="log-ref-line">{prefix}{reference_html} {toggle_html}</div>'


def _reference_view_target(line: str, hidden_path: str = "") -> dict[str, str | int] | None:
    """
    Extract an in-app source-log viewer target from an Original Log Reference line.

    Args:
        line: Markdown report line
        hidden_path: Optional source path from a legacy following Path line

    Returns:
        Dict with source path and target line, or None when unavailable
    """
    match = _REFERENCE_LINE_RE.match(line)
    if not match:
        return None

    reference = match.group("reference").strip()
    link_match = _MARKDOWN_LINK_RE.match(reference)
    reference_label = link_match.group("label") if link_match else reference

    file_uri = (match.group("file_uri") or "").strip()
    display_path = hidden_path or _path_from_file_uri(file_uri)
    target_line = parse_line_reference_start(reference_label)
    if not display_path or target_line is None:
        return None

    return {
        "source_path": display_path,
        "target_line": target_line,
        "reference_label": reference_label,
    }


def _reference_button_key(
    view_target: dict[str, str | int],
    index: int,
    key_prefix: str,
) -> str:
    """
    Build a stable Streamlit key for a citation viewer button.

    Args:
        view_target: Parsed source-log target
        index: Current line index in the rendered assistant message
        key_prefix: Caller-provided message scope

    Returns:
        Streamlit widget key
    """
    return build_log_reference_key(
        str(view_target.get("source_path", "")),
        int(view_target.get("target_line", 0)),
        key_prefix,
        index,
    )


def _render_log_reference_window(
    active_ref: dict[str, str | int],
    close_key: str,
) -> None:
    """
    Render a selected source-log context window inline with its citation.
    """
    source_path = str(active_ref.get("source_path", "")).strip()
    target_line_raw = active_ref.get("target_line")
    try:
        target_line = int(target_line_raw)
    except Exception:
        target_line = 0

    if not source_path or target_line < 1:
        st.session_state.active_log_reference = None
        st.session_state.active_log_reference_key = ""
        return

    file_name = os.path.basename(source_path)
    with st.expander(f"Log source: {file_name}, line {target_line}", expanded=True):
        action_col, _ = st.columns([1, 5])
        with action_col:
            if st.button("Close", key=close_key):
                st.session_state.active_log_reference = None
                st.session_state.active_log_reference_key = ""
                st.rerun()

        if not os.path.exists(source_path):
            st.warning(f"Source log file is no longer available: {source_path}")
            return

        try:
            line_window = read_log_line_window(
                source_path,
                target_line=target_line,
                context_radius=5,
            )
        except Exception as error:
            st.warning(f"Could not read source log window: {error}")
            return

        if not line_window.lines:
            st.info(f"Line {target_line} was not found in {file_name}.")
            return

        rendered_lines = []
        for window_line in line_window.lines:
            marker = ">>>" if window_line.is_target else "   "
            rendered_lines.append(f"{marker} {window_line.line_number}: {window_line.text}")
        st.code("\n".join(rendered_lines), language="text")


_PROGRESS_DETAILS_RE = re.compile(
    r'^(<details class="pipeline-progress-details">.*?</details>)\s*(.*)$',
    re.DOTALL,
)


def _compose_message_with_progress(
    body: str,
    progress_lines: list[str],
    *,
    summary_label: str = "Pipeline progress",
) -> str:
    """Prepend a collapsible progress block above the assistant response body."""
    progress_block = format_progress_details_block(progress_lines, summary_label=summary_label)
    if not progress_block:
        return body
    return f"{progress_block}\n\n{body}"


def _render_assistant_content(content: str, key_prefix: str = "") -> None:
    """
    Render assistant content while hiding full log paths behind path toggles.

    ``File-Wide Evidence Summaries`` (consolidated) and ``1. File-Wide Evidence
    Summary`` (API map + server monitoring per-file) are collapsed by default
    with an inline Expand/Collapse control beside the heading.

    Args:
        content: Assistant message Markdown
        key_prefix: Widget key scope for this rendered message
    """
    details_match = _PROGRESS_DETAILS_RE.match(content)
    if details_match:
        st.markdown(details_match.group(1), unsafe_allow_html=True)
        content = details_match.group(2)

    lines = content.splitlines()
    markdown_buffer: list[str] = []
    index = 0
    evidence_summary_occurrence = 0

    def flush_markdown_buffer() -> None:
        if markdown_buffer:
            st.markdown("\n".join(markdown_buffer))
            markdown_buffer.clear()

    def render_lines_range(start: int, end: int) -> None:
        local_index = start
        while local_index < end:
            line = lines[local_index]
            path_match = (
                _PATH_LINE_RE.match(lines[local_index + 1])
                if local_index + 1 < end
                else None
            )
            hidden_path = path_match.group("path").strip() if path_match else ""
            reference_html = _reference_line_html(line, hidden_path=hidden_path)

            if reference_html is None:
                markdown_buffer.append(line)
                local_index += 1
                continue

            flush_markdown_buffer()
            st.markdown(reference_html, unsafe_allow_html=True)
            view_target = _reference_view_target(line, hidden_path=hidden_path)
            if view_target is not None:
                button_key = _reference_button_key(view_target, local_index, key_prefix)
                if st.button(
                    "View in app",
                    key=button_key,
                ):
                    st.session_state.active_log_reference = view_target
                    st.session_state.active_log_reference_key = button_key
                    st.rerun()
                active_ref = st.session_state.get("active_log_reference")
                active_key = str(st.session_state.get("active_log_reference_key", ""))
                if active_key == button_key and isinstance(active_ref, dict):
                    _render_log_reference_window(
                        active_ref,
                        close_key=f"close_{button_key}",
                    )
            local_index += 2 if path_match else 1

    while index < len(lines):
        section_match = _match_evidence_summary_section(lines[index])
        if section_match is not None:
            flush_markdown_buffer()
            _, end_header = section_match
            end_index = _find_evidence_summary_end_index(lines, index, end_header)
            state_key = _evidence_summary_state_key(key_prefix, evidence_summary_occurrence)
            expanded = _is_evidence_summary_expanded(state_key)

            header_col, button_col = st.columns([8, 1])
            with header_col:
                st.markdown(lines[index])
            with button_col:
                toggle_label = "Collapse" if expanded else "Expand"
                if st.button(toggle_label, key=f"toggle_{state_key}"):
                    _set_evidence_summary_expanded(state_key, not expanded)
                    st.rerun()

            body_start = index + 1
            if expanded and body_start < end_index:
                render_lines_range(body_start, end_index)

            evidence_summary_occurrence += 1
            index = end_index
            continue

        line = lines[index]
        path_match = _PATH_LINE_RE.match(lines[index + 1]) if index + 1 < len(lines) else None
        hidden_path = path_match.group("path").strip() if path_match else ""
        reference_html = _reference_line_html(line, hidden_path=hidden_path)

        if reference_html is None:
            markdown_buffer.append(line)
            index += 1
            continue

        flush_markdown_buffer()
        st.markdown(reference_html, unsafe_allow_html=True)
        view_target = _reference_view_target(line, hidden_path=hidden_path)
        if view_target is not None:
            button_key = _reference_button_key(view_target, index, key_prefix)
            if st.button(
                "View in app",
                key=button_key,
            ):
                st.session_state.active_log_reference = view_target
                st.session_state.active_log_reference_key = button_key
                st.rerun()
            active_ref = st.session_state.get("active_log_reference")
            active_key = str(st.session_state.get("active_log_reference_key", ""))
            if active_key == button_key and isinstance(active_ref, dict):
                _render_log_reference_window(
                    active_ref,
                    close_key=f"close_{button_key}",
                )
        index += 2 if path_match else 1

    flush_markdown_buffer()


def _run_with_live_progress(
    task,
    *,
    summary_label: str = "Pipeline progress",
    render_final: bool = False,
    key_prefix: str = "",
):
    """
    Run a blocking task while streaming progress lines into the assistant chat.

    Returns:
        Tuple of (task result, accumulated progress lines).
    """
    progress_lines: list[str] = []

    with st.chat_message("assistant"):
        progress_placeholder = st.empty()

        def render_progress() -> None:
            if progress_lines:
                progress_placeholder.code("\n".join(progress_lines))

        def on_progress_line_live(message: str) -> None:
            progress_lines.append(message)
            render_progress()

        with progress_callback_scope(on_progress_line_live):
            result = task()

        if render_final:
            progress_placeholder.empty()
            body = result if isinstance(result, str) else str(result)
            combined = _compose_message_with_progress(
                body,
                progress_lines,
                summary_label=summary_label,
            )
            _render_assistant_content(combined, key_prefix=key_prefix or f"new_{len(st.session_state.messages)}")

    return result, progress_lines


def _push_assistant_message(content: str) -> None:
    """
    Render and persist assistant message.

    Args:
        content: Assistant text
    """
    with st.chat_message("assistant"):
        _render_assistant_content(content, key_prefix=f"new_{len(st.session_state.messages)}")
    st.session_state.messages.append({"role": "assistant", "content": content})


pending_command = st.session_state.pending_ui_command.strip().lower()
if pending_command:
    st.session_state.pending_ui_command = ""
    context = st.session_state.analysis_context
    if context is None:
        _push_assistant_message("No analysis context is available yet. Run a full analysis first.")
    elif pending_command == "/show metadata":
        st.session_state.show_retrieved_chunks = True
    elif pending_command == "/show coverage":
        st.session_state.show_coverage_summary = True


if st.session_state.show_coverage_summary:
    context = st.session_state.analysis_context
    is_server_only_context = (
        context is not None
        and context.entries
        and all(getattr(e, "category", "") == "server_monitoring" for e in context.entries)
    )
    if context is not None and not is_server_only_context:
        coverage_data = build_coverage_summary_table_data(context)
        file_rows = coverage_data.get("files", [])
        bucket_rows = coverage_data.get("buckets", [])

        with st.expander("Coverage Summary", expanded=True):
            action_col, _ = st.columns([1, 6])
            with action_col:
                if st.button("Hide", key="hide_coverage_summary"):
                    st.session_state.show_coverage_summary = False
                    st.rerun()

            if not file_rows:
                st.info("No coverage summary is available for the current analysis context.")
            else:
                st.caption(
                    "This shows file scan coverage and where selected evidence came from. "
                    "The LLM sees selected evidence, not the full raw log."
                )
                st.subheader("Files Scanned")
                st.dataframe(file_rows, use_container_width=True, height=180)

                if bucket_rows:
                    st.subheader("Selected Evidence by Line Range")
                    st.dataframe(bucket_rows, use_container_width=True, height=240)


if st.session_state.show_retrieved_chunks:
    context = st.session_state.analysis_context
    is_server_only_context = (
        context is not None
        and context.entries
        and all(getattr(e, "category", "") == "server_monitoring" for e in context.entries)
    )
    if context is not None and not is_server_only_context:
        is_api_context = any(getattr(entry, "category", "") == "api_request" for entry in context.entries)
        panel_data = build_retrieved_chunks_table_data(context, reduce_only=is_api_context)
        rows = panel_data.get("rows", [])
        summary = panel_data.get("summary", {})
        ordered_columns = panel_data.get("columns", [])

        panel_title = "Retrieved Rows" if is_api_context else "Retrieved Chunks"
        with st.expander(panel_title, expanded=True):
            action_col, _ = st.columns([1, 6])
            with action_col:
                if st.button("Hide", key="hide_retrieved_chunks"):
                    st.session_state.show_retrieved_chunks = False
                    st.rerun()

            if not rows:
                if is_api_context:
                    st.info("No rows were sent to the reduce-phase path for the current analysis context.")
                else:
                    st.info("No metadata rows found for current analysis context.")
            else:
                metric_col_1, _ = st.columns([1, 6])
                metric_label = "Total rows" if is_api_context else "Total chunks"
                metric_col_1.metric(metric_label, int(summary.get("total_chunks", 0)))

                hidden_columns = {
                    "chunk_index",
                    "artifact_file",
                    "sub_index",
                    "request_span_id",
                }
                if is_api_context:
                    hidden_columns.update({"raw_distance", "request_key", "request_level"})
                priority_columns = [
                    "source_file",
                    "primary_key",
                    "key_type",
                    "content",
                    "session_labels",
                    "session_label_count",
                    "line_count",
                    "start_time",
                    "end_time",
                ]
                display_label_map = {
                    "source_file": "Source File",
                    "primary_key": "Primary key",
                    "key_type": "Key type",
                    "content": "Content",
                    "session_labels": "session label",
                    "session_label_count": "session label count",
                    "line_count": "line count",
                    "start_time": "start time",
                    "end_time": "end time",
                }

                effective_order = list(ordered_columns) if ordered_columns else list(rows[0].keys())
                visible_columns: list[str] = []
                seen_columns: set[str] = set()
                candidate_columns = priority_columns + effective_order

                for column in candidate_columns:
                    if column in hidden_columns or column in seen_columns:
                        continue
                    if any(column in row for row in rows):
                        visible_columns.append(column)
                        seen_columns.add(column)

                display_rows = []
                for row in rows:
                    display_row = {}
                    for column in visible_columns:
                        display_name = display_label_map.get(column, column)
                        display_row[display_name] = row.get(column, "")
                    display_rows.append(display_row)

                st.dataframe(display_rows, use_container_width=True, height=460)


for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_content(
                message["content"],
                key_prefix=f"message_{message_index}",
            )
        else:
            st.markdown(message["content"])

user_query = st.chat_input("Describe the incident...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    stripped_query = user_query.strip()

    context = st.session_state.analysis_context
    same_scope_as_context = (
        context is not None
        and str(context.log_path).strip() == analysis_scope_label.strip()
        and str(context.start_time).strip() == start_time_text
        and str(context.end_time).strip() == end_time_text
    )

    if same_scope_as_context:
        from followup.server_sql import is_server_monitoring_followup_mode

        if is_server_monitoring_followup_mode(context):
            followup_answer, progress_lines = _run_with_live_progress(
                lambda: answer_analysis_results_query(
                    context=context,
                    query=stripped_query,
                    chat_history=st.session_state.messages,
                ),
                summary_label="Follow-up SQL progress",
                render_final=True,
                key_prefix=f"new_{len(st.session_state.messages)}",
            )
            combined_answer = _compose_message_with_progress(
                followup_answer,
                progress_lines,
                summary_label="Follow-up SQL progress",
            )
            st.session_state.messages.append({"role": "assistant", "content": combined_answer})
        else:
            followup_answer = answer_analysis_results_query(
                context=context,
                query=stripped_query,
                chat_history=st.session_state.messages,
            )
            _push_assistant_message(followup_answer)
    else:
        effective_query = stripped_query
        pipeline_input_path = ""
        if input_source == "Upload files":
            if not uploaded_files:
                _push_assistant_message("Please upload at least one log file in the sidebar before running analysis.")
            else:
                upload_dir = os.path.abspath(upload_session_dir(str(st.session_state.upload_session_id)))
                saved_paths = save_uploaded_files(uploaded_files, upload_dir)
                if not saved_paths:
                    _push_assistant_message("No supported log files were uploaded. Use .log, .txt, .out, .err, or .msg files.")
                else:
                    pipeline_input_path = upload_dir
        elif not log_path.strip():
            _push_assistant_message("Please provide a log path in the sidebar before running analysis.")
        else:
            pipeline_input_path = log_path.strip()

        if pipeline_input_path:
            query_context = build_query_context(
                query_text=effective_query,
                start_time=start_time_text,
                end_time=end_time_text,
            )

            parse_errors = query_context.get("time_parse_errors", [])
            if parse_errors:
                error_text = (
                    "Invalid incident time input. Supported examples: "
                    "`2025-09-12 15:00:00`, `2025-09-12T15:00:00`, "
                    "`10/03/2026 15:00`, `10/03/2026`, `2025-09-12`.\n\n"
                    + "\n".join(f"- {item}" for item in parse_errors)
                )
                _push_assistant_message(error_text)
            else:
                effective_ticket = st.session_state.get("ticket_text") if mode_value == "server_monitoring" else None
                pipeline_result, progress_lines = _run_with_live_progress(
                    lambda: run_pipeline(
                        pipeline_input_path,
                        query_context=query_context,
                        return_metadata=True,
                        mode=mode_value,
                        ticket_text=effective_ticket,
                    ),
                    summary_label="Pipeline progress",
                )

                if isinstance(pipeline_result, dict):
                    status = pipeline_result.get("status", "error")
                    if status == "invalid_query_window":
                        warning_text = str(
                            pipeline_result.get(
                                "message",
                                "Invalid date/time window for selected logs.",
                            )
                        )
                        _push_assistant_message(warning_text)
                    elif status == "ok":
                        result = str(pipeline_result.get("report", ""))
                        per_file_reports = pipeline_result.get("per_file_reports", [])
                        try:
                            effective_ticket = st.session_state.get("ticket_text") if mode_value == "server_monitoring" else None
                            if mode_value == "server_monitoring":
                                close_server_monitoring_connections(
                                    st.session_state.get("server_monitoring_conns")
                                )
                            st.session_state.analysis_context = build_analysis_context(
                                query_text=effective_query,
                                log_path=analysis_scope_label.strip(),
                                start_time=start_time_text,
                                end_time=end_time_text,
                                report_text=result,
                                per_file_reports=per_file_reports if isinstance(per_file_reports, list) else [],
                                ticket_text=effective_ticket or "",
                            )
                            if mode_value == "server_monitoring":
                                load_temp_duckdb_into_session(
                                    per_file_reports if isinstance(per_file_reports, list) else [],
                                    st.session_state,
                                )
                            else:
                                st.session_state.server_monitoring_conns = {}
                        except Exception as context_error:
                            st.session_state.analysis_context = None
                            close_server_monitoring_connections(
                                st.session_state.get("server_monitoring_conns")
                            )
                            st.session_state.server_monitoring_conns = {}
                            st.warning(f"Follow-up context build failed: {context_error}")
                        combined_result = _compose_message_with_progress(
                            result,
                            progress_lines,
                            summary_label="Pipeline progress",
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": combined_result}
                        )
                        st.rerun()
                    else:
                        error_text = str(pipeline_result.get("message", "Analysis failed."))
                        _push_assistant_message(error_text)
                else:
                    result = str(pipeline_result)
                    _push_assistant_message(result)
