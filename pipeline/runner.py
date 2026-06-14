"""Pipeline runner: high-level orchestration for file discovery, map-reduce analysis, and interactive mode."""

import os
import sys
from typing import Any, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage

from artifact_paths import debug_evidence_path, faiss_index_dir
from llm_factory import get_llm
from pipeline.analysis import analyze_single_file
from pipeline.files import get_log_files_from_path
from pipeline.query import build_query_filter_summary
from pipeline.progress import ProgressCallback, emit_ui_progress, progress_callback_scope
from pipeline.reporting import consolidate_reports, export_to_pdf

llm = get_llm()


def run_pipeline(
    paths: Union[str, list[str]],
    query_context: Optional[dict[str, Any]] = None,
    return_metadata: bool = False,
    mode: str = "api_request",
    ticket_text: Optional[str] = None,
    progress_callback: ProgressCallback | None = None,
) -> Union[str, dict[str, Any]]:
    """
    Run the full Map-Reduce pipeline.

    The `mode` parameter is the user-controlled toggle for the new DuckDB
    server_monitoring path (see analyze_single_file for details).

    The optional `ticket_text` (support ticket / incident description) is only
    used when mode="server_monitoring". It is sent to the agent **after** the
    normal agentic SQL loop produces its report, triggering a refinement
    iteration pass so the LLM can iterate the SQL exploration to better
    address the symptoms and request described in the ticket.

    Args:
        paths: Path or list of paths to log files or directories.
        query_context: Optional query context for time-window filtering.
        return_metadata: When True, return structured status payload for UI handling.
        mode: "api_request" (default) or "server_monitoring" (DuckDB + agentic SQL).
        ticket_text: Optional full text of an attached support ticket (used only
            for server_monitoring post-report refinement; ignored otherwise).
        progress_callback: Optional Streamlit callback for live UI progress lines.

    Returns:
        Final forensic report string (default) or structured status payload.
    """
    with progress_callback_scope(progress_callback):
        return _run_pipeline_impl(
            paths=paths,
            query_context=query_context,
            return_metadata=return_metadata,
            mode=mode,
            ticket_text=ticket_text,
        )


def _run_pipeline_impl(
    paths: Union[str, list[str]],
    query_context: Optional[dict[str, Any]] = None,
    return_metadata: bool = False,
    mode: str = "api_request",
    ticket_text: Optional[str] = None,
) -> Union[str, dict[str, Any]]:
    if isinstance(paths, str):
        paths_list = [paths]
    else:
        paths_list = paths

    all_log_files: list[str] = []
    for path_input in paths_list:
        if os.path.exists(path_input):
            files = get_log_files_from_path(path_input)
            all_log_files.extend(files)
        else:
            print(f"Warning: Path not found (skipping): {path_input}")

    if not all_log_files:
        error_message = "Error: No log files found."
        if return_metadata:
            return {
                "status": "error",
                "message": error_message,
            }
        return error_message

    print(f"\n[Pipeline] Processing {len(all_log_files)} file(s)...\n")

    filter_summary = build_query_filter_summary(query_context)
    if filter_summary is not None:
        print(f"[Query] {filter_summary}")

    if ticket_text and mode == "server_monitoring":
        print(f"[Ticket] {len(ticket_text)} chars of support ticket context loaded — will be sent for post-report refinement iteration of the agentic SQL.")

    # 2. Map — analyse each file independently
    all_findings: list[dict] = []
    for i, f in enumerate(all_log_files):
        print(f"\n[Pipeline] File {i + 1}/{len(all_log_files)}")
        if len(all_log_files) > 1:
            emit_ui_progress(f"[File {i + 1}/{len(all_log_files)}] {os.path.basename(f)}")
        try:
            report = analyze_single_file(
                f, query_context=query_context, mode=mode, ticket_text=ticket_text
            )
            all_findings.append(report)
        except Exception as e:
            print(f"[Pipeline] Error processing {f}: {e}")
            all_findings.append(
                {
                    "file": os.path.basename(f),
                    "findings": "",
                    "chunk_count": 0,
                    "high_anomaly_count": 0,
                    "metadata_rows": [],
                    "status": "pipeline_error",
                    "query_valid": True,
                    "query_validation_reason": "pipeline_error",
                    "category": "unclassified",
                    "subcategory": "unclassified",
                    "source_path": f,
                    "faiss_index_dir": faiss_index_dir(os.path.basename(f)),
                    "debug_evidence_file": debug_evidence_path(os.path.basename(f)),
                    "mode": mode,
                    "ticket_used": bool(ticket_text and mode == "server_monitoring"),
                    "ticket_chars": len(ticket_text) if (ticket_text and mode == "server_monitoring") else 0,
                }
            )

    invalid_query_reports = [
        r for r in all_findings if r.get("query_valid", True) is False
    ]
    if invalid_query_reports:
        first_invalid = invalid_query_reports[0]
        invalid_message = (
            "Invalid date/time window for selected logs."
            f"\n\n{first_invalid.get('findings', '').strip()}"
        )
        print(
            f"[Query] Blocking report generation due to invalid window in {len(invalid_query_reports)} file(s)."
        )
        if return_metadata:
            return {
                "status": "invalid_query_window",
                "message": invalid_message,
                "invalid_reports": invalid_query_reports,
            }
        return invalid_message

    # 3. Reduce — consolidate
    final_report = consolidate_reports(all_findings, mode=mode)
    if filter_summary is not None:
        final_report = (
            "## Query Time Filter\n"
            f"- {filter_summary}\n\n"
            f"{final_report}"
        )

    # 4. Export to PDF
    export_to_pdf(final_report)

    if return_metadata:
        return {
            "status": "ok",
            "report": final_report,
            "log_files": all_log_files,
            "per_file_reports": all_findings,
            "ticket_used": bool(ticket_text and mode == "server_monitoring"),
            "ticket_chars": len(ticket_text) if (ticket_text and mode == "server_monitoring") else 0,
        }

    return final_report


def interactive_mode() -> None:
    """
    Run the agent in interactive mode for ad-hoc queries.
    Users can type paths to analyse or ask questions.
    """
    print("\n" + "=" * 60)
    print("IAM Log Intelligence Agent - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  analyze <path>  - Run full pipeline on a file or folder")
    print("  exit / quit     - End session")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            # Check for analyze command
            if user_input.lower().startswith("analyze "):
                path = user_input[8:].strip().strip('"').strip("'")
                if os.path.exists(path):
                    result = run_pipeline(path)
                    print("\n" + "=" * 60)
                    print("ANALYSIS COMPLETE")
                    print("=" * 60)
                    print(result)
                else:
                    print(f"Error: Path not found: {path}")
                continue

            # General question — use LLM directly
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert IAM Log Intelligence Agent. "
                        "Answer the user's question concisely. If they want to analyse "
                        "logs, tell them to use: analyze <path>"
                    )
                ),
                HumanMessage(content=user_input),
            ]
            response = llm.invoke(messages)
            print(f"\nAgent: {response.content}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
