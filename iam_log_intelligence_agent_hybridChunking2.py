"""
IAM Log Intelligence Agent - Current HybridChunking2 Pipeline
=================================================
Current Streamlit/CLI pipeline with provider-agnostic LLM calls, deterministic
API request extraction, hybrid server-monitoring chunking, evidence citations,
and PDF export.

Pipeline stages:
  1. File Discovery          - get_log_files_from_path
  2. Preprocessing           - detect_log_structure + query-window validation
  3. Category Extraction     - deterministic API docs or hybrid server chunks
  4. Evidence Selection      - scoring, neighbours, citations, and budgets
  5. Map Phase               - structured per-file LLM analysis
  6. Reduce Phase            - final consolidation and report export
"""

from llm_factory import get_llm, get_embeddings

from pipeline.analysis import analyze_single_file
from pipeline.chunking import hybrid_chunk_log
from pipeline.evidence import select_evidence_chunks
from pipeline.files import get_log_files_from_path
from pipeline.parsing import detect_log_structure
from pipeline.query import build_query_context, load_retrieval_signals
from pipeline.reporting import consolidate_reports, export_to_pdf
from pipeline.runner import interactive_mode, run_pipeline
from pipeline.scoring import score_anomalies

llm = get_llm()
embeddings = get_embeddings()


# ---------------------------------------------------------------------------
# Optional CLI entry point (convenience for `python ... --mode server_monitoring path`)
# The primary entry points remain the Streamlit app and direct Python calls.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="IAM Log Intelligence Agent (hybridChunking2)")
    parser.add_argument("paths", nargs="+", help="One or more log files or directories")
    parser.add_argument(
        "--mode",
        choices=["api_request", "server_monitoring"],
        default="api_request",
        help="Analysis mode (default: api_request). 'server_monitoring' enables the DuckDB + agentic SQL path.",
    )
    parser.add_argument(
        "--return-metadata",
        action="store_true",
        help="Return structured metadata instead of plain report text",
    )
    parser.add_argument(
        "--ticket-file",
        "--ticket",
        dest="ticket_file",
        default=None,
        help="Optional path to a support ticket (.md or .txt). Only used in --mode server_monitoring for the post-report agentic SQL refinement iteration. No automatic directory scanning is performed.",
    )
    args = parser.parse_args()

    ticket_text = None
    if args.ticket_file:
        try:
            with open(args.ticket_file, "r", encoding="utf-8", errors="replace") as tf:
                ticket_text = tf.read()
        except Exception as e:
            print(f"[Warning] Could not read --ticket-file: {e}")
            ticket_text = None

    result = run_pipeline(
        args.paths,
        return_metadata=args.return_metadata,
        mode=args.mode,
        ticket_text=ticket_text,
    )
    if isinstance(result, dict):
        print(result.get("report", str(result)))
    else:
        print(result)
    sys.exit(0)
