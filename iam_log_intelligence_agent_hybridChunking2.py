"""
IAM Log Intelligence Agent - production shim
============================================
Thin CLI/UI entry point for the modular pipeline:
  - api_request (default): deterministic request extraction + Map/Reduce LLM
  - server_monitoring: DuckDB load + LangGraph structured SQL workflow
"""

from pipeline.query import build_query_context
from pipeline.runner import run_pipeline

__all__ = ["build_query_context", "run_pipeline"]

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="IAM Log Intelligence Agent")
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
        help="Optional path to a support ticket (.md or .txt). Only used in --mode server_monitoring for the post-report agentic SQL refinement iteration.",
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