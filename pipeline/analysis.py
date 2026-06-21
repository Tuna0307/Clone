"""Map-phase per-file analysis for the IAM log intelligence pipeline."""

import gc
import json
import os
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.chunking import extract_api_request_docs_deterministic
from pipeline.constants import (
    ANOMALY_HIGH_THRESHOLD,
    MAP_EVIDENCE_BUDGET_CHARS,
    MAP_NEIGHBOUR_RADIUS,
    MAP_TOP_N_CHUNKS,
    MAX_LOG_FILE_SIZE_BYTES,
    _DEFAULT_API_REQUEST_BOUNDARIES,
)
from pipeline.dedup import (
    build_metadata_rows_from_docs,
    extract_global_evidence_profile,
)
from pipeline.evidence import select_evidence_chunks
from pipeline.files import format_file_size, stream_file_lines
from pipeline.parsing import detect_log_structure, parse_query_datetime
from pipeline.query import (
    _lazy_get_detect_log_structure_hybrid,
    _schema_query_formats,
    _should_try_hybrid_schema,
    build_query_filter_summary,
    classify_api_subcategory,
    compute_file_time_coverage,
    load_retrieval_signals,
    validate_query_window,
)
from pipeline.progress import emit_ui_progress
from llm_factory import get_llm
from artifact_paths import debug_evidence_path, ensure_parent_dir, faiss_index_dir

llm = get_llm()


def analyze_single_file(
    file_path: str,
    query_context: Optional[dict[str, Any]] = None,
    mode: str = "api_request",
    ticket_text: Optional[str] = None,
) -> dict:
    """
    [MAP STEP] Analyse a single log file end-to-end.

    When mode="api_request" (default): uses the existing deterministic extraction path.
    When mode="server_monitoring": completely bypasses chunking/embedding/FAISS/anomaly
    and instead loads metrics into DuckDB for agentic SQL analysis by the LLM.

    The optional `ticket_text` is **only** acted upon in server_monitoring mode.
    Per the design, the ticket is sent to the LLM **after** the normal agentic SQL
    loop has produced its initial report. This triggers a short refinement iteration
    ("directly to the agent in the end") so the LLM can run additional targeted
    SQL + raw_line queries to better address the symptoms described in the ticket.

    Args:
        file_path: Absolute path to the log file
        query_context: Optional time window + query text
        mode: "api_request" (default, unchanged behavior) or "server_monitoring"
        ticket_text: Pre-formatted support ticket / incident description text
            (only used for the post-report refinement pass in server_monitoring mode).

    Returns:
        Dict compatible with runner / consolidate (findings, category, metadata etc.)
    """
    file_name = os.path.basename(file_path)
    debug_file = debug_evidence_path(file_name)
    metadata_rows: list[dict[str, Any]] = []
    selected_row_ids_for_reduce: list[str] = []
    source_reference_map: list[dict[str, Any]] = []
    evidence_profile: dict[str, Any] = {}
    file_size = os.path.getsize(file_path)
    print(f"\n{'='*60}")
    print(f"[MAP] Analysing: {file_name} ({format_file_size(file_size)})")
    print(f"{'='*60}")

    if file_size > MAX_LOG_FILE_SIZE_BYTES:
        print(f"  [Skip] File exceeds max size limit ({format_file_size(MAX_LOG_FILE_SIZE_BYTES)}): {format_file_size(file_size)}")
        return {
            "file": file_name,
            "findings": "",
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "metadata_rows": [],
            "selected_row_ids_for_reduce": [],
            "status": "skipped_file_too_large",
            "query_valid": True,
            "query_validation_reason": "file_too_large",
            "category": "unclassified",
            "subcategory": "unclassified",
            "evidence_profile": evidence_profile,
            "mode": mode,
        }

    retrieval_signals = load_retrieval_signals()
    query_text = str(query_context.get('query_text', '')) if query_context else ''

    category = mode
    print(f"  [Mode] Using user-selected analysis path: {mode}")

    subcategory = 'unclassified'
    if mode == 'api_request':
        subcategory = classify_api_subcategory(query_text, retrieval_signals['api_known_error_keywords']) if query_text else 'unknown_error'

    # ---- 2. Detect structure ----
    print("  [Structure] Sampling lines for schema detection...")
    emit_ui_progress("Sampling lines for schema detection...")
    sample_lines: list[str] = []
    try:
        for line in stream_file_lines(file_path):
            sample_lines.append(line)
            if len(sample_lines) >= 1200:
                break
    except Exception as e:
        print(f"  [Error] Cannot read file: {e}")
        return {
            "file": file_name,
            "findings": "",
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "metadata_rows": [],
            "selected_row_ids_for_reduce": [],
            "evidence_profile": evidence_profile,
        }

    if not sample_lines:
        print("  [Skip] Empty file.")
        return {
            "file": file_name,
            "findings": "",
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "metadata_rows": [],
            "selected_row_ids_for_reduce": [],
            "evidence_profile": evidence_profile,
            "mode": mode,
        }

    schema = detect_log_structure(sample_lines)
    if _should_try_hybrid_schema(schema):
        hybrid_detector = _lazy_get_detect_log_structure_hybrid()
        if hybrid_detector is not None:
            try:
                hybrid_schema = hybrid_detector(sample_lines, use_llm_fallback=True, enable_multiline=True)
                required_keys = {'timestamp_re', 'timestamp_fmt', 'thread_re', 'session_keys', 'stack_trace_re'}
                if isinstance(hybrid_schema, dict) and required_keys.issubset(hybrid_schema.keys()):
                    schema = hybrid_schema
                    print("  [Structure] Hybrid schema detector engaged")
                else:
                    print("  [Structure] Hybrid schema returned invalid contract; using regex schema")
            except Exception as e:
                print(f"  [Structure] Hybrid schema detection failed ({e}); using regex schema")
    ts_detected = schema['timestamp_fmt'] != ''
    thread_detected = schema['thread_re'] is not None
    print(f"    Timestamp detected: {ts_detected} | Thread detected: {thread_detected}")
    emit_ui_progress(f"Timestamp detected: {ts_detected} | Thread detected: {thread_detected}")
    if schema['session_keys']:
        print(f"    Session keys: {[k for _, k in schema['session_keys']]}")

    if query_context is not None:
        schema_formats = _schema_query_formats(schema)
        start_raw = str(query_context.get('start_time_raw', '')).strip()
        end_raw = str(query_context.get('end_time_raw', '')).strip()

        if start_raw:
            reparsed_start = parse_query_datetime(
                start_raw,
                use_end_of_day_for_date_only=False,
                additional_formats=schema_formats,
            )
            if reparsed_start is not None:
                query_context['start_time'] = reparsed_start

        if end_raw:
            reparsed_end = parse_query_datetime(
                end_raw,
                use_end_of_day_for_date_only=True,
                additional_formats=schema_formats,
            )
            if reparsed_end is not None:
                query_context['end_time'] = reparsed_end

    # ---- 3. Query window validation (before any chunking/extraction) ----
    min_ts, max_ts = compute_file_time_coverage(file_path, schema)
    is_valid, reason_code, reason_message = validate_query_window(query_context, min_ts, max_ts)
    if not is_valid:
        print(f"  [Query] Invalid query window: {reason_message}")
        return {
            "file": file_name,
            "findings": (
                f"# QUERY VALIDATION FAILED\n"
                f"- Reason: {reason_code}\n"
                f"- Detail: {reason_message}\n"
                f"- File coverage: {min_ts.isoformat() if min_ts else 'N/A'} -> {max_ts.isoformat() if max_ts else 'N/A'}"
            ),
            "chunk_count": 0,
            "high_anomaly_count": 0,
            "metadata_rows": [],
            "selected_row_ids_for_reduce": [],
            "status": "invalid_query_window",
            "query_valid": False,
            "query_validation_reason": reason_code,
            "category": category,
            "subcategory": subcategory,
            "evidence_profile": evidence_profile,
        }

    filter_summary = build_query_filter_summary(query_context)
    if filter_summary is not None:
        print(f"  [Query] {filter_summary} (pre-chunk validation passed)")

    # ---- 3b. File-wide deterministic profile (teammate hybridChunking2_1 merge) ----
    try:
        evidence_profile = extract_global_evidence_profile(
            file_path,
            schema,
            retrieval_signals,
            query_context,
        )
    except Exception as profiler_err:
        print(f"  [Profiler] Warning: Profile extraction failed ({profiler_err}); continuing without profile")
        evidence_profile = {"error": str(profiler_err)}

    # ---- 4. Category-aware extraction / loading ----
    if mode == "api_request":
        # ---- 4A. API deterministic extraction (existing path, 100% unchanged behavior) ----
        try:
            docs = extract_api_request_docs_deterministic(
                file_path,
                schema,
                retrieval_signals.get('api_request_boundaries', dict(_DEFAULT_API_REQUEST_BOUNDARIES)),
                retrieval_signals,
                query_context,
            )
        except Exception as e:
            print(f"  [Error] API deterministic extraction failed: {e}")
            return {
                "file": file_name,
                "findings": "",
                "chunk_count": 0,
                "high_anomaly_count": 0,
                "metadata_rows": [],
                "selected_row_ids_for_reduce": [],
                "status": "api_deterministic_error",
                "query_valid": True,
                "query_validation_reason": "api_deterministic_error",
                "category": category,
                "subcategory": subcategory,
                "evidence_profile": evidence_profile,
                "mode": mode,
            }

        if not docs:
            print("  [Skip] No API request evidence extracted.")
            return {
                "file": file_name,
                "findings": "",
                "chunk_count": 0,
                "high_anomaly_count": 0,
                "metadata_rows": [],
                "selected_row_ids_for_reduce": [],
                "status": "no_chunks",
                "query_valid": True,
                "query_validation_reason": "no_chunks",
                "category": category,
                "subcategory": subcategory,
                "evidence_profile": evidence_profile,
                "mode": mode,
            }

        chunk_count = len(docs)
        high_anomaly_count = sum(1 for d in docs if float(d.metadata.get('anomaly_score', 0.0)) > ANOMALY_HIGH_THRESHOLD)
        for idx, doc in enumerate(docs):
            doc.metadata['row_id'] = f"{file_name}::{idx}"
        metadata_rows = build_metadata_rows_from_docs(docs)
        print("  [Mode] API deterministic mode active: skipping embedding + anomaly stage")

        # Evidence selection (API path)
        evidence_text, selected_row_ids_for_reduce, source_reference_map = select_evidence_chunks(
            docs,
            top_n=MAP_TOP_N_CHUNKS,
            neighbour_radius=MAP_NEIGHBOUR_RADIUS,
            max_total_chars=MAP_EVIDENCE_BUDGET_CHARS,
            category=category,
        )

        # Free memory before LLM
        del docs
        gc.collect()

    else:
        # =====================================================================
        # 4B. SERVER MONITORING — DuckDB + structured workflow (Phase 4 final path)
        # Completely bypasses chunking, embedding, FAISS, anomaly scoring.
        # Always uses the explicit-phase LangGraph-powered workflow in server_sql_graph.py.
        # No feature flag. No legacy ReAct fallback.
        # =====================================================================
        print("  [Server] Using STRUCTURED WORKFLOW (LangGraph + explicit phases)...")
        from pipeline.server_sql_graph import analyze_server_log_with_workflow
        result = analyze_server_log_with_workflow(
            file_path, schema, query_context, ticket_text
            # llm will be obtained inside the workflow if not passed
        )
        if result.get("debug_evidence_file"):
            print(f"  [Server] Structured workflow artifacts written to: {result.get('debug_evidence_file')}")
        return result

    # (Legacy server_monitoring ReAct code removed in Phase 4.)

# ---- 5. Evidence selection for API path only (server path already handled above) ----
    if mode == "api_request":
        # evidence_text / selected... already populated inside the if
        pass
    else:
        # Server path already set the return variables above
        evidence_text = ""
        selected_row_ids_for_reduce = []
        source_reference_map = []

    if mode == "api_request" and not evidence_text.strip():
        print("  [Skip] No evidence to analyse.")
        return {
            "file": file_name,
            "findings": "",
            "chunk_count": chunk_count,
            "high_anomaly_count": high_anomaly_count,
            "metadata_rows": metadata_rows,
            "selected_row_ids_for_reduce": selected_row_ids_for_reduce,
            "source_reference_map": source_reference_map,
            "status": "no_evidence",
            "query_valid": True,
            "query_validation_reason": "ok",
            "category": category,
            "subcategory": subcategory,
            "evidence_profile": evidence_profile,
            "mode": mode,
        }

    # Free chunk memory (API path only)
    if mode == "api_request":
        try:
            del docs
            gc.collect()
        except Exception:
            pass

    # ---- 6. Structured LLM analysis (API path only) ----
    # Server-monitoring mode already produced its findings (with exact required headers)
    # inside the agentic SQL loop above. We skip the old evidence-based map prompt entirely.
    if mode == "api_request":
        print("  [LLM] Running forensic analysis (API path)...")
        emit_ui_progress("[LLM] Running forensic analysis")

        from pipeline.prompts_api import build_api_map_messages

        system_prompt, user_prompt = build_api_map_messages(
            file_name=file_name,
            category=category,
            subcategory=subcategory,
            evidence_profile=evidence_profile,
            evidence_text=evidence_text,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Debug output + save (same pattern as original)
        evidence_preview = evidence_text[:500] + "..." if len(evidence_text) > 500 else evidence_text
        print(f"  [DEBUG] Sending {len(evidence_text):,} chars to LLM")
        emit_ui_progress(f"Sending {len(evidence_text):,} chars to LLM")
        print(f"  [DEBUG] Evidence preview: {evidence_preview[:200]}...")

        try:
            ensure_parent_dir(debug_file)
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== EVIDENCE PROFILE ===\n{json.dumps(evidence_profile, indent=2)}\n\n")
                f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
                f.write(f"=== USER PROMPT ===\n{user_prompt}\n")
            print(f"  [DEBUG] Evidence saved to {debug_file}")
        except Exception as debug_err:
            print(f"  [WARNING] Could not save debug file: {debug_err}")

        try:
            response = llm.invoke(messages)
            findings = response.content
            print(f"  [DEBUG] LLM returned {len(findings):,} chars")
            emit_ui_progress(f"LLM returned {len(findings):,} chars")
        except Exception as e:
            error_msg = str(e)
            print(f"  [ERROR] LLM call failed: {error_msg}")
            findings = "# AGENT ERROR (LLM call failed in API path)"

    # ---- Final return (common) ----
    # Provide safe defaults so both paths always produce a complete dict
    findings = locals().get("findings", "")
    chunk_count = locals().get("chunk_count", 0)
    high_anomaly_count = locals().get("high_anomaly_count", 0)
    metadata_rows = locals().get("metadata_rows", [])
    selected_row_ids_for_reduce = locals().get("selected_row_ids_for_reduce", [])
    source_reference_map = locals().get("source_reference_map", [])

    final_return = {
        "file": file_name,
        "findings": findings,
        "chunk_count": chunk_count,
        "high_anomaly_count": high_anomaly_count,
        "metadata_rows": metadata_rows,
        "selected_row_ids_for_reduce": selected_row_ids_for_reduce,
        "source_reference_map": source_reference_map,
        "status": "ok",
        "query_valid": True,
        "query_validation_reason": "ok",
        "category": category,
        "subcategory": subcategory,
        "evidence_profile": evidence_profile,
        "mode": mode,
    }

    # Add faiss dir only for API path (server has none)
    if mode == "api_request":
        final_return["faiss_index_dir"] = faiss_index_dir(file_name)
        final_return["debug_evidence_file"] = debug_evidence_path(file_name)
    else:
        final_return["faiss_index_dir"] = None
        final_return["debug_evidence_file"] = debug_evidence_path(file_name)
        final_return["duckdb_row_count"] = row_count if "row_count" in dir() else 0

    return final_return
