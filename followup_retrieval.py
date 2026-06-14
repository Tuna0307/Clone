"""
Artifact-first follow-up retrieval helpers for IAM log analysis chat.

Follow-up answers are grounded in artifacts from the most recent analysis:
FAISS metadata, debug evidence, raw uploaded/local logs, the final report, and
the session vector store. This avoids rerunning the full pipeline for normal
clarification questions.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from config import (
    LLM_MODEL_ID,
    EMBEDDING_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from llm_factory import get_llm, get_embeddings

try:
    import faiss
except Exception:  # pragma: no cover - optional runtime dependency
    faiss = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]

from artifact_paths import debug_evidence_path, faiss_index_dir

from followup.context import (
    FOLLOWUP_HIGH_ANOMALY_THRESHOLD,
    ArtifactEntry,
    AnalysisContext,
    FollowupIntent,
    EvidenceItem,
    _safe_abspath,
    build_analysis_context,
    _try_parse_datetime,
    _load_metadata_rows,
    _as_float,
    build_retrieved_chunks_table_data,
    build_coverage_summary_table_data,
    build_analysis_results_metadata_markdown,
)

from followup.intent import (
    _get_followup_llm,
    _get_followup_embeddings,
    _extract_first_json_object,
    _format_chat_history,
    _fallback_intent_from_query,
    _parse_intent,
    _SUMMARY_BROAD_MARKERS,
    _BROAD_ISSUE_TERMS,
)

from followup.sources import (
    parse_debug_evidence_file,
    _is_api_followup_mode,
    _extract_ref_ids,
    _debug_ref_candidates,
    build_analysis_results_debug_markdown,
    _metadata_candidates,
    _faiss_semantic_candidates,
    _split_candidate_snippets,
    _debug_evidence_candidates,
    _extract_raw_log_windows,
    _raw_log_candidates,
    _vector_store_candidates,
    FOLLOWUP_EVIDENCE_ITEM_PROMPT_CHARS,
)

from followup.answer import (
    DEFAULT_TOP_K_RESULTS,
    FOLLOWUP_MAX_EVIDENCE_ITEMS,
    FOLLOWUP_LLM_MAX_TOKENS,
    FOLLOWUP_EVIDENCE_BUDGET_CHARS,
    _SOURCE_WEIGHTS,
    _SOURCE_CAPS,
    _intent_terms,
    _rank_and_select_evidence,
    _build_evidence_table,
    _build_evidence_block_for_prompt,
    _generate_conversational_answer,
    answer_analysis_results_query,
)
