"""
Artifact-first follow-up retrieval helpers for IAM log analysis chat.

Follow-up answers are grounded in artifacts from the most recent analysis:
debug evidence, raw uploaded/local logs, and the final report. This avoids
rerunning the full pipeline for normal clarification questions.
"""

from followup.answer import (
    DEFAULT_TOP_K_RESULTS,
    FOLLOWUP_EVIDENCE_BUDGET_CHARS,
    FOLLOWUP_LLM_MAX_TOKENS,
    FOLLOWUP_MAX_EVIDENCE_ITEMS,
    answer_analysis_results_query,
)
from followup.context import (
    AnalysisContext,
    ArtifactEntry,
    EvidenceItem,
    FollowupIntent,
    build_analysis_context,
    build_analysis_results_metadata_markdown,
    build_coverage_summary_table_data,
    build_retrieved_chunks_table_data,
)
from followup.sources import (
    _debug_ref_candidates,
    _extract_ref_ids,
    _is_api_followup_mode,
    _raw_log_candidates,
    build_analysis_results_debug_markdown,
    parse_debug_evidence_file,
)

__all__ = [
    "DEFAULT_TOP_K_RESULTS",
    "FOLLOWUP_EVIDENCE_BUDGET_CHARS",
    "FOLLOWUP_LLM_MAX_TOKENS",
    "FOLLOWUP_MAX_EVIDENCE_ITEMS",
    "AnalysisContext",
    "ArtifactEntry",
    "EvidenceItem",
    "FollowupIntent",
    "answer_analysis_results_query",
    "build_analysis_context",
    "build_analysis_results_debug_markdown",
    "build_analysis_results_metadata_markdown",
    "build_coverage_summary_table_data",
    "build_retrieved_chunks_table_data",
    "parse_debug_evidence_file",
    "_debug_ref_candidates",
    "_extract_ref_ids",
    "_is_api_followup_mode",
    "_raw_log_candidates",
]