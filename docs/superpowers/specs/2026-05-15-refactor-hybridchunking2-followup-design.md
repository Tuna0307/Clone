# Refactor: HybridChunking2 Pipeline + Follow-up Retrieval

> **Date:** 2026-05-15
> **Scope:** `iam_log_intelligence_agent_hybridChunking2.py` and `followup_retrieval.py`
> **Constraint:** All functionality must be identical. No behavior changes.
> **Approach:** Conservative thin-shim extraction with test-first migration.

---

## 1. Problem Statement

The two key production files have grown monolithic:

| File | Lines | Functions/Classes |
|:---|:---|:---|
| `iam_log_intelligence_agent_hybridChunking2.py` | ~4,300 | ~60 |
| `followup_retrieval.py` | ~1,500 | ~25 |

There are **zero automated tests**. Adding tests to these files directly is difficult because responsibilities are tangled (chunking mixed with LLM calls, scoring mixed with file I/O, etc.). A conservative refactor into single-responsibility modules enables testing without changing any behavior.

---

## 2. Goals

1. **Identical behavior** — every public function signature, return value, and side effect stays the same
2. **Testability** — each new module can be unit-tested in isolation
3. **Readability** — a developer can open a module and understand its scope in one screen
4. **Zero breaking changes** — `app.py` and CLI imports continue to work without modification

---

## 3. Non-Goals

- No dependency injection of `llm`/`embeddings` (kept as lazy globals for identical behavior)
- No new exception types or logging infrastructure
- No type-hint improvements beyond what's necessary for extraction
- No performance changes
- No new features

---

## 4. Architecture

### 4.1 Thin-Shim Pattern

Original files become pass-through shims. They re-export everything from new submodules so existing importers (`app.py`, CLI, notebooks) need zero changes.

```
app.py ──► iam_log_intelligence_agent_hybridChunking2.py (thin shim)
              ├── from pipeline.constants import * → re-export
              ├── from pipeline.query import * → re-export
              ├── from pipeline.files import * → re-export
              ├── from pipeline.parsing import * → re-export
              ├── from pipeline.chunking import * → re-export
              ├── from pipeline.dedup import * → re-export
              ├── from pipeline.scoring import * → re-export
              ├── from pipeline.evidence import * → re-export
              ├── from pipeline.analysis import * → re-export
              ├── from pipeline.reporting import * → re-export
              └── from pipeline.runner import * → re-export

app.py ──► followup_retrieval.py (thin shim)
              ├── from followup.context import * → re-export
              ├── from followup.intent import * → re-export
              ├── from followup.sources import * → re-export
              └── from followup.answer import * → re-export
```

### 4.2 Pipeline Module Decomposition

From `iam_log_intelligence_agent_hybridChunking2.py`:

| Module | Responsibility | Key Functions/Classes |
|:---|:---|:---|
| `pipeline/constants.py` | Budget constants, thresholds, window sizes | `MAP_EVIDENCE_BUDGET_CHARS`, `SERVER_MONITOR_WINDOW_SECONDS`, ... |
| `pipeline/query.py` | Search-config loading, query parsing, classification | `load_search_config`, `parse_query_datetime`, `classify_query_category`, `validate_query_window`, ... |
| `pipeline/files.py` | File discovery, streaming, size formatting | `get_log_files_from_path`, `stream_file_lines`, `format_file_size` |
| `pipeline/parsing.py` | Log structure detection, line parsing | `detect_log_structure`, `_parse_line`, `_extract_session_label`, `_parse_iso_timestamp` |
| `pipeline/chunking.py` | All chunking strategies | `hybrid_chunk_log`, `chunk_server_monitoring_log`, `chunk_api_requests_hierarchical`, `extract_api_request_docs_deterministic` |
| `pipeline/dedup.py` | Pre-embedding deduplication and downselection | `deduplicate_chunks_safe`, `downselect_chunks_for_embedding`, `filter_chunks_by_signal` |
| `pipeline/scoring.py` | Embedding + anomaly scoring | `_embed_batch_with_retry`, `_embed_documents_batched`, `score_anomalies` |
| `pipeline/evidence.py` | Evidence selection, metadata, profiling | `select_evidence_chunks`, `build_metadata_rows_from_docs`, `extract_global_evidence_profile` |
| `pipeline/analysis.py` | Per-file Map-phase LLM analysis | `analyze_single_file` |
| `pipeline/reporting.py` | Report consolidation, PDF export, reference replacement | `consolidate_reports`, `export_to_pdf`, `_replace_chunk_refs_with_original_references`, `_markdown_links_to_reportlab` |
| `pipeline/runner.py` | Top-level orchestration | `run_pipeline`, `interactive_mode` |

### 4.3 Follow-up Module Decomposition

From `followup_retrieval.py`:

| Module | Responsibility | Key Functions/Classes |
|:---|:---|:---|
| `followup/context.py` | Dataclasses and context building | `ArtifactEntry`, `AnalysisContext`, `FollowupIntent`, `EvidenceItem`, `build_analysis_context` |
| `followup/intent.py` | Query intent parsing | `_parse_intent`, `_fallback_intent_from_query`, `_format_chat_history` |
| `followup/sources.py` | Evidence source retrieval | `_faiss_semantic_candidates`, `_metadata_candidates`, `_raw_log_candidates`, `_debug_evidence_candidates`, `_vector_store_candidates` |
| `followup/answer.py` | Ranking, prompt building, LLM answer generation | `_rank_and_select_evidence`, `_build_evidence_block_for_prompt`, `_generate_conversational_answer`, `answer_analysis_results_query` |

### 4.4 Dependency Flow

```
pipeline.constants
    ↓
pipeline.query  ←──  pipeline.files  ←──  pipeline.parsing
    ↓                    ↓                  ↓
    └──────────────→  pipeline.chunking  ←──┘
                         ↓
                    pipeline.dedup
                         ↓
                    pipeline.scoring
                         ↓
                    pipeline.evidence
                         ↓
                    pipeline.analysis  ←──  pipeline.reporting
                                              ↓
                                         pipeline.runner

followup.context
    ↓
followup.intent
    ↓
followup.sources  ←──  followup.context
    ↓
followup.answer   ←──  followup.intent + followup.sources
```

No circular dependencies. `followup.*` imports `pipeline.constants` and `pipeline.query` for shared config but never the full pipeline.

---

## 5. Global State Preservation

The original files define module-level globals:

- `llm = get_llm()`, `embeddings = get_embeddings()` in `iam_log_intelligence_agent_hybridChunking2.py`
- `_FOLLOWUP_LLM = None`, `_FOLLOWUP_EMBEDDINGS = None` in `followup_retrieval.py`

**Conservative decision:** These stay exactly as-is in their new home modules. The thin shims re-export them so any code that does `from iam_log_intelligence_agent_hybridChunking2 import llm` continues to work.

No dependency injection. No parameter changes. No initialization timing changes.

---

## 6. Error Handling

All existing `try/except`, `print(...)` diagnostics, `# AGENT ERROR` return strings, and graceful-degradation paths are preserved verbatim. No new exception types. No new logging infrastructure.

---

## 7. Testing Strategy

### 7.1 Test-First Migration Order

For each module, write its tests **before** moving code into it:

1. Write test file targeting the new module path
2. Run tests → they fail (module doesn't exist yet)
3. Create module, move code, update shim
4. Run tests → pass
5. Commit

### 7.2 Test Files

| Module | Test File |
|:---|:---|
| `pipeline/constants.py` | `tests/pipeline/test_constants.py` |
| `pipeline/query.py` | `tests/pipeline/test_query.py` |
| `pipeline/files.py` | `tests/pipeline/test_files.py` |
| `pipeline/parsing.py` | `tests/pipeline/test_parsing.py` |
| `pipeline/chunking.py` | `tests/pipeline/test_chunking.py` |
| `pipeline/dedup.py` | `tests/pipeline/test_dedup.py` |
| `pipeline/scoring.py` | `tests/pipeline/test_scoring.py` |
| `pipeline/evidence.py` | `tests/pipeline/test_evidence.py` |
| `pipeline/analysis.py` | `tests/pipeline/test_analysis.py` |
| `pipeline/reporting.py` | `tests/pipeline/test_reporting.py` |
| `followup/context.py` | `tests/followup/test_context.py` |
| `followup/intent.py` | `tests/followup/test_intent.py` |
| `followup/sources.py` | `tests/followup/test_sources.py` |
| `followup/answer.py` | `tests/followup/test_answer.py` |

### 7.3 Testing Infrastructure

- `pytest` as runner
- `tests/conftest.py` with fixtures:
  - `mock_llm` — monkey-patches `get_llm()` to return a mock
  - `mock_embeddings` — monkey-patches `get_embeddings()` to return deterministic vectors
  - `sample_log_file` — creates a temporary `.log` file with realistic lines
  - `sample_schema` — returns a pre-built schema dict for tests that need it
- All tests use temp files and synthetic data — no real log files required

### 7.4 Smoke Test Preservation

After full migration, the existing smoke tests from `AGENTS.md` must still pass:

```bash
streamlit run app.py --server.fileWatcherType none
python iam_log_intelligence_agent_hybridChunking2.py path/to/log/or/folder
python -m unittest discover -s tests
```

---

## 8. File Structure (After)

```
├── app.py                          (unchanged)
├── iam_log_intelligence_agent_hybridChunking2.py   (thin shim)
├── followup_retrieval.py           (thin shim)
├── pipeline/
│   ├── __init__.py
│   ├── constants.py
│   ├── query.py
│   ├── files.py
│   ├── parsing.py
│   ├── chunking.py
│   ├── dedup.py
│   ├── scoring.py
│   ├── evidence.py
│   ├── analysis.py
│   ├── reporting.py
│   └── runner.py
├── followup/
│   ├── __init__.py
│   ├── context.py
│   ├── intent.py
│   ├── sources.py
│   └── answer.py
├── tests/
│   ├── conftest.py
│   ├── pipeline/
│   │   ├── test_constants.py
│   │   ├── test_query.py
│   │   ├── test_files.py
│   │   ├── test_parsing.py
│   │   ├── test_chunking.py
│   │   ├── test_dedup.py
│   │   ├── test_scoring.py
│   │   ├── test_evidence.py
│   │   ├── test_analysis.py
│   │   └── test_reporting.py
│   └── followup/
│       ├── test_context.py
│       ├── test_intent.py
│       ├── test_sources.py
│       └── test_answer.py
└── docs/superpowers/specs/2026-05-15-refactor-hybridchunking2-followup-design.md
```

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|:---|:---|
| Import cycles during extraction | Follow the dependency flow diagram; verify with `python -c "import pipeline.X"` after each module |
| `app.py` import breakage | Keep thin shims as explicit `from pipeline.X import func; __all__ += ['func']` re-exports |
| Tests pass but behavior drifted | Write integration smoke tests that call the top-level functions with real temp logs before and after |
| Module-level globals break timing | Re-export the same globals from the shim; do not re-call `get_llm()` in the shim |
| Lost docstrings/comments | Copy all docstrings and comments verbatim during move |

---

## 10. Rollback Plan

Because the originals become thin shims, the actual code still exists — just in new files. If anything breaks:

1. The shim can be replaced with the original inline code (copy back from git history)
2. No `app.py` changes means the only risk is within the two target files
3. Git commits after each module extraction allow per-module rollback

---

## 11. Success Criteria

- [ ] All 15 new modules extracted and shimmed
- [ ] `app.py` requires zero import changes
- [ ] `python -m py_compile` passes on all new files + shims
- [ ] All new tests pass (`pytest tests/`)
- [ ] Smoke tests from `AGENTS.md` pass unchanged
- [ ] `iam_log_intelligence_agent_hybridChunking2.py` and `followup_retrieval.py` are <200 lines each
