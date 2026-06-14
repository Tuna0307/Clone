# AGENTS.md

This document provides developer guidance for the IAM Log Intelligence Agent repository.

## 1. Build and Environment

### Prerequisites

- Python 3.10 or higher.
- A virtual environment or conda environment is recommended.
- OpenAI credentials are required when `LLM_PROVIDER=openai`.
- AWS credentials are required only when `LLM_PROVIDER=bedrock`.

### Setup Commands

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root. Use `.env.example` as the template.

OpenAI example:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL_ID=gpt-5.5
EMBEDDING_MODEL_ID=text-embedding-3-small
```

Bedrock example:

```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
LLM_MODEL_ID=us.meta.llama3-1-8b-instruct-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

## 2. Repository Structure

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit UI for uploads, local paths, incident windows, reports, and follow-up chat. |
| `iam_log_intelligence_agent_hybridChunking2.py` | Current production pipeline. Target active development here. |
| `iam_log_intelligence_agent_hybridChunking.py` | Older hybrid pipeline retained for comparison/reference. |
| `iam_log_intelligence_agent.py` | Legacy v1 agent that uses generic text splitting and tool-calling. |
| `llm_factory.py` | Provider factory for OpenAI and Bedrock chat/embedding clients. |
| `config.py` | Central environment-variable configuration and credential validation. |
| `artifact_paths.py` | Centralized output paths under `outputs/`. |
| `followup_retrieval.py` | Artifact-first retrieval for post-analysis follow-up questions. |
| `chat_vector_store.py` | Session-only vector store for report-level chat context. |
| `upload_utils.py` | Safe handling for Streamlit-uploaded log files. |
| `ui_time_utils.py` | UI date/time formatting helper. |
| `schema.py` | Optional schema inference helpers used when regex detection is low-confidence. |
| `search_config.json` | Routing terms, IAM keywords, request boundaries, and retrieval buckets. |
| `requirements.txt` | Pinned Python dependencies. |

Active development should target `iam_log_intelligence_agent_hybridChunking2.py` and the Streamlit flow in `app.py`.

## 3. Testing and Verification

### Streamlit smoke test

```bash
streamlit run app.py --server.fileWatcherType none
```

Use the UI to upload one or more `.log`, `.txt`, `.out`, `.err`, or `.msg` files. Confirm that:

1. The run prints `[MAP]`, routing, evidence, and `[REDUCE]` phases.
2. The report is rendered in the chat response.
3. `outputs/reports/IAM_Forensic_Report.pdf` is created.
4. Debug and FAISS artifacts are grouped under `outputs/debug/` and `outputs/faiss/`.

### CLI smoke test

```bash
python iam_log_intelligence_agent_hybridChunking2.py path/to/log/or/folder
```

Multiple paths are supported:

```bash
python iam_log_intelligence_agent_hybridChunking2.py path/to/file1.log path/to/folder2
```

### Unit checks

A local `tests/` folder may exist for developer-only checks, but it is ignored by Git in this repository. If present, run:

```bash
python -m unittest discover -s tests
```

## 4. Code Style and Conventions

### General Python Guidelines

- Follow PEP 8 and use 4 spaces for indentation.
- Add type hints to function signatures.
- Use `snake_case` for functions/variables, `UpperCamelCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Add docstrings where they explain module purpose, public helpers, or non-obvious logic.
- Keep comments short and technical. Avoid narrating obvious assignments.

### Imports

Order imports as:

1. Standard library.
2. Third-party libraries.
3. Project-local modules.
4. LangChain ecosystem imports.

The current code has some historical import ordering drift in legacy files. New edits should follow the order above without broad unrelated rewrites.

## 5. Pipeline Architecture Rules

The current pipeline processes each file in a staged Map-Reduce flow:

| Stage | Function | Purpose |
| :--- | :--- | :--- |
| 1 | `get_log_files_from_path` | Discover valid log files recursively or accept a single file. |
| 2 | `detect_log_structure` | Detect timestamps, thread IDs, session keys, and stack traces. |
| 2b | `extract_api_request_docs_deterministic` or chunking path | Route into API/request extraction or server monitoring chunking. |
| 2.5 | Large-log compression | Deduplicate and downselect very large server-monitoring chunk sets before embedding. |
| 3 | `score_anomalies` | Embed server-monitoring candidates and compute distance-based anomaly scores. |
| 4 | `select_evidence_chunks` | Merge ranked evidence, neighbours, citations, and budget limits. |
| 5 | `analyze_single_file` | Per-file map LLM analysis. |
| 6 | `consolidate_reports` | Cross-file reduce synthesis and final report text. |

Do not reorder these stages without updating the process diagram, README, and tests.

## 6. Citation and Line Reference Rules

The LLM-facing evidence still uses internal `[REF_...]` chunk IDs. User-facing report text must resolve those IDs back to original file and line references.

Important metadata fields:

- `source_file`: Source file base name.
- `source_path`: Absolute path used during analysis.
- `line_ranges`: Full chunk/request span.
- `error_line_ranges`: More precise error-bearing lines inside a larger chunk.

When both `error_line_ranges` and `line_ranges` exist, report rendering should prefer `error_line_ranges` for user-facing citations. This keeps reports useful to non-developer users who need to inspect the exact log line, not an internal chunk ID.

Do not weaken citation enforcement in prompts or reference replacement code.

## 7. Artifact Management

Generated files belong under `outputs/`:

- `outputs/reports/` for PDF reports.
- `outputs/debug/` for map evidence text.
- `outputs/faiss/` for FAISS indexes, embeddings, and metadata.
- `outputs/uploads/` for Streamlit-uploaded files.

Root-level `debug_evidence_*`, `faiss_index_*`, old PDFs, and upload folders are legacy patterns and should not be reintroduced.

## 8. Configuration Rules

- Keep domain search terms in `search_config.json`.
- Keep provider/model settings in `.env`.
- Avoid hardcoding customer-specific log paths.
- If adding new dependencies, update `requirements.txt` with pinned versions.
- Preserve the noise suppression and IAM-critical signal logic unless the sample incident set has been retested.

## 9. Error Handling

- File I/O, provider calls, FAISS work, and PDF export should fail gracefully.
- A single bad file should not crash a multi-file run.
- LLM failures should return a clear `# AGENT ERROR` result so reduce can skip bad map outputs safely.

## 10. Linting

No strict linter config is committed. Recommended local checks:

```bash
python -m py_compile app.py iam_log_intelligence_agent_hybridChunking2.py
# ruff check .  # if ruff is installed
```
