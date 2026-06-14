# IAM Log Intelligence Agent

## Project Overview

The IAM Log Intelligence Agent analyzes Identity and Access Management (IAM) logs and produces evidence-grounded forensic reports. The current application runs through Streamlit (`app.py`) and uses `iam_log_intelligence_agent_hybridChunking2.py` as the active analysis pipeline.

The code supports both OpenAI and AWS Bedrock through `llm_factory.py`, with the active provider configured in `.env`. Analysis artifacts are written under `outputs/` so generated debug evidence, FAISS indexes, uploaded files, and PDF reports do not clutter the repository root.

The system is designed for:

- Large log sets: a Map-Reduce pipeline processes files one at a time.
- Mixed log formats: schema detection identifies timestamps, threads, session keys, and stack-trace continuations.
- Request lifecycle analysis: API/request logs can be extracted deterministically without embedding every line.
- Server monitoring analysis: infrastructure-style logs can use either the legacy chunking/embedding/anomaly path or the opt-in DuckDB + agentic SQL path (recommended for UAM resource logs).
- Evidence traceability: final reports replace internal chunk IDs with original file names and precise line references where possible.

## Key Features

| Feature | Description |
| :--- | :--- |
| Streamlit UI | Upload one or more log files, or analyze a local file/folder path. |
| Date/time controls | Optional calendar and time pickers validate incident windows without manual timestamp typing. |
| Category routing | The pipeline routes work into API request analysis or server monitoring analysis based on query and log signals. |
| Deterministic API extraction | Request/event spans are built from thread, boundary markers, timestamps, and IAM-critical signals. |
| Hybrid chunking | Server monitoring logs are grouped by thread/session first, then by time windows, then catch-all chunks. |
| Pre-embedding compression | Large logs use conservative canonical deduplication before expensive embedding work. |
| Anomaly scoring | Server monitoring chunks use embedding distance, z-scores, IAM/error boosts, and noise suppression. |
| Strict citations | Reports cite original log file names and line numbers, including precise error-line references inside larger chunks. |
| Follow-up chat | After analysis, follow-up questions reuse stored artifacts instead of rerunning the pipeline by default. |
| Organized artifacts | Debug evidence, FAISS indexes, uploads, and reports are stored in `outputs/`. |

## Active Files

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit chat UI, file upload/local path input, incident time controls, and follow-up panels. |
| `iam_log_intelligence_agent_hybridChunking2.py` | Active analysis pipeline used by the app and CLI. |
| `llm_factory.py` | Provider factory for OpenAI or Bedrock chat models and embeddings. |
| `config.py` | Central environment configuration and credential validation. |
| `artifact_paths.py` | Central output paths for debug files, FAISS artifacts, reports, and uploads. |
| `followup_retrieval.py` | Artifact-first retrieval for follow-up questions after an analysis run. |
| `chat_vector_store.py` | Session-only Chroma vector store for report-level follow-up context. |
| `upload_utils.py` | Safe persistence for Streamlit-uploaded log files. |
| `ui_time_utils.py` | Small UI helper for formatting optional date/time inputs. |
| `schema.py` | Optional schema inference helpers used when regex schema detection is low-confidence. |
| `search_config.json` | Configurable IAM keywords, API request boundaries, routing terms, and retrieval buckets. |
| `iam_log_intelligence_agent_hybridChunking.py` | Older hybrid pipeline retained for reference. |
| `iam_log_intelligence_agent.py` | Legacy v1 agent using generic text splitting and tool-calling. |

## Installation

### Prerequisites

- Python 3.10 or newer.
- A virtual environment or conda environment is recommended.
- An OpenAI API key when `LLM_PROVIDER=openai`.
- AWS credentials only when `LLM_PROVIDER=bedrock`.

### Setup

```powershell
cd C:\Users\chimw\OneDrive\Desktop\GitHub\AAI_2002-ITP
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in the provider credentials you are using.

For OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
LLM_MODEL_ID=gpt-5.5
EMBEDDING_MODEL_ID=text-embedding-3-small
```

For Bedrock:

```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
LLM_MODEL_ID=us.meta.llama3-1-8b-instruct-v1:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

## Running the App

Start Streamlit with the file watcher disabled. This avoids known watcher noise from some Torch/Streamlit installations.

```powershell
streamlit run app.py --server.fileWatcherType none
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

In the sidebar:

1. Choose `Upload files` to select only the logs you want to analyze.
2. Choose `Local path` to analyze a specific local file or an entire folder.
3. Enable optional start/end time filters with the calendar and time controls.
4. Ask the analysis question in the chat box.

## CLI Usage

The active pipeline can also run directly from the command line:

```powershell
python iam_log_intelligence_agent_hybridChunking2.py path\to\logfile.log
```

Multiple paths are supported:

```powershell
python iam_log_intelligence_agent_hybridChunking2.py path\to\file1.log path\to\folder2
```

Running without arguments starts the script's interactive mode:

```powershell
python iam_log_intelligence_agent_hybridChunking2.py
```

## Output Layout

Generated files are intentionally grouped under `outputs/`:

| Path | Contents |
| :--- | :--- |
| `outputs/reports/IAM_Forensic_Report.pdf` | Generated PDF report. |
| `outputs/debug/debug_evidence_<file>.txt` | Exact evidence text sent to the map LLM for each source file. |
| `outputs/faiss/faiss_index_<file>/` | FAISS index, embeddings, and metadata for follow-up retrieval. |
| `outputs/uploads/<session_id>/` | Streamlit-uploaded files for the current analysis session. |

`outputs/`, `.env`, `SanitisedData/`, and local `tests/` are ignored by Git.

## Pipeline Summary

The active pipeline is a six-stage Map-Reduce workflow:

1. File discovery with `get_log_files_from_path`.
2. Structure detection and category-aware chunking/extraction.
3. Optional pre-embedding compression for large server-monitoring logs.
4. Evidence selection with anomaly scores, deterministic scores, neighbours, and budget limits.
5. Per-file map analysis with strict evidence grounding.
6. Cross-file reduce synthesis and PDF export.

For API/request logs, the deterministic path groups request lifecycles and signal events directly, then skips embedding/anomaly scoring. For server monitoring logs, the pipeline **by default** uses the legacy hybrid chunking + anomaly path described above. An opt-in alternative (`--mode server_monitoring` or the UI radio) loads UAM server metrics into DuckDB and lets the LLM perform iterative evidence-grounded SQL exploration instead (see Claude.md §2.2 for the full bifurcation and agentic loop details). The API request path is never affected.

## Citation Model

Internal evidence chunks still receive `[REF_...]` IDs so the LLM can ground its intermediate reasoning. Before report text is shown to a user, those internal references are replaced with readable original source references:

```text
Original Log Reference: sanitized_example.log, lines 614
Path: C:\...\outputs\uploads\<session>\sanitized_example.log
```

When the relevant error is inside a larger multi-line request chunk, the pipeline prefers `error_line_ranges` over the broader chunk range. This is why the report can cite the exact error line instead of only citing the whole request span.

## Configuration

Most domain tuning belongs in `search_config.json`, not in Python code:

- `iam_critical_keywords`: domain terms that should be boosted during evidence selection.
- `api_request_boundaries`: start/end markers for deterministic request extraction.
- `route_categories`: routing hints for API request versus server monitoring analysis.
- `buckets`: retrieval buckets for targeted evidence collection.

Model/provider settings belong in `.env` and are loaded by `config.py`.

## Reliability Notes

- One failed file should not crash the whole run; map failures return an explicit `# AGENT ERROR`.
- Debug evidence files are intentional diagnostics and are stored under `outputs/debug/`.
- FAISS artifacts are retained under `outputs/faiss/` so follow-up chat can inspect evidence without rerunning analysis.
- The app stores uploaded files under `outputs/uploads/` to preserve line references during the session.

## License

Internal use only.
