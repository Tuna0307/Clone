# AAI_2002-ITP: LLM Configuration and Deployment Notes

## Current Configuration Model

The project uses a provider-agnostic factory in `llm_factory.py`. The active provider is selected by `LLM_PROVIDER` in `.env`.

| Provider | Chat model examples | Embedding examples | Notes |
| :--- | :--- | :--- | :--- |
| OpenAI | `gpt-5.5`, `gpt-4o` | `text-embedding-3-small` | Current team direction. Requires `OPENAI_API_KEY`. |
| Bedrock | `us.meta.llama3-1-8b-instruct-v1:0` | `amazon.titan-embed-text-v2:0` | Supported fallback. Requires AWS credentials. |

The same pipeline code calls `get_llm()` and `get_embeddings()`; it should not import provider-specific clients directly outside `llm_factory.py`.

## Recommended `.env` Values

OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL_ID=gpt-5.5
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=16384
OPENAI_REASONING_EFFORT=none
EMBEDDING_MODEL_ID=text-embedding-3-small
```

Bedrock:

```env
LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
LLM_MODEL_ID=us.meta.llama3-1-8b-instruct-v1:0
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=8192
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
```

## Active Pipeline Constants

The active pipeline is `iam_log_intelligence_agent_hybridChunking2.py`.

| Constant | Value | Purpose |
| :--- | :--- | :--- |
| `MAP_EVIDENCE_BUDGET_CHARS` | `200_000` | Hard cap on evidence text sent to each map LLM call. |
| `MAP_TOP_N_CHUNKS` | `60` | Number of ranked seed chunks selected before neighbour expansion. |
| `MAP_MAX_CHUNKS` | `150` | Total chunk cap after ranked seeds and neighbours are merged. |
| `REDUCE_EVIDENCE_BUDGET_CHARS` | `160_000` | Hard cap on compiled reduce evidence. |
| `REDUCE_PER_FILE_CAP_CHARS` | `8_000` | Max per-file finding text included in reduce. |
| `LLM_MAX_TOKENS` | From `.env`, default `16384` | Max response tokens passed to the chat model. |
| `OPENAI_REASONING_EFFORT` | `none` for GPT-5-family OpenAI models | Keeps reasoning tokens from consuming the forensic report output budget. |
| `EMBEDDING_MAX_CHARS` | `16_000` in `hybridChunking2` | Conservative per-document embedding input cap. |

## Why the Factory Layer Matters

The pipeline has two model-dependent workloads:

1. Chat completion for routing, schema fallback, map reports, reduce reports, and follow-up answers.
2. Embeddings for server-monitoring anomaly scoring and follow-up retrieval.

Keeping provider setup in `llm_factory.py` prevents model changes from spreading across the pipeline. It also keeps Streamlit and CLI execution consistent.

## Output Artifacts

The model pipeline produces artifacts used by both the report and follow-up chat:

| Artifact | Path | Purpose |
| :--- | :--- | :--- |
| Debug evidence | `outputs/debug/debug_evidence_<file>.txt` | Exact map evidence prompt context for a source file. |
| FAISS index | `outputs/faiss/faiss_index_<file>/index.faiss` | Vector index for follow-up retrieval. |
| Metadata | `outputs/faiss/faiss_index_<file>/metadata.json` | Chunk metadata, line ranges, source paths, and scores. |
| PDF report | `outputs/reports/IAM_Forensic_Report.pdf` | Exported forensic report. |
| Uploaded logs | `outputs/uploads/<session_id>/` | Session-local copies of Streamlit-uploaded logs. |

## Citation Behavior

LLM prompts still cite internal `[REF_...]` IDs so every claim maps to a selected evidence item. Before showing the report to a user, the pipeline replaces those IDs with original log references:

```text
Original Log Reference: file.log, lines 614
Path: C:\...\file.log
```

For multi-line request chunks, `error_line_ranges` are preferred over broad `line_ranges`. This gives users the exact failure line when the chunk contains surrounding context.

## Local Model Alternative

The provider factory does not currently include a local/Ollama provider. If the team needs local inference later, add it inside `llm_factory.py` rather than replacing provider calls throughout the codebase.

Possible local options:

- Ollama chat model for LLM calls.
- Sentence-transformers or Ollama embeddings for vector work.
- A new `LLM_PROVIDER=ollama` branch with matching environment variables.

Any local provider change should be tested against both API/request logs and server-monitoring logs because embedding behavior affects anomaly ranking.
