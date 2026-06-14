"""
Legacy hybrid chunking pipeline retained for reference.

New development should target iam_log_intelligence_agent_hybridChunking2.py.
This older module preserves the first hybrid-thread/time pipeline design and is
useful for comparing behavior during regressions, but the Streamlit app no
longer imports it.
"""

# ============================================================================
# Imports
# ============================================================================

# Standard library
import os
import sys
import gc
import json
import re
import random
import hashlib
import shutil
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Iterator, Optional
from collections import defaultdict

# Third-party
import numpy as np
from botocore.exceptions import ClientError
from tqdm import tqdm

from llm_factory import get_llm, get_embeddings

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain ecosystem
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import FAISS as LangchainFAISS


# ============================================================================
# LLM and Embeddings Configuration (provider-agnostic via llm_factory)
# ============================================================================

llm = get_llm()
embeddings = get_embeddings()

# ============================================================================
# Hardware-Aware Context Budget Configuration
# ============================================================================
# Target: RTX 5080 (16 GB VRAM) + Llama 3.1 8B Q4_K_M (4.9 GB)
#
# VRAM breakdown:
#   Model weights (Q4_K_M) ....... 4.9 GB
#   CUDA / runtime overhead ....... 0.5 GB
#   Activations (batch=1) ......... 0.5 GB
#   Available for KV cache ........ 10.1 GB
#
# KV cache per token (Llama 3.1 8B, GQA with 8 KV heads, FP16):
#   2 (K+V) x 32 layers x 8 heads x 128 dim x 2 bytes = 128 KB/token
#   Max context: 10.1 GB / 128 KB ≈ 82,700 tokens
#   With 90% safety margin: ~74,000 tokens usable
#
# Per-phase token budget (~74K tokens):
#   System + user prompt .......... ~1,000 tokens
#   LLM output (max_tokens) ....... 8,192 tokens
#   Safety buffer ................. ~4,800 tokens
#   Evidence budget ............... ~60,000 tokens ≈ 240,000 chars
#
# Note: Llama 3.1 8B performs best with focused evidence (~30-40K tokens).
# We use ~60K as the hard ceiling, but aim for ~40K through chunk selection.
# ============================================================================

# -- Map phase (per-file analysis) --
MAP_EVIDENCE_BUDGET_CHARS: int = 200_000   # Hard cap on evidence chars sent to map LLM
MAP_TOP_N_CHUNKS: int = 60                 # Top anomaly-scored chunks to select
MAP_MAX_CHUNKS: int = 150                  # Max total chunks (anomaly + targeted + neighbours)
MAP_NEIGHBOUR_RADIUS: int = 2              # Temporal neighbours per selected chunk

# -- Chunk truncation --
ERROR_CHUNK_MAX_CHARS: int = 8_000         # Max chars for error-bearing chunks (preserve diagnostic detail)
BENIGN_CHUNK_MAX_CHARS: int = 2_000        # Max chars for benign/routine chunks

# -- Reduce phase (cross-file consolidation) --
REDUCE_EVIDENCE_BUDGET_CHARS: int = 160_000  # Hard cap on total compiled evidence
REDUCE_PER_FILE_CAP_CHARS: int = 8_000       # Max chars per file's findings in reduce

# -- Hybrid chunking --
MAX_GROUP_CHARS: int = 15_000              # Max chars before splitting a thread group
CHUNK_OVERLAP_CHARS: int = 500             # Overlap between split thread groups
NO_TS_CATCH_ALL_CHARS: int = 10_000        # Catch-all chunk size for no-timestamp lines
UNGROUPED_WINDOW_SECONDS: int = 120        # Time-window size for ungrouped timestamped lines
UNGROUPED_MAX_LINES_PER_CHUNK: int = 700   # Split oversized ungrouped windows by line count

# -- Pre-embedding compression (Stage 2.5; applies BEFORE anomaly scoring only) --
LARGE_LOG_CHUNK_TRIGGER: int = 3_000         # Enable conservative pre-embedding compression
VERY_LARGE_LOG_CHUNK_TRIGGER: int = 12_000   # Enable additional downselection for extreme chunk counts
MAX_EMBEDDING_CHUNKS_VERY_LARGE: int = 6_000  # Hard cap for chunks sent to embedding in very large mode
DEDUP_NUMERIC_TOKEN_MIN_LEN: int = 4         # Replace long numeric tokens during canonical dedup normalization

# -- Embedding safety --
EMBEDDING_MAX_CHARS: int = 48_000          # Titan Embed v2 hard limit is 50,000 chars; leave margin
EMBEDDING_CONCURRENCY: int = 5             # Parallel Bedrock embedding workers
EMBEDDING_MAX_RETRIES: int = 5             # Retry attempts for transient/throttling failures
EMBEDDING_BACKOFF_BASE_SECONDS: float = 1.5  # Exponential backoff base for embed retries

# -- Anomaly scoring --
ANOMALY_REF_SAMPLE_RATIO: float = 0.25    # Fraction of chunks sampled as "normal" reference
ANOMALY_REF_MAX: int = 600                # Max reference set size
ANOMALY_K_NEIGHBOURS: int = 6             # kNN for distance calculation
ANOMALY_HIGH_THRESHOLD: float = 2.5       # z-score threshold for "high-anomaly"
ERROR_SCORE_BOOST: float = 2.0            # Score boost for error-bearing chunks
IAM_CRITICAL_SCORE_BOOST: float = 4.0     # Extra boost for IAM-domain-critical chunks (stacks on ERROR_SCORE_BOOST)
NOISE_SCORE_PENALTY: float = 1.5          # Score penalty for noisy/benign chunks
NOISE_SCORE_CLAMP: float = 1.0            # Hard cap on anomaly score for noise-only chunks (prevents infra noise domination)

# (LLM_MAX_TOKENS is defined above the LLM initialization, line ~80)

# ============================================================================
# Search Configuration (Diversity Buckets)
# ============================================================================

# Keywords that indicate error-bearing content (used for score boosting)
_ERROR_KEYWORDS: list[str] = [
    'ERROR', 'Exception', 'FATAL', 'Failed', 'Refused', 'CRITICAL',
    'SecurityException', 'SessionInvalid', 'VerificationFailed',
    'NullPointerException', 'Caused by:', 'stack trace',
]

# Patterns for benign/noisy log lines (used for score penalty / clamping)
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r'Audit took \d+', re.IGNORECASE),
    re.compile(r'refreshSession.*success', re.IGNORECASE),
    re.compile(r'^\s*INFO\s.*\bstarted\b', re.IGNORECASE),
    re.compile(r'^\s*INFO\s.*\bhealthy\b', re.IGNORECASE),
    # DB / SQL infrastructure noise
    re.compile(r'ERRORCODE=.*SQLSTATE=', re.IGNORECASE),
    re.compile(r'Connection\s*(reset|refused)', re.IGNORECASE),
    re.compile(r'ConnectException', re.IGNORECASE),
    re.compile(r'Error opening socket to server', re.IGNORECASE),
    # WebSphere infrastructure noise
    re.compile(r'Start Display Current Environment', re.IGNORECASE),
    re.compile(r'End Display Current Environment', re.IGNORECASE),
    re.compile(r'log4j:WARN', re.IGNORECASE),
    re.compile(r'\[Fatal Error\].*Premature end of file', re.IGNORECASE),
    re.compile(r'\[Fatal Error\].*Element type', re.IGNORECASE),
    # HQL debug queries (routine DB operations, not errors)
    re.compile(r'hql:\s*select\b', re.IGNORECASE),
]

# Conservative normalization patterns used for exact/canonical dedup only
# (accuracy-first: avoid aggressive semantic collapsing)
_DEDUP_UUID_RE = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
)
_DEDUP_HEX_ADDR_RE = re.compile(r'\b0x[0-9a-fA-F]+\b')
_DEDUP_LONG_HEX_RE = re.compile(r'\b[0-9a-fA-F]{12,}\b')
_DEDUP_WS_RE = re.compile(r'\s+')


def load_search_config(config_path: str = "search_config.json") -> dict:
    """
    Load search configuration from JSON file.
    Returns default config if file is missing or invalid.

    Args:
        config_path: Path to the search config JSON file

    Returns:
        Dict with 'buckets' key containing list of search bucket configs
    """
    default_config: dict = {
        "buckets": [
            {
                "name": "DEFAULT_CRITICAL",
                "query": "Exception Error Critical Security",
                "top_k": 20
            }
        ]
    }

    if not os.path.exists(config_path):
        print(f"[Config] Warning: {config_path} not found. Using defaults.")
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            if "buckets" not in config or not isinstance(config["buckets"], list):
                print(f"[Config] Invalid format in {config_path}. Using defaults.")
                return default_config
            return config
    except Exception as e:
        print(f"[Config] Error loading {config_path}: {e}")
        return default_config


def decode_url_encoded_errors(text: str) -> str:
    """
    Decode URL-encoded segments within error messages to make them
    human-readable for LLM analysis.

    Handles common URL encoding in Java stack traces and IAM log messages
    (e.g., %24 -> $, %3D -> =, %26amp; -> &, + -> space).

    Args:
        text: Raw log text possibly containing URL-encoded segments

    Returns:
        Text with URL-encoded segments decoded
    """
    # Only decode lines that look URL-encoded (contain %XX patterns)
    lines = text.split('\n')
    decoded_lines: list[str] = []
    url_pattern = re.compile(r'%[0-9A-Fa-f]{2}')

    for line in lines:
        if url_pattern.search(line):
            try:
                # First pass: decode HTML entities like %26amp;
                cleaned = line.replace('%26amp;', '&')
                cleaned = cleaned.replace('%26gt;', '>')
                cleaned = cleaned.replace('%26lt;', '<')
                # Second pass: standard URL decoding
                decoded = urllib.parse.unquote_plus(cleaned)
                decoded_lines.append(decoded)
            except Exception:
                decoded_lines.append(line)
        else:
            decoded_lines.append(line)

    return '\n'.join(decoded_lines)


def _is_error_bearing(text: str) -> bool:
    """
    Determine whether a chunk contains error-bearing signals.

    Args:
        text: Chunk content

    Returns:
        True if text contains error indicators, else False
    """
    return any(keyword in text for keyword in _ERROR_KEYWORDS)


def _canonicalize_for_dedup(text: str, schema: dict) -> str:
    """
    Produce a conservative canonical form for pre-embedding deduplication.

    Notes:
        - This is intentionally conservative (accuracy-first).
        - It removes volatile identifiers while preserving lexical context.
        - It does NOT do semantic collapsing or fuzzy matching.

    Args:
        text: Raw chunk content
        schema: Detected schema from detect_log_structure

    Returns:
        Canonicalized text for exact/canonical hash dedup
    """
    canonical = text

    timestamp_re = schema.get('timestamp_re')
    if timestamp_re is not None and schema.get('timestamp_fmt'):
        canonical = timestamp_re.sub('<TS>', canonical)

    canonical = _DEDUP_UUID_RE.sub('<UUID>', canonical)
    canonical = _DEDUP_HEX_ADDR_RE.sub('<HEXADDR>', canonical)
    canonical = _DEDUP_LONG_HEX_RE.sub('<HEX>', canonical)

    canonical = re.sub(
        rf'\b\d{{{DEDUP_NUMERIC_TOKEN_MIN_LEN},}}\b',
        '<NUM>',
        canonical,
    )
    canonical = _DEDUP_WS_RE.sub(' ', canonical).strip()
    return canonical


def deduplicate_chunks_safe(docs: list[Document], schema: dict) -> list[Document]:
    """
    Perform conservative pre-embedding deduplication.

    Strategy:
        - Exact/canonical dedup only (no fuzzy matching).
        - Error-bearing chunks are deduped only with other error-bearing chunks.
        - Representative keeps original text/metadata; occurrence stats are merged.

    Args:
        docs: Chunk list from Stage 2
        schema: Detected schema for timestamp-aware canonicalization

    Returns:
        Deduplicated chunk list
    """
    if not docs:
        return docs

    print(f"  [Dedup] Conservative dedup on {len(docs):,} chunks...")

    unique_docs: list[Document] = []
    key_to_index: dict[str, int] = {}
    merged_count = 0

    for doc in docs:
        content = doc.page_content
        error_flag = _is_error_bearing(content)
        canonical = _canonicalize_for_dedup(content, schema)
        digest = hashlib.sha1(canonical.encode('utf-8', errors='ignore')).hexdigest()
        dedup_key = f"err={int(error_flag)}|h={digest}"

        if dedup_key not in key_to_index:
            rep = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            rep.metadata['dedup_count'] = 1
            rep.metadata['dedup_first_start'] = rep.metadata.get('start_time', '')
            rep.metadata['dedup_last_end'] = rep.metadata.get('end_time', '')
            unique_docs.append(rep)
            key_to_index[dedup_key] = len(unique_docs) - 1
            continue

        merged_count += 1
        rep_idx = key_to_index[dedup_key]
        rep = unique_docs[rep_idx]
        rep.metadata['dedup_count'] = int(rep.metadata.get('dedup_count', 1)) + 1

        current_first = rep.metadata.get('dedup_first_start', '')
        current_last = rep.metadata.get('dedup_last_end', '')
        candidate_start = doc.metadata.get('start_time', '')
        candidate_end = doc.metadata.get('end_time', '')

        if candidate_start and (not current_first or candidate_start < current_first):
            rep.metadata['dedup_first_start'] = candidate_start
        if candidate_end and (not current_last or candidate_end > current_last):
            rep.metadata['dedup_last_end'] = candidate_end

    reduction_pct = ((len(docs) - len(unique_docs)) / len(docs)) * 100.0
    print(f"    {len(docs):,} -> {len(unique_docs):,} unique "
          f"({reduction_pct:.1f}% reduction, {merged_count:,} merged)")
    return unique_docs


def downselect_chunks_for_embedding(docs: list[Document], max_chunks: int) -> list[Document]:
    """
    Downselect chunks only for very large files before embedding.

    Selection policy (accuracy-first):
        1. Keep all IAM-critical and error-bearing chunks first.
        2. Fill remaining capacity using round-robin across key_type buckets
           to preserve thread/time/no-timestamp diversity.

    Args:
        docs: Deduplicated chunk list
        max_chunks: Hard cap for chunks passed to embedding stage

    Returns:
        Selected chunk list (<= max_chunks)
    """
    if len(docs) <= max_chunks:
        return docs

    search_cfg = load_search_config()
    iam_critical_keywords: list[str] = search_cfg.get('iam_critical_keywords', [])

    selected: list[Document] = []
    selected_ids: set[int] = set()

    def _priority(doc: Document) -> tuple:
        content = doc.page_content
        is_critical = any(keyword in content for keyword in iam_critical_keywords)
        is_error = _is_error_bearing(content)
        dedup_count = int(doc.metadata.get('dedup_count', 1))
        line_count = int(doc.metadata.get('line_count', 0))
        return (is_critical, is_error, dedup_count, line_count)

    ordered_indices = sorted(
        range(len(docs)),
        key=lambda i: _priority(docs[i]),
        reverse=True,
    )

    for idx in ordered_indices:
        content = docs[idx].page_content
        is_critical = any(keyword in content for keyword in iam_critical_keywords)
        is_error = _is_error_bearing(content)
        if not (is_critical or is_error):
            continue
        selected.append(docs[idx])
        selected_ids.add(idx)
        if len(selected) >= max_chunks:
            break

    if len(selected) >= max_chunks:
        print(f"  [Downselect] {len(docs):,} -> {len(selected):,} (critical/error priority)")
        return selected[:max_chunks]

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx in ordered_indices:
        if idx in selected_ids:
            continue
        key_type = str(docs[idx].metadata.get('key_type', 'unknown'))
        buckets[key_type].append(idx)

    bucket_keys = sorted(buckets.keys())
    bucket_pos: dict[str, int] = {k: 0 for k in bucket_keys}
    made_progress = True

    while len(selected) < max_chunks and made_progress:
        made_progress = False
        for key in bucket_keys:
            pos = bucket_pos[key]
            if pos >= len(buckets[key]):
                continue
            idx = buckets[key][pos]
            bucket_pos[key] += 1
            selected.append(docs[idx])
            selected_ids.add(idx)
            made_progress = True
            if len(selected) >= max_chunks:
                break

    print(f"  [Downselect] {len(docs):,} -> {len(selected):,} "
          f"(very-large pre-embedding cap={max_chunks:,})")
    return selected


# ============================================================================
# Stage 1 — File Discovery (Unchanged)
# ============================================================================

def get_log_files_from_path(path: str) -> list[str]:
    """
    Recursively find all log files in a directory or return the single file.

    Args:
        path: Path to a file or directory

    Returns:
        List of absolute file paths
    """
    if os.path.isfile(path):
        return [os.path.abspath(path)]

    log_files: list[str] = []
    valid_extensions = {'.log', '.txt', '.out', '.err', '.msg'}

    print(f"-> Scanning directory: {path}")
    for root, _, files in os.walk(path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                full_path = os.path.join(root, file)
                log_files.append(os.path.abspath(full_path))

    print(f"   Found {len(log_files)} log files.")
    return log_files


def format_file_size(size_bytes: int) -> str:
    """
    Return human-readable file size string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like '524.6 MB'
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def stream_file_lines(file_path: str) -> Iterator[str]:
    """
    Generator that yields lines from a file one at a time.
    Memory-efficient for extremely large files.

    Args:
        file_path: Absolute path to the file

    Yields:
        Individual lines (with newline chars)
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            yield line

# ============================================================================
# Stage 2 — Preprocessing & Hybrid Chunking
# ============================================================================

# ---------------------------------------------------------------------------
# 2a. Automatic log-structure detection via regex heuristics
# ---------------------------------------------------------------------------

# Common timestamp patterns ordered by specificity (most specific first)
_TIMESTAMP_PATTERNS: list[tuple[str, str]] = [
    # ISO-style with millis — "2025-09-12 14:27:45.798"
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', '%Y-%m-%d %H:%M:%S.%f'),
    # ISO-style no millis — "2025-09-12 14:27:45"
    (r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '%Y-%m-%d %H:%M:%S'),
    # ISO-8601 T separator — "2025-09-12T14:27:45.798"
    (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}', '%Y-%m-%dT%H:%M:%S.%f'),
    # US date with slashes + 4-digit year — "09/12/2025 14:27:45"
    (r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '%m/%d/%Y %H:%M:%S'),
    # WebSphere colon-millis + 2-digit year — "[11/24/25 13:59:01:674 BNT]" or "[5/9/18 3:11:25:585 BNT]"
    # Uses colon before millis (HH:MM:SS:mmm) which strptime can't parse directly;
    # _parse_line normalises the last colon to a dot before calling strptime.
    # Hour can be single-digit (e.g. 3:11:25) in WebSphere logs.
    (r'\d{1,2}/\d{1,2}/\d{2} \d{1,2}:\d{2}:\d{2}:\d{3}', '%m/%d/%y %H:%M:%S.%f'),
    # WebSphere colon-millis + 4-digit year — "[11/24/2025 13:59:01:674 BNT]"
    (r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}:\d{3}', '%m/%d/%Y %H:%M:%S.%f'),
]

# Common thread-name patterns
_THREAD_PATTERNS: list[str] = [
    r'\[(https?-[\w-]+-exec-\d+)\]',          # Tomcat NIO executor [http-nio2-8080-exec-307], [https-jsse-nio-7284-exec-7]
    r'\[(https?-[\w-]+-\d+)\]',               # Tomcat-style (other) [https-jsse-nio-7284-exec-7]
    r'\[([\w]+-[\w]+-\d+-\d+)\]',             # Generic word-word-num-num
    r'\[([A-Z][A-Za-z]+ [\w-]+-\d+-\d+)\]',  # Named workers [AM EventWorker-1-46]
    r'\]\s+([0-9a-f]{8})\s',                   # WebSphere hex thread ID: [timestamp] 000000c9 ComponentName
    r'\[([\w.-]+)\]',                         # Broad fallback [threadName]
]

# Session / transaction key patterns (key=value or key:value or key/value)
_SESSION_KEY_PATTERNS: list[tuple[str, str]] = [
    (r'(?:txId|tx_id|TXID)[=:/]\s*(\S+)', 'txId'),
    (r'(?:sesId|SESSION_ID|session_id|sessionId)[=:/]\s*(\S+)', 'sesId'),
    (r'(?:IID|iid)[=:/]\s*(\S+)', 'iid'),
    (r'(?:USER_ID|userId|user_id)[=:/]\s*(\S+)', 'userId'),
    (r'(?:e2eeSid)[=:/]\s*(\S+)', 'e2eeSid'),
    (r'(?:correlationId|CORRELATION_ID|corrId)[=:/]\s*(\S+)', 'correlationId'),
]

# Regex to strip "[Line NNN] " prefixes added by pre-filter tools
_LINE_PREFIX_RE = re.compile(r'^\[Line \d+\]\s*')


def detect_log_structure(sample_lines: list[str]) -> dict:
    """
    Detect the log schema from the first 800-1200 sample lines using regex
    heuristics.  Returns a dict with compiled regexes and field names so
    downstream functions can parse efficiently without re-compiling.

    Args:
        sample_lines: First 800-1200 lines of the log file

    Returns:
        Schema dict with keys:
            timestamp_re   — compiled regex with one group for the timestamp string
            timestamp_fmt  — strptime format string
            thread_re      — compiled regex with one group for thread name (or None)
            session_keys   — list of (compiled_regex, key_name) tuples
            stack_trace_re — compiled regex to identify continuation / stack trace lines
    """
    schema: dict = {
        'timestamp_re': None,
        'timestamp_fmt': None,
        'thread_re': None,
        'session_keys': [],
        'stack_trace_re': re.compile(
            r'^(?:\s+at |\s*Caused by:|\s*\.\.\. \d+ more|'
            r'\s*[a-zA-Z][\w.$]*(?:Exception|Error|Throwable))'
        ),
    }

    # Strip optional "[Line N] " prefixes from sample lines
    cleaned: list[str] = [_LINE_PREFIX_RE.sub('', l) for l in sample_lines]

    # Pre-filter: exclude stack-trace continuation lines from detection pool.
    # Files with many exceptions can have >80% stack-trace lines, which dilute
    # the hit rate for timestamp/thread patterns below detection thresholds.
    _continuation_re = re.compile(
        r'^(?:\s+at |\s*Caused by:|\s*\.\.\. \d+ more'
        r'|\s+[a-zA-Z][\w.$]*(?:Exception|Error|Throwable)'
        r'|\s*~\[)'                      # Gradle/Maven module hints  ~[am.jar:?]
    )
    primary_lines: list[str] = [
        l for l in cleaned[:1200]
        if l.strip() and not _continuation_re.match(l)
    ]
    # Use up to 200 primary (non-continuation) lines for detection
    detect_pool: list[str] = primary_lines[:200]
    detect_pool_size: int = len(detect_pool) if detect_pool else 1  # avoid div-by-zero

    # --- Detect timestamp pattern ---
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        compiled = re.compile(pattern)
        hits = sum(1 for l in detect_pool if compiled.search(l))
        if hits > detect_pool_size * 0.3:  # 30 % threshold
            schema['timestamp_re'] = re.compile(f'({pattern})')
            schema['timestamp_fmt'] = fmt
            break

    # Fallback: if no timestamp detected, use a permissive regex that won't match
    if schema['timestamp_re'] is None:
        schema['timestamp_re'] = re.compile(r'(NOTIMESTAMP)')
        schema['timestamp_fmt'] = ''

    # --- Detect thread pattern ---
    for pat in _THREAD_PATTERNS:
        compiled = re.compile(pat)
        hits = sum(1 for l in detect_pool if compiled.search(l))
        if hits > detect_pool_size * 0.2:  # 20 % threshold
            schema['thread_re'] = compiled
            break

    # --- Detect session / transaction keys ---
    # Use a wider window (up to 400 primary lines) for session key detection
    session_pool: list[str] = primary_lines[:400]
    for pat, key_name in _SESSION_KEY_PATTERNS:
        compiled = re.compile(pat, re.IGNORECASE)
        hits = sum(1 for l in session_pool if compiled.search(l))
        if hits > 0:
            schema['session_keys'].append((compiled, key_name))

    return schema


# ---------------------------------------------------------------------------
# 2b. Parse a single line into (timestamp, primary_key, raw_text)
# ---------------------------------------------------------------------------

def _parse_line(line: str, schema: dict) -> tuple[Optional[datetime], Optional[str], str]:
    """
    Parse one log line using the detected schema.

    Args:
        line:   Raw log line (may include "[Line N] " prefix)
        schema: Dict returned by detect_log_structure

    Returns:
        (timestamp or None, primary_key or None, cleaned line text)
    """
    clean = _LINE_PREFIX_RE.sub('', line).rstrip('\n')

    # --- Timestamp ---
    ts: Optional[datetime] = None
    ts_match = schema['timestamp_re'].search(clean)
    if ts_match and schema['timestamp_fmt']:
        try:
            ts_str = ts_match.group(1)
            # Normalise WebSphere colon-millis (HH:MM:SS:mmm -> HH:MM:SS.mmm)
            # so strptime's %f can parse it correctly.
            if '.%f' in schema['timestamp_fmt'] and ':' == ts_str[-4:-3]:
                ts_str = ts_str[:-4] + '.' + ts_str[-3:]
            ts = datetime.strptime(ts_str, schema['timestamp_fmt'])
        except (ValueError, IndexError):
            pass

    # --- Primary key: prefer thread, then first session key ---
    pk: Optional[str] = None
    if schema['thread_re'] is not None:
        m = schema['thread_re'].search(clean)
        if m:
            pk = m.group(1)
    # If no thread, try session keys
    if pk is None:
        for compiled, key_name in schema['session_keys']:
            m = compiled.search(clean)
            if m:
                pk = f"{key_name}:{m.group(1)}"
                break

    return ts, pk, clean


# ---------------------------------------------------------------------------
# 2c. Hybrid chunking: thread/session groups + time-window fallback
# ---------------------------------------------------------------------------

def hybrid_chunk_log(file_path: str, schema: dict) -> list[Document]:
    """
    Stream a log file and produce semantically coherent Document chunks.

    Strategy:
        1. Parse every line for (timestamp, primary_key, text).
        2. Group lines sharing the same primary_key together.
           - Stack-trace continuation lines (no timestamp) inherit the key
             of the preceding line.
           - If a group exceeds 15 000 chars, split chronologically with
             500-char overlap.
        3. Lines with no primary_key are collected separately and chunked
           using 2-minute sliding windows (30-second step) sorted by timestamp.

    Args:
        file_path: Absolute path to the log file
        schema:    Dict returned by detect_log_structure

    Returns:
        List of Document objects with rich metadata
    """
    file_name = os.path.basename(file_path)

    # ---- Pass 1: Stream & parse lines ----
    grouped: dict[str, list[tuple[Optional[datetime], str]]] = defaultdict(list)
    ungrouped: list[tuple[Optional[datetime], str]] = []

    last_pk: Optional[str] = None
    line_count = 0

    print(f"  [Chunk] Parsing lines from {file_name}...")
    for raw_line in stream_file_lines(file_path):
        line_count += 1
        ts, pk, clean = _parse_line(raw_line, schema)

        # Stack-trace continuation inherits previous primary key
        if pk is None and schema['stack_trace_re'].match(clean):
            pk = last_pk

        if pk is not None:
            grouped[pk].append((ts, clean))
            last_pk = pk
        else:
            ungrouped.append((ts, clean))

    print(f"    {line_count:,} lines parsed -> {len(grouped):,} thread/session groups, "
          f"{len(ungrouped):,} ungrouped lines")

    # ---- Helper: build Document from a list of (ts, text) ----
    def _make_doc(
        lines: list[tuple[Optional[datetime], str]],
        key_label: str,
        key_value: str,
        sub_index: int = 0,
    ) -> Document:
        """
        Build a Document from parsed lines with metadata.

        Args:
            lines:     List of (timestamp, text) tuples
            key_label: Type of key (thread, time_window, no_timestamp)
            key_value: Value of the primary key
            sub_index: Sub-chunk index for split groups

        Returns:
            Document with page_content and metadata
        """
        timestamps = [t for t, _ in lines if t is not None]
        start = min(timestamps) if timestamps else None
        end = max(timestamps) if timestamps else None
        content = '\n'.join(text for _, text in lines)
        return Document(
            page_content=content,
            metadata={
                'source_file': file_name,
                'primary_key': key_value,
                'key_type': key_label,
                'start_time': start.isoformat() if start else '',
                'end_time': end.isoformat() if end else '',
                'line_count': len(lines),
                'sub_index': sub_index,
            },
        )

    docs: list[Document] = []

    # ---- Pass 2a: Grouped chunks (thread / session) ----
    for pk, entries in grouped.items():
        total_chars = sum(len(t) for _, t in entries)

        if total_chars <= MAX_GROUP_CHARS:
            docs.append(_make_doc(entries, 'thread', pk))
        else:
            # Split chronologically with overlap
            # Sort by timestamp (None timestamps go to the end)
            entries.sort(key=lambda x: x[0] if x[0] else datetime.max)
            chunk_lines: list[tuple[Optional[datetime], str]] = []
            chunk_chars = 0
            sub = 0
            overlap_buffer: list[tuple[Optional[datetime], str]] = []

            for entry in entries:
                chunk_lines.append(entry)
                chunk_chars += len(entry[1]) + 1  # +1 for newline

                if chunk_chars >= MAX_GROUP_CHARS:
                    docs.append(_make_doc(chunk_lines, 'thread', pk, sub))
                    sub += 1
                    # Keep last CHUNK_OVERLAP_CHARS worth of lines as overlap
                    overlap_buffer = []
                    buf_chars = 0
                    for e in reversed(chunk_lines):
                        buf_chars += len(e[1]) + 1
                        overlap_buffer.insert(0, e)
                        if buf_chars >= CHUNK_OVERLAP_CHARS:
                            break
                    chunk_lines = list(overlap_buffer)
                    chunk_chars = sum(len(t) + 1 for _, t in chunk_lines)

            if chunk_lines:
                docs.append(_make_doc(chunk_lines, 'thread', pk, sub))

    # ---- Pass 2b: Ungrouped lines -> time windows ----
    if ungrouped:
        # Filter to only lines with timestamps, sort
        with_ts = [(t, txt) for t, txt in ungrouped if t is not None]
        without_ts = [(t, txt) for t, txt in ungrouped if t is None]

        if with_ts:
            with_ts.sort(key=lambda x: x[0])
            window_duration = timedelta(seconds=UNGROUPED_WINDOW_SECONDS)
            # Non-overlapping windows reduce duplicate chunk generation in dense logs.
            step = window_duration
            window_start = with_ts[0][0]
            window_end = with_ts[-1][0] + timedelta(seconds=1)
            win_idx = 0
            ptr = 0
            total_with_ts = len(with_ts)

            while window_start < window_end:
                w_end = window_start + window_duration
                window_lines: list[tuple[Optional[datetime], str]] = []

                while ptr < total_with_ts and with_ts[ptr][0] < window_start:
                    ptr += 1

                scan_ptr = ptr
                while scan_ptr < total_with_ts and with_ts[scan_ptr][0] < w_end:
                    window_lines.append(with_ts[scan_ptr])
                    scan_ptr += 1

                if window_lines:
                    label = f"time:{window_start.strftime('%H:%M:%S')}-{w_end.strftime('%H:%M:%S')}"
                    total_win_chars = sum(len(txt) + 1 for _, txt in window_lines)
                    if total_win_chars <= MAX_GROUP_CHARS and len(window_lines) <= UNGROUPED_MAX_LINES_PER_CHUNK:
                        docs.append(_make_doc(window_lines, 'time_window', label, win_idx))
                        win_idx += 1
                    else:
                        # Split oversized time-window into sub-chunks
                        tw_batch: list[tuple[Optional[datetime], str]] = []
                        tw_chars = 0
                        tw_lines = 0
                        for entry in window_lines:
                            tw_batch.append(entry)
                            tw_chars += len(entry[1]) + 1
                            tw_lines += 1
                            if tw_chars >= MAX_GROUP_CHARS or tw_lines >= UNGROUPED_MAX_LINES_PER_CHUNK:
                                docs.append(_make_doc(tw_batch, 'time_window', label, win_idx))
                                win_idx += 1
                                tw_batch = []
                                tw_chars = 0
                                tw_lines = 0
                        if tw_batch:
                            docs.append(_make_doc(tw_batch, 'time_window', label, win_idx))
                            win_idx += 1
                ptr = scan_ptr
                window_start += step

        # Lines with no timestamp at all -> single catch-all chunk (capped at ~10K)
        if without_ts:
            sub = 0
            batch: list[tuple[Optional[datetime], str]] = []
            batch_chars = 0
            for entry in without_ts:
                batch.append(entry)
                batch_chars += len(entry[1]) + 1
                if batch_chars >= NO_TS_CATCH_ALL_CHARS:
                    docs.append(_make_doc(batch, 'no_timestamp', 'unstructured', sub))
                    sub += 1
                    batch = []
                    batch_chars = 0
            if batch:
                docs.append(_make_doc(batch, 'no_timestamp', 'unstructured', sub))

    print(f"    {len(docs):,} chunks produced")
    return docs

# ============================================================================
# Stage 3 — Embedding & Runtime Anomaly Scoring (Ladle-inspired)
# ============================================================================

def _embed_batch_with_retry(
    batch_index: int,
    batch_texts: list[str],
) -> tuple[int, list[list[float]]]:
    """
    Embed one batch with retry/backoff for transient Bedrock failures.

    Args:
        batch_index: Sequential batch index for deterministic re-ordering
        batch_texts: Text payload for one embedding API call

    Returns:
        Tuple of (batch index, embedding vectors)
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            vectors = embeddings.embed_documents(batch_texts)
            return batch_index, vectors
        except ClientError as exc:
            last_error = exc
            err_code = exc.response.get("Error", {}).get("Code", "")
            retryable = err_code in {
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "InternalServerException",
            }
            if not retryable or attempt == EMBEDDING_MAX_RETRIES:
                raise

            sleep_seconds = EMBEDDING_BACKOFF_BASE_SECONDS ** attempt
            print(
                f"    [Embed Retry] batch={batch_index} attempt={attempt} "
                f"code={err_code} sleep={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)
        except Exception as exc:
            last_error = exc
            if attempt == EMBEDDING_MAX_RETRIES:
                raise

            sleep_seconds = EMBEDDING_BACKOFF_BASE_SECONDS ** attempt
            print(
                f"    [Embed Retry] batch={batch_index} attempt={attempt} "
                f"sleep={sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding failed with unknown error")

def _embed_documents_batched(
    docs: list[Document],
    batch_size: int = 50,
    label: str = "Embedding",
) -> np.ndarray:
    """
    Embed a list of Documents in batches using Titan embeddings.
    Returns a 2-D numpy array of shape (n_docs, dim).

    Args:
        docs:       List of Document objects
        batch_size: Number of texts per API call
        label:      Label for the progress bar

    Returns:
        np.ndarray of shape (n_docs, embedding_dim)
    """
    # Truncate texts to Titan Embed v2 character limit (50K hard limit)
    texts = [d.page_content[:EMBEDDING_MAX_CHARS] for d in docs]
    batches: list[list[str]] = [
        texts[i:i + batch_size]
        for i in range(0, len(texts), batch_size)
    ]
    ordered_batch_vectors: list[Optional[list[list[float]]]] = [None] * len(batches)

    with tqdm(total=len(texts), desc=f"    {label}", unit=" docs",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
            future_to_batch_info = {
                executor.submit(_embed_batch_with_retry, idx, batch): (idx, len(batch))
                for idx, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch_info):
                _, batch_len = future_to_batch_info[future]
                batch_index, batch_vectors = future.result()
                ordered_batch_vectors[batch_index] = batch_vectors
                pbar.update(batch_len)

    all_vecs: list[list[float]] = []
    for batch_vectors in ordered_batch_vectors:
        if batch_vectors is None:
            raise RuntimeError("Missing embedding batch result")
        all_vecs.extend(batch_vectors)

    return np.array(all_vecs, dtype=np.float32)


def score_anomalies(
    docs: list[Document],
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> list[Document]:
    """
    Zero-shot runtime anomaly scoring inspired by the Ladle paper.

    Steps:
        1. Embed all chunks.
        2. Sample 20-30% of chunks (max 600) as "normal" reference set.
        3. Build an in-memory FAISS index on the reference embeddings.
        4. For every chunk, query k=6 nearest neighbours -> mean Euclidean distance.
        5. Compute trimmed mean & std on reference distances (remove top/bottom 10%).
        6. z_score = (distance - trimmed_mean) / std_dev for each chunk.
        7. Annotate metadata with anomaly_score and raw_distance.

    Args:
        docs:                  List of Document chunks (from hybrid_chunk_log)
        precomputed_embeddings: Optional precomputed embedding matrix of shape
                                (n_docs, embedding_dim) to avoid re-embedding

    Returns:
        Same list of Documents with added anomaly metadata, sorted by score desc
    """
    import faiss  # Local import — only needed during scoring

    n = len(docs)
    if n == 0:
        return docs

    print(f"\n  [Anomaly] Scoring {n:,} chunks...")

    owns_embeddings = precomputed_embeddings is None

    # 1. Embed all chunks (or reuse precomputed embeddings)
    if precomputed_embeddings is None:
        all_embeddings = _embed_documents_batched(docs, batch_size=50, label="Embedding chunks")
    else:
        all_embeddings = precomputed_embeddings
        if all_embeddings.shape[0] != n:
            raise ValueError(
                f"Precomputed embeddings row count mismatch: {all_embeddings.shape[0]} != {n}"
            )
    dim = all_embeddings.shape[1]

    # 2. Sample reference set (20-30%, capped)
    ref_size = min(max(int(n * ANOMALY_REF_SAMPLE_RATIO), 1), ANOMALY_REF_MAX)
    if ref_size >= n:
        ref_indices = list(range(n))
    else:
        ref_indices = sorted(random.sample(range(n), ref_size))

    ref_embeddings = all_embeddings[ref_indices]
    print(f"    Reference set: {len(ref_indices):,} chunks")

    # 3. Build FAISS index on reference embeddings (L2 / Euclidean)
    index = faiss.IndexFlatL2(dim)
    index.add(ref_embeddings)

    # 4. Query every chunk against the reference index
    k = min(ANOMALY_K_NEIGHBOURS, len(ref_indices))
    distances, _ = index.search(all_embeddings, k)  # shape (n, k)
    mean_distances = distances.mean(axis=1)          # shape (n,)

    # 5. Compute baseline from reference set only
    ref_distances = mean_distances[ref_indices]
    sorted_ref = np.sort(ref_distances)
    trim = max(1, int(len(sorted_ref) * 0.10))
    trimmed = sorted_ref[trim:-trim] if trim < len(sorted_ref) // 2 else sorted_ref
    trimmed_mean_val = float(trimmed.mean())
    std_val = float(trimmed.std()) if len(trimmed) > 1 else 1.0

    # Guard against near-zero std
    if std_val < 1e-6:
        std_val = 1.0

    # 6. Compute z-scores
    print(f"    Baseline: trimmed_mean={trimmed_mean_val:.4f}, std={std_val:.4f}")

    for i, doc in enumerate(docs):
        raw_dist = float(mean_distances[i])
        z = (raw_dist - trimmed_mean_val) / std_val
        doc.metadata['anomaly_score'] = round(z, 4)
        doc.metadata['raw_distance'] = round(raw_dist, 4)

    # 7. Three-tier score adjustment: IAM-critical boost, generic error boost, noise suppression
    #
    # Strategy:
    #   Tier 1 — IAM-critical: chunks with domain-critical keywords (from search_config.json)
    #            get ERROR_SCORE_BOOST + IAM_CRITICAL_SCORE_BOOST = +6.0.
    #   Tier 2 — Generic error: chunks with error keywords but no IAM signal get +2.0.
    #   Tier 3 — Noise clamping: chunks matching noise patterns (DB/SQL, WebSphere infra,
    #            log4j, HQL) WITHOUT IAM-critical content are clamped to NOISE_SCORE_CLAMP
    #            regardless of their raw z-score. This prevents infrastructure noise
    #            from dominating over actual IAM diagnostic content.

    # Load IAM-critical keywords from search_config.json (follows AGENTS.md: no hardcoding)
    _search_cfg = load_search_config()
    iam_critical_keywords: list[str] = _search_cfg.get('iam_critical_keywords', [])

    error_boosted = 0
    iam_critical_boosted = 0
    noise_clamped = 0
    noise_penalised = 0

    for doc in docs:
        content = doc.page_content
        score = doc.metadata['anomaly_score']

        # --- Detect content categories ---
        has_iam_critical = any(kw in content for kw in iam_critical_keywords)
        has_error = any(kw in content for kw in _ERROR_KEYWORDS)
        is_noisy = any(p.search(content) for p in _NOISE_PATTERNS)
        is_structural_noise = doc.metadata.get('key_type') == 'no_timestamp'

        doc.metadata['iam_critical'] = has_iam_critical

        # --- Tier 1: IAM-critical boost (highest priority) ---
        if has_iam_critical:
            doc.metadata['anomaly_score'] = round(
                score + ERROR_SCORE_BOOST + IAM_CRITICAL_SCORE_BOOST, 4
            )
            doc.metadata['error_boosted'] = True
            doc.metadata['noise_penalised'] = False
            iam_critical_boosted += 1
            error_boosted += 1
            continue  # Skip further adjustments — IAM-critical is protected

        # --- Tier 2: Generic error boost (only if NOT noisy infrastructure) ---
        if has_error and not is_noisy:
            doc.metadata['anomaly_score'] = round(score + ERROR_SCORE_BOOST, 4)
            doc.metadata['error_boosted'] = True
            error_boosted += 1
        else:
            doc.metadata['error_boosted'] = False

        # --- Tier 3: Noise suppression with score clamping ---
        # Noisy chunks or structural noise (no-timestamp fragments) without
        # IAM signal get their score hard-capped. This prevents DB connection
        # errors, WebSphere env dumps, log4j warnings, and HQL queries from
        # outscoring actual IAM diagnostic content.
        if is_noisy or (is_structural_noise and not has_error):
            clamped_score = min(doc.metadata['anomaly_score'], NOISE_SCORE_CLAMP)
            if clamped_score < doc.metadata['anomaly_score']:
                doc.metadata['anomaly_score'] = round(clamped_score, 4)
                doc.metadata['noise_penalised'] = True
                noise_clamped += 1
            else:
                doc.metadata['noise_penalised'] = False
        else:
            # Penalise: short benign chunks (< 3 lines, no errors)
            line_count = content.count('\n') + 1
            if not has_error and (line_count < 3 and score < 3.0):
                doc.metadata['anomaly_score'] = round(
                    doc.metadata['anomaly_score'] - NOISE_SCORE_PENALTY, 4
                )
                doc.metadata['noise_penalised'] = True
                noise_penalised += 1
            else:
                doc.metadata['noise_penalised'] = False

    print(f"    Score adjustments: {error_boosted} error-boosted "
          f"({iam_critical_boosted} IAM-critical), "
          f"{noise_clamped} noise-clamped, {noise_penalised} noise-penalised")

    # Sort descending by anomaly score
    docs.sort(key=lambda d: d.metadata['anomaly_score'], reverse=True)

    high_count = sum(1 for d in docs if d.metadata['anomaly_score'] > ANOMALY_HIGH_THRESHOLD)
    print(f"    {high_count:,} chunks flagged as high-anomaly (z > {ANOMALY_HIGH_THRESHOLD})")

    # Clean up FAISS index from memory
    if owns_embeddings:
        del index, all_embeddings, ref_embeddings, distances
    else:
        del index, ref_embeddings, distances
    gc.collect()

    return docs

# ============================================================================
# Stage 3b — Targeted Retrieval (Diversity Bucket Search)
# ============================================================================

def retrieve_targeted_chunks(
    docs: list[Document],
    file_name: str = "",
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> set[int]:
    """
    Run targeted semantic search using search_config.json diversity buckets.
    This guarantees retrieval of domain-critical chunks (SecurityException,
    Certificate, Session, Token, etc.) regardless of anomaly score.

    Uses LangChain's FAISS wrapper to build a temporary in-memory index,
    then runs each bucket's query to find matching chunks.

    Args:
        docs:                  List of Document chunks (from hybrid_chunk_log)
        file_name:             Base name of the source file (used for saving vectorstore)
        precomputed_embeddings: Optional precomputed embedding matrix to avoid
                                re-embedding during vectorstore construction

    Returns:
        Set of indices into the docs list that matched targeted queries
    """
    if not docs:
        return set()

    config = load_search_config()
    buckets = config.get('buckets', [])
    if not buckets:
        return set()

    print(f"  [Targeted] Running {len(buckets)} diversity bucket searches...")

    # Build temporary LangChain FAISS index
    # Prefer precomputed embeddings to avoid a second embedding pass.
    # Fallback to from_documents if precomputed embeddings are unavailable.
    embedding_texts: list[str] = []
    embedding_docs: list[Document] = []
    for doc in docs:
        if len(doc.page_content) > EMBEDDING_MAX_CHARS:
            truncated_content = doc.page_content[:EMBEDDING_MAX_CHARS]
            embedding_docs.append(Document(
                page_content=truncated_content,
                metadata=doc.metadata.copy(),
            ))
            embedding_texts.append(truncated_content)
        else:
            embedding_docs.append(doc)
            embedding_texts.append(doc.page_content)

    try:
        can_reuse_precomputed = precomputed_embeddings is not None
        aligned_embeddings: Optional[np.ndarray] = None

        if can_reuse_precomputed and precomputed_embeddings.shape[0] != len(docs):
            print(
                "    Precomputed embeddings shape mismatch; "
                "falling back to on-demand targeted retrieval embeddings"
            )
            can_reuse_precomputed = False

        if can_reuse_precomputed:
            original_indices: list[int] = []
            for doc in docs:
                original_idx = doc.metadata.get('original_doc_index')
                if (
                    not isinstance(original_idx, int)
                    or original_idx < 0
                    or original_idx >= precomputed_embeddings.shape[0]
                ):
                    can_reuse_precomputed = False
                    break
                original_indices.append(original_idx)

            if can_reuse_precomputed:
                aligned_embeddings = precomputed_embeddings[np.array(original_indices, dtype=np.int64)]
            else:
                print(
                    "    Missing/invalid original_doc_index metadata; "
                    "falling back to on-demand targeted retrieval embeddings"
                )

        if can_reuse_precomputed and aligned_embeddings is not None:
            text_embeddings = [
                (embedding_texts[i], aligned_embeddings[i].tolist())
                for i in range(len(embedding_texts))
            ]
            vectorstore = LangchainFAISS.from_embeddings(
                text_embeddings,
                embeddings,
                metadatas=[doc.metadata.copy() for doc in embedding_docs],
            )
            print("    Reused aligned precomputed embeddings for targeted retrieval index")
        else:
            vectorstore = LangchainFAISS.from_documents(embedding_docs, embeddings)
            print("    Built targeted retrieval index with on-demand embeddings")
    except Exception as e:
        print(f"  [Targeted] Failed to build index: {e}")
        return set()

    # Create content -> index mapping for fast lookup
    content_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(docs):
        # Use first 200 chars as key to avoid hash collisions on long content
        key = doc.page_content[:200]
        content_to_indices[key].append(idx)

    targeted_indices: set[int] = set()
    targeted_details: list[str] = []

    for bucket in buckets:
        bucket_name = bucket.get('name', 'Unknown')
        query = bucket.get('query', 'Error')
        top_k = bucket.get('top_k', 10)

        try:
            results = vectorstore.similarity_search(query, k=top_k)
            matched = 0
            for result in results:
                # Only include chunks containing error-related content
                content = result.page_content
                has_signal = any(kw in content for kw in
                    ['Exception', 'Error', 'Failed', 'Refused', 'FATAL',
                     'CRITICAL', 'SecurityException', 'SessionInvalid',
                     'VerificationFailed', 'Caused by:'])
                if has_signal:
                    key = content[:200]
                    if key in content_to_indices:
                        for idx in content_to_indices[key]:
                            targeted_indices.add(idx)
                            # Tag the doc with its bucket source
                            docs[idx].metadata['targeted_bucket'] = bucket_name
                        matched += 1
            targeted_details.append(f"{bucket_name}: {matched} matches")
        except Exception as e:
            print(f"    [Targeted] Bucket '{bucket_name}' failed: {e}")
            continue

    # Save vectorstore for post-analysis (embeddings + documents + metadata)
    if file_name:
        save_dir = f"faiss_index_{file_name}"
        vectorstore.save_local(save_dir)
        print(f"    Saved vectorstore to '{save_dir}/'")

    # Clean up
    del vectorstore
    gc.collect()

    print(f"    Targeted retrieval: {len(targeted_indices)} unique chunks")
    for detail in targeted_details:
        print(f"      {detail}")

    return targeted_indices


# ============================================================================
# Stage 4 — Prioritized Retrieval for Map Phase
# ============================================================================

def select_evidence_chunks(
    scored_docs: list[Document],
    top_n: int = MAP_TOP_N_CHUNKS,
    neighbour_radius: int = MAP_NEIGHBOUR_RADIUS,
    targeted_indices: Optional[set[int]] = None,
    max_total_chars: int = MAP_EVIDENCE_BUDGET_CHARS,
) -> str:
    """
    Select evidence chunks using a hybrid strategy:
      1. Top-N anomaly-scored chunks (statistical outliers)
      2. Targeted retrieval chunks (domain-specific bucket matches)
      3. Temporal/thread neighbours for context

    URL-encoded error messages are decoded for LLM readability.
    Error-bearing chunks are NOT truncated to preserve diagnostic detail.
    A hard total character budget prevents context window overflow.

    Args:
        scored_docs:       Documents sorted by anomaly_score descending
        top_n:             Number of highest-scoring chunks to select
        neighbour_radius:  Number of chunks before/after to include for context
        targeted_indices:  Optional set of indices from targeted retrieval
        max_total_chars:   Hard cap on total evidence characters

    Returns:
        Formatted evidence string ready for the LLM prompt
    """
    n = len(scored_docs)
    if n == 0:
        return ""

    if targeted_indices is None:
        targeted_indices = set()

    # Build a lookup from (source_file, primary_key) -> list of indices in
    # scored_docs so we can find neighbours within the same group
    key_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, doc in enumerate(scored_docs):
        group_key = (
            doc.metadata.get('source_file', ''),
            doc.metadata.get('primary_key', ''),
        )
        key_to_indices[str(group_key)].append(idx)

    # Select top-N anomaly indices
    selected_indices: set[int] = set()
    top_indices = list(range(min(top_n, n)))
    selected_indices.update(top_indices)

    # Merge targeted retrieval indices (guaranteed domain-relevant chunks)
    selected_indices.update(targeted_indices)

    # Add neighbours within the same thread/session group
    seed_indices = list(selected_indices)
    for idx in seed_indices:
        if idx >= n:
            continue
        doc = scored_docs[idx]
        group_key = str((
            doc.metadata.get('source_file', ''),
            doc.metadata.get('primary_key', ''),
        ))
        group_indices = key_to_indices[group_key]
        pos = group_indices.index(idx) if idx in group_indices else -1
        if pos >= 0:
            for delta in range(-neighbour_radius, neighbour_radius + 1):
                ni = pos + delta
                if 0 <= ni < len(group_indices):
                    selected_indices.add(group_indices[ni])

    # Cap total to prevent prompt overflow
    # Sort with IAM-critical chunks first (highest priority), then by anomaly score descending.
    # This ensures domain-critical evidence (e.g., decryptValueAsBinary/WrapAEK) appears
    # at the top of the LLM's evidence window even if budget truncation occurs.
    def _evidence_sort_key(idx: int) -> tuple:
        doc = scored_docs[idx]
        is_critical = doc.metadata.get('iam_critical', False)
        score = doc.metadata.get('anomaly_score', 0.0)
        # Primary: IAM-critical first (True > False → negate for descending)
        # Secondary: anomaly score descending (negate for descending)
        return (not is_critical, -score)

    selected_list = sorted(selected_indices, key=_evidence_sort_key)[:MAP_MAX_CHUNKS]

    # Format evidence text with total budget enforcement
    parts: list[str] = []
    total_chars = 0
    for rank, idx in enumerate(selected_list):
        doc = scored_docs[idx]
        score = doc.metadata.get('anomaly_score', 0.0)
        pk = doc.metadata.get('primary_key', 'unknown')
        src = doc.metadata.get('source_file', 'unknown')
        start = doc.metadata.get('start_time', '')
        end = doc.metadata.get('end_time', '')
        ref_id = f"REF_{src}_{pk}_{rank}"

        # Determine retrieval source tag
        is_targeted = idx in targeted_indices
        is_anomaly = idx < min(top_n, n)
        bucket = doc.metadata.get('targeted_bucket', '')
        if is_targeted and is_anomaly:
            source_tag = f"both:anomaly+{bucket}"
        elif is_targeted:
            source_tag = f"targeted:{bucket}"
        else:
            source_tag = "anomaly"

        header = (
            f"--- Chunk [score={score:.2f}] [{ref_id}] "
            f"[source={source_tag}] "
            f"[file={src}] [key={pk}] [{start} -> {end}] ---"
        )

        content = doc.page_content

        # Smart truncation: do NOT truncate error-bearing chunks
        has_error_content = any(
            kw in content for kw in
            ['Exception', 'Error', 'Failed', 'FATAL', 'CRITICAL',
             'SecurityException', 'SessionInvalid', 'Caused by:']
        )
        if has_error_content:
            # Preserve full content for error chunks
            if len(content) > ERROR_CHUNK_MAX_CHARS:
                content = content[:ERROR_CHUNK_MAX_CHARS] + "\n... [truncated] ..."
        else:
            # Truncate benign chunks to save token budget
            if len(content) > BENIGN_CHUNK_MAX_CHARS:
                content = content[:BENIGN_CHUNK_MAX_CHARS] + "\n... [truncated] ..."

        # Decode URL-encoded error messages for LLM readability
        content = decode_url_encoded_errors(content)

        chunk_text = f"{header}\n{content}"

        # Enforce total evidence budget
        if total_chars + len(chunk_text) > max_total_chars:
            print(f"  [Evidence] Budget cap reached at {total_chars:,} chars "
                  f"({len(parts)} chunks). Remaining chunks skipped.")
            break
        total_chars += len(chunk_text)
        parts.append(chunk_text)

    evidence_text = "\n\n".join(parts)

    anomaly_count = sum(1 for i in selected_list if i < min(top_n, n))
    targeted_count = sum(1 for i in selected_list if i in targeted_indices)
    neighbour_count = len(selected_list) - len(
        set(selected_list) & (set(range(min(top_n, n))) | targeted_indices)
    )
    print(f"  [Evidence] Selected {len(selected_list):,} chunks "
          f"({anomaly_count} anomaly, {targeted_count} targeted, "
          f"{neighbour_count} neighbours)")
    return evidence_text

# ============================================================================
# Stage 5 — Map Phase (Per-file structured LLM analysis)
# ============================================================================

def analyze_single_file(file_path: str) -> dict:
    """
    [MAP STEP] Analyse a single log file end-to-end:
      1. Detect structure -> hybrid chunk -> embed & score anomalies
      2. Select top anomalous evidence
      3. Send structured prompt to LLM for forensic analysis
      4. Return findings dict

    Args:
        file_path: Absolute path to the log file

    Returns:
        Dict with keys: 'file', 'findings' (str), 'chunk_count', 'high_anomaly_count'
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"\n{'='*60}")
    print(f"[MAP] Analysing: {file_name} ({format_file_size(file_size)})")
    print(f"{'='*60}")

    # ---- 2. Detect structure ----
    print("  [Structure] Sampling lines for schema detection...")
    sample_lines: list[str] = []
    try:
        for line in stream_file_lines(file_path):
            sample_lines.append(line)
            if len(sample_lines) >= 1200:
                break
    except Exception as e:
        print(f"  [Error] Cannot read file: {e}")
        return {"file": file_name, "findings": "", "chunk_count": 0, "high_anomaly_count": 0}

    if not sample_lines:
        print("  [Skip] Empty file.")
        return {"file": file_name, "findings": "", "chunk_count": 0, "high_anomaly_count": 0}

    schema = detect_log_structure(sample_lines)
    ts_detected = schema['timestamp_fmt'] != ''
    thread_detected = schema['thread_re'] is not None
    print(f"    Timestamp detected: {ts_detected} | Thread detected: {thread_detected}")
    if schema['session_keys']:
        print(f"    Session keys: {[k for _, k in schema['session_keys']]}")

    # ---- 3. Hybrid chunking ----
    try:
        docs = hybrid_chunk_log(file_path, schema)
    except Exception as e:
        print(f"  [Error] Chunking failed: {e}")
        return {"file": file_name, "findings": "", "chunk_count": 0, "high_anomaly_count": 0}

    if not docs:
        print("  [Skip] No chunks produced.")
        return {"file": file_name, "findings": "", "chunk_count": 0, "high_anomaly_count": 0}

    raw_chunk_count = len(docs)

    # ---- 3b. Pre-embedding compression (Stage 2.5; pre-embedding only) ----
    if raw_chunk_count >= LARGE_LOG_CHUNK_TRIGGER:
        docs = deduplicate_chunks_safe(docs, schema)

    if len(docs) >= VERY_LARGE_LOG_CHUNK_TRIGGER:
        docs = downselect_chunks_for_embedding(docs, MAX_EMBEDDING_CHUNKS_VERY_LARGE)

    chunk_count = len(docs)
    if chunk_count != raw_chunk_count:
        print(f"  [Pre-Embed] Chunks: {raw_chunk_count:,} -> {chunk_count:,}")

    for original_idx, doc in enumerate(docs):
        doc.metadata['original_doc_index'] = original_idx

    # ---- 4. Embed once (shared by anomaly scoring + targeted retrieval) ----
    all_embeddings: Optional[np.ndarray] = None
    try:
        all_embeddings = _embed_documents_batched(
            docs,
            batch_size=50,
            label="Embedding chunks",
        )
    except Exception as e:
        print(f"  [Error] Embedding failed: {e}")

    # ---- 4a. Anomaly scoring ----
    try:
        docs = score_anomalies(docs, precomputed_embeddings=all_embeddings)
    except Exception as e:
        print(f"  [Error] Anomaly scoring failed: {e}")
        # Continue with unscored docs — set default scores
        for d in docs:
            d.metadata.setdefault('anomaly_score', 0.0)
            d.metadata.setdefault('raw_distance', 0.0)

    high_anomaly_count = sum(1 for d in docs if d.metadata.get('anomaly_score', 0) > ANOMALY_HIGH_THRESHOLD)

    # ---- 4b. Targeted retrieval (diversity bucket search) ----
    try:
        targeted_indices = retrieve_targeted_chunks(
            docs,
            file_name=file_name,
            precomputed_embeddings=all_embeddings,
        )
    except Exception as e:
        print(f"  [Warning] Targeted retrieval failed: {e}")
        targeted_indices = set()

    # Shared embeddings are no longer needed after stage 4b
    if all_embeddings is not None:
        del all_embeddings
        gc.collect()

    # ---- 5. Prioritised evidence selection (hybrid: anomaly + targeted) ----
    evidence_text = select_evidence_chunks(
        docs,
        top_n=MAP_TOP_N_CHUNKS,
        neighbour_radius=MAP_NEIGHBOUR_RADIUS,
        targeted_indices=targeted_indices,
        max_total_chars=MAP_EVIDENCE_BUDGET_CHARS,
    )

    if not evidence_text.strip():
        print("  [Skip] No evidence to analyse.")
        return {"file": file_name, "findings": "", "chunk_count": chunk_count,
                "high_anomaly_count": high_anomaly_count}

    # Free chunk memory before LLM call
    del docs
    gc.collect()

    # ---- 6. Structured LLM analysis ----
    print("  [LLM] Running forensic analysis...")

    system_prompt = """You are a seasoned IAM forensic investigator analysing server logs.
You receive log chunks retrieved by two methods:
  1. Anomaly scoring (statistical outliers from normal baseline)
  2. Targeted domain search (security, auth, crypto, connectivity queries)

Your primary task is ROOT CAUSE IDENTIFICATION — determine the single most likely
cause of the incident by focusing on ERROR and EXCEPTION lines.

STRICT RULES:
- ONLY cite errors, exceptions, or patterns that appear verbatim in the provided evidence.
- NEVER invent log messages, root causes, or user actions not supported by evidence.
- When quoting evidence, include the [REF_...] citation ID.
- Pay special attention to ERROR and EXCEPTION lines — these typically contain
  diagnostic messages that reveal the root cause.
- If an error message contains configuration instructions (e.g., property names,
  file paths, class names), quote them IN FULL — they are the resolution clue.
- If the evidence is insufficient to determine a root cause, say so explicitly.

OUTPUT FORMAT (Markdown):
## File Summary
- File name, time range, chunk count, high-anomaly count

## Critical Errors Identified
For each distinct error/exception:
  - Full error message (decoded, with [REF_...] ID)
  - Thread / session key
  - Time of occurrence
  - Category: Security / Crypto / Auth / Network / Performance / System
  - Diagnostic detail: any property names, file paths, or config hints in the message

## Root Cause Assessment
- The single most likely root cause (based ONLY on evidence)
- Supporting evidence chain
- Confidence level (High / Medium / Low)
"""

    user_prompt = f"""Analyse the following evidence from file **{file_name}**
({chunk_count} total chunks, {high_anomaly_count} high-anomaly).

EVIDENCE:
{evidence_text}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Debug output
    evidence_preview = evidence_text[:500] + "..." if len(evidence_text) > 500 else evidence_text
    print(f"  [DEBUG] Sending {len(evidence_text):,} chars to LLM")
    print(f"  [DEBUG] Evidence preview: {evidence_preview[:200]}...")
    
    # Save evidence to debug file
    debug_file = f"debug_evidence_{file_name}.txt"
    try:
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
            f.write(f"=== USER PROMPT ===\n{user_prompt}\n")
        print(f"  [DEBUG] Evidence saved to {debug_file}")
    except Exception as debug_err:
        print(f"  [WARNING] Could not save debug file: {debug_err}")

    try:
        response = llm.invoke(messages)
        findings = response.content
        print(f"  [DEBUG] LLM returned {len(findings):,} chars")
    except Exception as e:
        error_msg = str(e)
        print(f"  [ERROR] LLM call failed: {error_msg}")
        print(f"  [ERROR] This is an AGENT ERROR, not a log analysis result!")
        print(f"  [ERROR] Check {debug_file} to see what was sent to the LLM")
        # Return empty findings to skip this file rather than polluting the report
        return {
            "file": file_name,
            "findings": "",  # Empty so it won't be included in final report
            "chunk_count": chunk_count,
            "high_anomaly_count": high_anomaly_count,
        }

    print(f"  [MAP] Complete for {file_name}.")
    return {
        "file": file_name,
        "findings": findings,
        "chunk_count": chunk_count,
        "high_anomaly_count": high_anomaly_count,
    }

# ============================================================================
# Stage 6 — Reduce Phase (Final consolidation)
# ============================================================================

def consolidate_reports(all_findings: list[dict]) -> str:
    """
    [REDUCE STEP] Synthesise per-file findings into a single forensic report.

    Args:
        all_findings: List of dicts from analyze_single_file

    Returns:
        Final forensic report as Markdown string
    """
    print(f"\n{'='*60}")
    print(f"[REDUCE] Consolidating findings from {len(all_findings)} file(s)...")
    print(f"{'='*60}")

    # Compile evidence from all files
    compiled_evidence = ""
    contributing_files = 0
    failed_files = 0

    for report in all_findings:
        findings_text = report.get("findings", "")
        if not findings_text.strip():
            failed_files += 1
            print(f"  [WARNING] Skipping {report['file']} - no findings (likely LLM error)")
            continue

        file_name = report['file']
        chunk_count = report.get('chunk_count', 0)
        high_count = report.get('high_anomaly_count', 0)

        # Smart truncation: cap per-file evidence to stay within context window
        if len(findings_text) > REDUCE_PER_FILE_CAP_CHARS:
            findings_text = findings_text[:REDUCE_PER_FILE_CAP_CHARS] + "\n... [Findings Truncated for Conciseness] ..."

        compiled_evidence += (
            f"\n\n{'='*50}\n"
            f"=== FILE: {file_name} | Chunks: {chunk_count} | High-anomaly: {high_count} ===\n"
            f"{'='*50}\n"
            f"{findings_text}\n"
        )
        contributing_files += 1

        # Enforce total reduce evidence budget
        if len(compiled_evidence) > REDUCE_EVIDENCE_BUDGET_CHARS:
            print(f"  [WARNING] Reduce evidence budget reached at {len(compiled_evidence):,} chars. "
                  f"Remaining files will be summarised more aggressively.")
            compiled_evidence = compiled_evidence[:REDUCE_EVIDENCE_BUDGET_CHARS] + (
                "\n... [Evidence Truncated — Budget Limit Reached] ..."
            )
            break

    if not compiled_evidence.strip():
        if failed_files > 0:
            return (
                f"# ANALYSIS FAILED\n\n"
                f"All {failed_files} file(s) failed to analyze due to LLM errors.\n"
                f"This is typically caused by:\n"
                f"- AWS Bedrock timeout (increase timeout or reduce evidence size)\n"
                f"- Network connectivity issues\n"
                f"- Rate limiting on the LLM API\n\n"
                f"Try reducing top_n parameter in select_evidence_chunks() or check AWS credentials."
            )
        return "No critical anomalies detected in any files."

    print(f"  Compiled evidence from {contributing_files} file(s) ({failed_files} failed).")

    # ---- Final LLM consolidation ----
    system_text = """You are a Lead Forensic Investigator producing the final incident report.
You have received per-file analysis reports from your forensic data scientists.
Each report contains evidence retrieved via anomaly scoring and targeted domain search.

STRICT RULES:
- Correlate findings across files: look for matching timestamps, threads, or error chains.
- Prioritise evidence that contains specific error messages with diagnostic details
  (property names, file paths, exception types, configuration hints) over generic
  anomaly indicators or routine log entries.
- ONLY state root causes supported by quoted evidence with [REF_...] IDs.
- State your best assessment based on available evidence — prefer a specific conclusion
  with stated confidence over a vague hedge.
- NEVER invent scenarios, user actions, or system behaviours not in the evidence.

# REPORT STRUCTURE:

## 1. Executive Summary
(2-3 sentences: What is the root cause? How severe? How widespread?)

## 2. Root Cause Analysis
(Identify the single most likely root cause. Correlate clues across files.
Quote the key diagnostic error message in full. Be specific and actionable.)

## 3. Verified Evidence
(Quote specific error messages with [REF_...] IDs. Group by category.
Include any configuration properties, file paths, or class names mentioned in errors.)

## 4. Recommendations
(Actionable technical fixes ranked by priority.
Include specific config changes, property names, or file paths if mentioned in evidence.)
"""

    user_prompt = f"""Here are the compiled per-file forensic analyses:
{compiled_evidence}

Generate the Final Forensic Incident Report.
"""

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_prompt),
    ]

    try:
        print(f"  [DEBUG] Sending {len(compiled_evidence):,} chars to final LLM")
        response = llm.invoke(messages)
        print(f"  [DEBUG] Final LLM returned {len(response.content):,} chars")
        return response.content
    except Exception as e:
        error_msg = str(e)
        print(f"  [ERROR] Final LLM call failed: {error_msg}")
        # Return a proper error report instead of trying to format the error as findings
        return (
            f"# AGENT ERROR - Report Generation Failed\n\n"
            f"The final report could not be generated due to an LLM timeout or error:\n"
            f"`{error_msg}`\n\n"
            f"This is an infrastructure issue with the AI service, not an analysis result.\n\n"
            f"## Partial Evidence Summary\n"
            f"The following files were analyzed but could not be consolidated:\n"
            f"{compiled_evidence[:BENIGN_CHUNK_MAX_CHARS]}"
        )

# ============================================================================
# PDF Export
# ============================================================================

def export_to_pdf(report_text: str, filename: str = "IAM_Forensic_Report.pdf") -> Optional[str]:
    """
    Export the final text report to a professional PDF file.

    Args:
        report_text: Markdown-formatted report string
        filename:    Output PDF filename

    Returns:
        Absolute path to the generated PDF, or None on failure
    """
    print(f"\n[Export] Generating PDF report: {filename}...")

    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story: list = []

        # Custom styles
        title_style = styles['Title']
        heading_style = styles['Heading2']
        body_style = styles['BodyText']
        code_style = ParagraphStyle(
            'Code',
            parent=styles['BodyText'],
            fontName='Courier',
            fontSize=8,
            leading=10,
            backColor=colors.lightgrey,
            borderPadding=5,
        )

        lines = report_text.split('\n')

        story.append(Paragraph("IAM Forensic Investigation Report", title_style))
        story.append(Spacer(1, 12))

        buffer_text: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                if buffer_text:
                    story.append(Paragraph(" ".join(buffer_text), body_style))
                    buffer_text = []
                    story.append(Spacer(1, 6))

                clean_header = line.replace('#', '').strip()
                story.append(Paragraph(clean_header, heading_style))
                story.append(Spacer(1, 6))

            elif line.startswith('*') or line.startswith('-') or '[REF_' in line:
                if buffer_text:
                    story.append(Paragraph(" ".join(buffer_text), body_style))
                    buffer_text = []
                story.append(Paragraph(line, code_style))
                story.append(Spacer(1, 4))

            else:
                buffer_text.append(line)

        if buffer_text:
            story.append(Paragraph(" ".join(buffer_text), body_style))

        doc.build(story)
        abs_path = os.path.abspath(filename)
        print(f"[Export] PDF saved successfully to: {abs_path}")
        return abs_path

    except Exception as e:
        print(f"[Export] Failed to generate PDF: {e}")
        return None

# ============================================================================
# Pipeline Execution
# ============================================================================

def run_pipeline(path_input: str) -> str:
    """
    Run the full Map-Reduce pipeline:
      1. Discover log files
      2. For each file: chunk -> score -> select evidence -> LLM map analysis
      3. Consolidate all findings into final report
      4. Export PDF

    Args:
        path_input: Path to a single log file or directory of log files

    Returns:
        Final forensic report as string
    """
    # 1. Discovery
    log_files = get_log_files_from_path(path_input)
    if not log_files:
        return "Error: No log files found."

    print(f"\n[Pipeline] Processing {len(log_files)} file(s)...\n")

    # 2. Map — analyse each file independently
    all_findings: list[dict] = []
    for i, f in enumerate(log_files):
        print(f"\n[Pipeline] File {i+1}/{len(log_files)}")
        try:
            report = analyze_single_file(f)
            all_findings.append(report)
        except Exception as e:
            print(f"[Pipeline] Error processing {f}: {e}")
            all_findings.append({
                "file": os.path.basename(f),
                "findings": "",
                "chunk_count": 0,
                "high_anomaly_count": 0,
            })

    # 3. Reduce — consolidate
    final_report = consolidate_reports(all_findings)

    # 4. Export to PDF
    export_to_pdf(final_report)

    return final_report

# ============================================================================
# Interactive Mode (Kept for ad-hoc usage)
# ============================================================================

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

            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            # Check for analyze command
            if user_input.lower().startswith('analyze '):
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
                SystemMessage(content=(
                    "You are an expert IAM Log Intelligence Agent. "
                    "Answer the user's question concisely. If they want to analyse "
                    "logs, tell them to use: analyze <path>"
                )),
                HumanMessage(content=user_input),
            ]
            response = llm.invoke(messages)
            print(f"\nAgent: {response.content}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IAM Log Intelligence Agent")
    print("=" * 60 + "\n")

    if len(sys.argv) > 1:
        # User provided one or more paths (files or folders)
        input_paths = sys.argv[1:]
        
        # Collect all log files from all provided paths
        all_log_files: list[str] = []
        for path_input in input_paths:
            if os.path.exists(path_input):
                files = get_log_files_from_path(path_input)
                all_log_files.extend(files)
            else:
                print(f"Warning: Path not found (skipping): {path_input}")
        
        if not all_log_files:
            print("Error: No valid log files found in the provided paths.")
            sys.exit(1)
        
        print(f"\n[Pipeline] Processing {len(all_log_files)} file(s)...\n")
        
        # Map — analyse each file independently
        all_findings: list[dict] = []
        for i, f in enumerate(all_log_files):
            print(f"\n[Pipeline] File {i+1}/{len(all_log_files)}")
            try:
                result = analyze_single_file(f)
                all_findings.append(result)
            except Exception as e:
                print(f"[X] Failed to analyse {os.path.basename(f)}: {e}")
                import traceback
                traceback.print_exc()
                all_findings.append({
                    "file": os.path.basename(f),
                    "findings": "",
                    "chunk_count": 0,
                    "high_anomaly_count": 0,
                })
        
        # Reduce — consolidate
        final_report = consolidate_reports(all_findings)
        
        # Export to PDF
        export_to_pdf(final_report)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(final_report)

    else:
        # Interactive mode
        print("No log file/folder provided. Starting interactive mode...")
        interactive_mode()
