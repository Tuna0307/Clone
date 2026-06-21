"""Pipeline constants and default search / retrieval signals.

Extracted from iam_log_intelligence_agent_hybridChunking2.py as part of
a conservative modular refactor.
"""

import re

__all__ = [
    "ANOMALY_HIGH_THRESHOLD",
    "BENIGN_CHUNK_MAX_CHARS",
    "ERROR_CHUNK_MAX_CHARS",
    "ERROR_SCORE_BOOST",
    "IAM_CRITICAL_SCORE_BOOST",
    "MAP_EVIDENCE_BUDGET_CHARS",
    "MAP_MAX_CHUNKS",
    "MAP_NEIGHBOUR_RADIUS",
    "MAP_TOP_N_CHUNKS",
    "MAX_LOG_FILE_SIZE_BYTES",
    "REDUCE_EVIDENCE_BUDGET_CHARS",
    "REDUCE_PER_FILE_CAP_CHARS",
    # DuckDB / server_monitoring agentic SQL (new path)
    "SERVER_SQL_MAX_STEPS",
    "SERVER_SQL_MAX_RECLASSIFICATIONS",
    "SERVER_SQL_RESULT_TRUNCATE",
    "SERVER_SQL_FOLLOWUP_MAX_STEPS",
    "EVIDENCE_GATHERING_MAX_TURNS",
    "EVIDENCE_GATHERING_MAX_QUERIES_PER_TURN",
    "EVIDENCE_GATHERING_MAX_CRITIC_RETRY_LOOPS",
    "SERVER_MONITORING_DB_DIR",
    "LOG_GAP_THRESHOLD_SECONDS",
    "STRUCTURAL_SIGNAL_MIN_STRENGTH",
    # Log events table + pre-scan for application outliers (server_monitoring path)
    "SERVER_LOG_EVENTS_TABLE",
    "HIGH_SIGNAL_PATTERNS",
    "MAX_PRE_SCAN_CANDIDATES",
    # Ticket context (server_monitoring post-report refinement only)
    "TICKET_CONTEXT_MAX_CHARS",
    "TICKET_REFINEMENT_EXTRA_STEPS",
    "_DEFAULT_API_KNOWN_ERROR_KEYWORDS",
    "_DEFAULT_API_REQUEST_BOUNDARIES",
    "_DEFAULT_ERROR_KEYWORDS",
    "_DEFAULT_IAM_CRITICAL_KEYWORDS",
    "_DEFAULT_NOISE_PATTERNS",
    "_DEDUP_UUID_RE",
    "_DEDUP_WS_RE",
    "_QUERY_DATE_ONLY_FORMATS",
    "_QUERY_DATETIME_FORMATS",
    "_STACK_TRACE_LINE_RE",
]

# -- Map phase (per-file analysis) --
MAP_EVIDENCE_BUDGET_CHARS: int = 800_000   # Hard cap on evidence chars sent to map LLM
MAP_TOP_N_CHUNKS: int = 180                 # Top anomaly-scored chunks to select
MAP_MAX_CHUNKS: int = 450                  # Max total chunks (ranked seeds + neighbours)
MAP_NEIGHBOUR_RADIUS: int = 2              # Temporal neighbours per selected chunk
MAX_LOG_FILE_SIZE_BYTES: int = 5 * 1024 * 1024 * 1024  # 5GB limit per file

# -- DuckDB server_monitoring agentic SQL path (new opt-in mode) --
SERVER_SQL_MAX_STEPS: int = 30              # Hard cap on iterative SQL queries / LLM turns before forcing final report.
                                              # Covers archetype classification, onset analysis, critic loops, and optional ticket refinement.

# (USE_STRUCTURED_SERVER_WORKFLOW flag removed in Phase 4 — the structured LangGraph workflow is now the only server_monitoring path.)
SERVER_SQL_MAX_RECLASSIFICATIONS: int = 1   # Max critic-driven archetype reclassification loop-backs
SERVER_SQL_RESULT_TRUNCATE: int = 3000      # Max chars per SQL observation fed back to LLM
SERVER_SQL_FOLLOWUP_MAX_STEPS: int = 15     # Max agentic SQL turns for server_monitoring follow-up chat
EVIDENCE_GATHERING_MAX_TURNS: int = 3       # LLM turns per evidence_gathering visit (observe → refine loop)
EVIDENCE_GATHERING_MAX_QUERIES_PER_TURN: int = 3  # Max fenced SQL blocks executed per evidence turn
EVIDENCE_GATHERING_MAX_CRITIC_RETRY_LOOPS: int = 1  # Times critic RETRY may loop back to evidence_gathering
LOG_GAP_THRESHOLD_SECONDS: int = 120        # Minimum inter-log gap to flag as a structural signal
STRUCTURAL_SIGNAL_MIN_STRENGTH: float = 0.3 # Minimum strength to include in classification prompts
SERVER_MONITORING_DB_DIR: str = "outputs/faiss"  # Base dir for optional persisted .duckdb artifacts (future)

# -- Ticket / incident context support (server_monitoring post-report refinement only) --
TICKET_CONTEXT_MAX_CHARS: int = 12000       # Hard cap on ticket text injected into refinement prompt
TICKET_REFINEMENT_EXTRA_STEPS: int = 3      # Additional LLM turns allowed for the ticket-guided iteration pass

# -- DuckDB server_monitoring: full log events table + deterministic outlier pre-scan --
# These enable the agent to discover application-level root causes (high row counts,
# N+1 authz loops, extreme latencies) that live in plain log lines rather than
# the periodic numeric "Server statistics={...}" metric snapshots.
SERVER_LOG_EVENTS_TABLE: str = "log_events"
MAX_PRE_SCAN_CANDIDATES: int = 80  # Hard cap on pre-detected high-signal events injected into seed facts

# General, non-incident-specific regex patterns for pre-detecting application outliers
# that commonly cause CPU/throughput saturation when infrastructure metrics look "normal".
# Each entry: (signal_type, compiled_regex_with_optional_capture_for_the_big_number).
# Used by pre_detect_high_signal_events() in server_metrics.py. Keep small & high-precision.
HIGH_SIGNAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("high_result_count", re.compile(r"(?i)\b(Count|rows?|returned|result\s*(count|size)?)\s*[:=]?\s*(\d{3,})")),
    ("extreme_latency", re.compile(r"lapse\(ms\)\s*=\s*(\d{4,})")),
    ("authz_loop_candidate", re.compile(r"(?i)(RoleValidator|CheckCredentialRole|getRelationsByObj|getCredentialTO).*?(entry| - entry)")),
    ("heavy_repository_op", re.compile(r"(?i)UCMRepository.*?(Request|Credential|Request\.java)")),
    ("large_cache_event", re.compile(r"(?i)(Eviction|cache).*?(size|count)\s*=\s*(\d{3,})")),
    ("slow_ldap_or_db", re.compile(r"(?i)(findByFilter|LDAP|jdbc|SQL).*?(lapse|time|ms)\s*[:=]?\s*(\d{4,})")),
]

# Fast reject regex for the signal pre-scan: covers the dominant keywords
# across all HIGH_SIGNAL_PATTERNS. Skips the expensive 6-pattern loop on
# ~95% of benign log lines. Not a perfect superset — very rare false
# negatives (e.g. bare "result size = 500" without count/row/returned) are
# acceptable because the full patterns still catch them on the remaining 5%.
_SIGNAL_QUICK_REJECT_RE = re.compile(
    r"(?i)\b(count|rows?|returned|lapse|rolevalidator|checkcredentialrole|"
    r"getrelationsbyobj|getcredential|ucmrepository|eviction|cache|"
    r"findbyfilter|ldap|jdbc|sql)"
)

# -- Chunk truncation --
ERROR_CHUNK_MAX_CHARS: int = 8_000         # Max chars for error-bearing chunks (preserve diagnostic detail)
BENIGN_CHUNK_MAX_CHARS: int = 2_000        # Max chars for benign/routine chunks

# -- Reduce phase (cross-file consolidation) --
REDUCE_EVIDENCE_BUDGET_CHARS: int = 600_000  # Hard cap on total compiled evidence
REDUCE_PER_FILE_CAP_CHARS: int = 60_000       # Max chars per file's findings in reduce

# -- Deterministic API scoring / follow-up ranking --
ANOMALY_HIGH_THRESHOLD: float = 2.5       # Score threshold for high-signal evidence
ERROR_SCORE_BOOST: float = 2.0            # Score boost for error-bearing chunks
IAM_CRITICAL_SCORE_BOOST: float = 4.0     # Extra boost for IAM-domain-critical chunks (stacks on ERROR_SCORE_BOOST)

# LLM_MAX_TOKENS is loaded in config.py and consumed by llm_factory.py.

# ============================================================================
# Search Configuration (Retrieval Signals)
# ============================================================================

_DEFAULT_IAM_CRITICAL_KEYWORDS: list[str] = [
    'decryptValueAsBinary',
    'keyId=WrapAEK',
    'WrapAEK',
    'CryptoService',
    'GenericTokenService',
    'TokenException',
    'AuthenticationException',
    'VerificationFailed',
    'SessionReplaced',
    'SessionInvalid',
    'sesToken',
    'simCert',
    'amsystem.properties',
    'VascoToken',
    'TokenNotActive',
    'HSM',
    'PKCS11',
]

# Keywords that indicate error-bearing content.
_DEFAULT_ERROR_KEYWORDS: list[str] = [
    'ERROR', 'Exception', 'FATAL', 'Failed', 'Refused', 'CRITICAL',
    'SecurityException', 'SessionInvalid', 'VerificationFailed',
    'NullPointerException', 'Caused by:', 'stack trace',
]

# Patterns for benign/noisy log lines.
_DEFAULT_NOISE_PATTERNS: list[str] = [
    r'Audit took \d+',
    r'refreshSession.*success',
    r'^\s*INFO\s.*\bstarted\b',
    r'^\s*INFO\s.*\bhealthy\b',
    r'ERRORCODE=.*SQLSTATE=',
    r'Connection\s*(reset|refused)',
    r'ConnectException',
    r'Error opening socket to server',
    r'Start Display Current Environment',
    r'End Display Current Environment',
    r'log4j:WARN',
    r'\[Fatal Error\].*Premature end of file',
    r'\[Fatal Error\].*Element type',
    r'hql:\s*select\b',
]

_DEFAULT_API_KNOWN_ERROR_KEYWORDS: list[str] = [
    'tokenexception',
    'authenticationexception',
    'verificationfailed',
    'sessioninvalid',
    'nullpointerexception',
    'securityexception',
    'hsm',
    'pkcs11',
    'wrapaek',
    'errorcode',
    'http 401',
    'http 403',
    'http 500',
]

_DEFAULT_API_REQUEST_BOUNDARIES: dict[str, list[str]] = {
    'start_markers': [' - entry', ':entry', ' entry,'],
    'end_markers': [' - exit', ':exit', ' exit,', 'lapse(ms)='],
}

_DEDUP_UUID_RE = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
)
_DEDUP_WS_RE = re.compile(r'\s+')
_STACK_TRACE_LINE_RE = re.compile(r'^\s*(?:at\s+|\.\.\.\s+\d+\s+more\b)')

_QUERY_DATETIME_FORMATS: list[str] = [
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M',
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
]

_QUERY_DATE_ONLY_FORMATS: list[str] = [
    '%Y-%m-%d',
    '%d/%m/%Y',
]
