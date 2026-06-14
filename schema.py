"""
Schema inference and logical-entry grouping helpers.

The active pipeline first tries fast regex-based schema detection. This module
contains deeper helpers for grouping multiline log entries and for assisting
low-confidence schema cases without changing the main Map-Reduce flow.
"""

# ============================================================================
# Imports
# ============================================================================

# Standard library
import os
import sys
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Iterator, Optional, TextIO
from collections import defaultdict

# Third-party
import numpy as np
from tqdm import tqdm

from llm_factory import get_llm

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# LangChain ecosystem
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional, Dict, Any

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import statistics
from datetime import datetime, timedelta


# ============================================================================
# LLM Configuration (provider-agnostic via llm_factory)
# ============================================================================

llm = get_llm()

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

def group_log_entries(
    file_handle: TextIO, 
    schema: dict, 
    max_lines_per_entry: int = 100
) -> Iterator[List[str]]:
    """
    Groups physical lines into logical log entries based on schema.
    
    Logic:
    1. If line matches stack_trace_re -> Continuation.
    2. If line matches timestamp_re at start -> New Entry.
    3. Otherwise -> Continuation (wrapped line).
    """
    buffer: List[str] = []
    timestamp_re = schema.get('timestamp_re')
    stack_trace_re = schema.get('stack_trace_re')
    
    # Pre-compile a "Start of Record" check (timestamp at beginning of line)
    # We strip the line prefix first to check accurately
    line_prefix_re = re.compile(r'^\[Line \d+\]\s*')
    
    for line in file_handle:
        clean_line = line.rstrip('\n')
        stripped_line = line_prefix_re.sub('', clean_line)
        
        is_continuation = False
        
        # 1. Check explicit stack trace pattern
        if stack_trace_re and stack_trace_re.match(stripped_line):
            is_continuation = True
            
        # 2. Check if this line starts a NEW record (has timestamp at start)
        elif timestamp_re:
            # Check if timestamp is at the beginning (allowing for optional prefix)
            match = timestamp_re.search(stripped_line)
            if match and match.start() == 0:
                is_continuation = False
            else:
                # No timestamp at start -> likely continuation/wrap
                is_continuation = True
        else:
            # No timestamp schema -> treat every line as new entry (safe fallback)
            is_continuation = False
        
        # Handle Buffering
        if is_continuation and buffer:
            buffer.append(clean_line)
            # Safety valve: prevent infinite buffering on broken logs
            if len(buffer) >= max_lines_per_entry:
                yield buffer
                buffer = []
        else:
            # New entry detected
            if buffer:
                yield buffer
            buffer = [clean_line]
    
    # Yield remaining
    if buffer:
        yield buffer

def _parse_entry(entry_lines: List[str], schema: dict) -> tuple[Optional[datetime], Optional[str], str]:
    """
    Parse a logical log entry (potentially multiline).
    
    Args:
        entry_lines: List of physical lines making up one log entry
        schema: Detected schema
    
    Returns:
        (timestamp, primary_key, full_entry_text)
    """
    if not entry_lines:
        return None, None, ""
        
    # Metadata usually lives on the FIRST line
    first_line = entry_lines[0]
    clean_first = _LINE_PREFIX_RE.sub('', first_line).rstrip('\n')
    
    # --- Timestamp (from first line) ---
    ts: Optional[datetime] = None
    ts_match = schema['timestamp_re'].search(clean_first) if schema.get('timestamp_re') else None
    if ts_match and schema.get('timestamp_fmt'):
        try:
            ts_str = ts_match.group(1)
            if '.%f' in schema['timestamp_fmt'] and len(ts_str) > 4 and ts_str[-4:-3] == ':':
                ts_str = ts_str[:-4] + '.' + ts_str[-3:]
            ts = datetime.strptime(ts_str, schema['timestamp_fmt'])
        except (ValueError, IndexError):
            pass

    # --- Primary Key (from first line) ---
    pk: Optional[str] = None
    if schema.get('thread_re'):
        m = schema['thread_re'].search(clean_first)
        if m:
            pk = m.group(1)
    
    if pk is None and schema.get('session_keys'):
        for compiled, key_name in schema['session_keys']:
            m = compiled.search(clean_first)
            if m:
                pk = f"{key_name}:{m.group(1)}"
                break

    # --- Full Text (join all lines) ---
    # This preserves stack traces or wrapped context
    full_text = "\n".join(entry_lines)
    
    return ts, pk, full_text

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

        'is_multiline': False,  # New flag
         'continuation_re': re.compile(r'^\s+'),  # Default: lines starting with whitespace
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

def _generate_schema_llm(sample_lines: list[str]) -> Optional[Dict[str, Any]]:
    """
    Use LLM to infer log schema when regex heuristics fail.
    Returns a schema dict compatible with _parse_line, or None if inference fails.
    """
    # Prepare sample (limit to avoid token limits)
    sample_text = "\n".join(sample_lines[:50])
    
    system_prompt = """You are a log parsing expert. Analyze the sample log lines and extract:
1. TIMESTAMP: The regex pattern to capture the timestamp and the strptime format string.
2. THREAD: The regex pattern to capture thread IDs (if present).
3. SESSION_KEYS: List of key-value patterns for transaction/session IDs.

Return ONLY valid JSON with this structure:
{
    "timestamp_regex": "regex pattern with ONE capturing group for the timestamp",
    "timestamp_format": "strptime format string (e.g., '%Y-%m-%d %H:%M:%S.%f')",
    "thread_regex": "regex pattern with ONE capturing group for thread ID (or null)",
    "session_keys": [
        {"regex": "pattern with ONE capturing group for the value", "name": "key_name"}
    ]
}

Rules:
- Use Python regex syntax.
- Ensure all regexes have exactly ONE capturing group () for the value.
- For timestamps with milliseconds separated by colon (e.g. 10:20:30:456), 
  set format as '%H:%M:%S.%f' (the parser will normalize the colon to dot).
- If no pattern found for a field, use null.
- Do not include markdown code blocks or explanations. Return raw JSON only.
- Focus on precision: it's better to miss a few matches than to return an overly broad regex that captures non-timestamp/thread data.
- The sample log lines may contain stack traces or wrapped lines; focus on the main log entry format."""

    human_prompt = f"Analyze these log lines and return the schema JSON:\n\n{sample_text}"
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        content = response.content.strip()
        
        # Clean up potential markdown artifacts
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        schema = json.loads(content.strip())
        
        # Validate and compile regexes
        compiled_schema = {
            'timestamp_re': None,
            'timestamp_fmt': schema.get('timestamp_format'),
            'thread_re': None,
            'session_keys': [],
            'stack_trace_re': re.compile(r'^(?:\s+at |\s*Caused by:)')
        }
        
        if schema.get('timestamp_regex'):
            compiled_schema['timestamp_re'] = re.compile(f"({schema['timestamp_regex']})")
            
        if schema.get('thread_regex'):
            compiled_schema['thread_re'] = re.compile(schema['thread_regex'])
            
        for key in schema.get('session_keys', []):
            compiled_schema['session_keys'].append((
                re.compile(key['regex'], re.IGNORECASE),
                key['name']
            ))
            
        return compiled_schema
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM schema inference failed: {e}")
        return None
    
def detect_log_structure_hybrid(
    sample_lines: list[str], 
    use_llm_fallback: bool = True,
    enable_multiline: bool = True
) -> dict:
    """
    Detect log schema using regex heuristics first, LLM as fallback.
    Includes multiline/stack-trace detection and continuation pattern inference.
    
    Args:
        sample_lines: First 800-1200 lines of the log file
        use_llm_fallback: Whether to use LLM if regex confidence is low
        enable_multiline: Whether to detect and enable multiline entry grouping
    
    Returns:
        Schema dict with keys:
            timestamp_re   — compiled regex with one group for the timestamp string
            timestamp_fmt  — strptime format string
            thread_re      — compiled regex with one group for thread name (or None)
            session_keys   — list of (compiled_regex, key_name) tuples
            stack_trace_re — compiled regex to identify continuation / stack trace lines
            is_multiline   — bool indicating if logs span multiple lines
            continuation_re — regex to match continuation lines (if is_multiline=True)
    """
    # -------------------------------------------------------------------------
    # Step 1: Run fast regex detection
    # -------------------------------------------------------------------------
    schema = detect_log_structure(sample_lines)
    
    # Initialize multiline fields
    schema['is_multiline'] = False
    schema['continuation_re'] = None
    
    # -------------------------------------------------------------------------
    # Step 2: Calculate confidence score
    # -------------------------------------------------------------------------
    confidence = _calculate_detection_confidence(schema, sample_lines)
    
    # -------------------------------------------------------------------------
    # Step 3: Detect multiline properties (if enabled)
    # -------------------------------------------------------------------------
    if enable_multiline:
        _detect_multiline_properties(sample_lines, schema)
    
    # -------------------------------------------------------------------------
    # Step 4: Use LLM if confidence is low
    # -------------------------------------------------------------------------
    if use_llm_fallback and confidence < 0.5:
        print(f"Regex detection confidence low ({confidence:.2f}), trying LLM...")
        llm_schema = _generate_schema_llm(sample_lines)
        
        if llm_schema:
            # Merge: prefer LLM fields where regex failed
            if schema['timestamp_re'] is None and llm_schema.get('timestamp_re'):
                schema['timestamp_re'] = llm_schema['timestamp_re']
                schema['timestamp_fmt'] = llm_schema.get('timestamp_fmt')
                
            if schema['thread_re'] is None and llm_schema.get('thread_re'):
                schema['thread_re'] = llm_schema['thread_re']
                
            if not schema['session_keys'] and llm_schema.get('session_keys'):
                schema['session_keys'] = llm_schema['session_keys']
            
            # Merge multiline properties from LLM
            if enable_multiline:
                if not schema['is_multiline'] and llm_schema.get('is_multiline'):
                    schema['is_multiline'] = llm_schema['is_multiline']
                    schema['continuation_re'] = llm_schema.get('continuation_re')
                elif schema['is_multiline'] and llm_schema.get('continuation_re'):
                    # LLM may have a better continuation pattern
                    schema['continuation_re'] = llm_schema['continuation_re']
                
            print("LLM fallback successful!")
        else:
            print("LLM fallback failed, using regex schema")
    
    # -------------------------------------------------------------------------
    # Step 5: Validate multiline detection (sanity checks)
    # -------------------------------------------------------------------------
    if enable_multiline and schema['is_multiline']:
        _validate_multiline_detection(schema, sample_lines)
    
    return schema


def _detect_multiline_properties(sample_lines: list[str], schema: dict) -> None:
    """
    Analyze samples to determine if logs are multiline and set continuation patterns.
    Updates schema in-place.
    
    Heuristics:
    1. Stack trace lines (at, Caused by, etc.)
    2. Lines without timestamps following lines with timestamps
    3. Lines starting with whitespace
    4. JSON blocks spanning multiple lines
    """
    import re
    
    cleaned = [_LINE_PREFIX_RE.sub('', l).rstrip('\n') for l in sample_lines]
    
    # --- 1. Detect stack traces ---
    stack_trace_re = schema.get('stack_trace_re')
    stack_trace_count = 0
    if stack_trace_re:
        stack_trace_count = sum(1 for l in cleaned if stack_trace_re.match(l))
    
    # --- 2. Detect lines without timestamps ---
    timestamp_re = schema.get('timestamp_re')
    no_ts_indices = []
    
    for i, line in enumerate(cleaned):
        if not line.strip():
            continue
        if timestamp_re:
            match = timestamp_re.search(line)
            # Check if timestamp is at or near the start of the line
            if not match or match.start() > 10:  # Allow small prefix
                no_ts_indices.append(i)
        else:
            # No timestamp schema → can't detect multiline reliably
            return
    
    no_ts_count = len(no_ts_indices)
    ratio_no_ts = no_ts_count / len(cleaned) if cleaned else 0
    
    # --- 3. Detect lines starting with whitespace (common continuation pattern) ---
    whitespace_start_count = sum(1 for l in cleaned if l and l[0].isspace())
    ratio_whitespace = whitespace_start_count / len(cleaned) if cleaned else 0
    
    # --- 4. Heuristic decision ---
    # Multiline if: stack traces exist OR many lines lack timestamps OR many start with whitespace
    is_multiline = (
        stack_trace_count > 0 or
        ratio_no_ts > 0.15 or  # >15% of lines have no timestamp
        ratio_whitespace > 0.15  # >15% start with whitespace
    )
    
    if is_multiline:
        schema['is_multiline'] = True
        
        # Determine best continuation regex
        if stack_trace_count > 0:
            # Use existing stack_trace_re for Java-style traces
            schema['continuation_re'] = stack_trace_re
        elif ratio_whitespace > 0.1:
            # Whitespace-indented continuations
            schema['continuation_re'] = re.compile(r'^\s+')
        else:
            # Fallback: lines without timestamps at start are continuations
            # This is less precise but catches wrapped console output
            schema['continuation_re'] = re.compile(r'^(?!\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{2})')
        
        print(f"Multiline detected: stack_traces={stack_trace_count}, "
              f"no_ts_ratio={ratio_no_ts:.2f}, whitespace_ratio={ratio_whitespace:.2f}")
    else:
        schema['is_multiline'] = False
        schema['continuation_re'] = None


def _validate_multiline_detection(schema: dict, sample_lines: list[str]) -> None:
    """
    Sanity-check multiline detection to avoid false positives.
    Updates schema in-place if issues are found.
    """
    if not schema.get('is_multiline'):
        return
    
    cleaned = [_LINE_PREFIX_RE.sub('', l).rstrip('\n') for l in sample_lines]
    timestamp_re = schema.get('timestamp_re')
    continuation_re = schema.get('continuation_re')
    
    if not timestamp_re or not continuation_re:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print("⚠️  Multiline disabled: missing timestamp or continuation regex")
        return
    
    # Count how many lines would be classified as continuations
    continuation_count = sum(1 for l in cleaned if continuation_re.match(l))
    ratio_continuation = continuation_count / len(cleaned) if cleaned else 0
    
    # If >80% of lines are "continuations", detection is probably wrong
    if ratio_continuation > 0.8:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print(f"⚠️  Multiline disabled: too many continuation matches ({ratio_continuation:.1%})")
        return
    
    # If <1% are continuations, multiline may not be needed
    if ratio_continuation < 0.01:
        schema['is_multiline'] = False
        schema['continuation_re'] = None
        print(f"ℹ️  Multiline disabled: too few continuation matches ({ratio_continuation:.1%})")
        return
    
    print(f"✓ Multiline validation passed: {ratio_continuation:.1%} continuation lines")


def _calculate_detection_confidence(schema: dict, sample_lines: list[str]) -> float:
    """
    Calculate confidence score (0.0 - 1.0) for detected schema.
    Now accounts for multiline detection quality.
    """
    import re
    
    score = 0.0
    weights = {'timestamp': 0.5, 'thread': 0.3, 'session': 0.2}
    
    cleaned = [_LINE_PREFIX_RE.sub('', l) for l in sample_lines[:200]]
    if not cleaned:
        return 0.0
    
    # Timestamp confidence
    if schema.get('timestamp_re'):
        hits = sum(1 for l in cleaned if schema['timestamp_re'].search(l))
        score += weights['timestamp'] * (hits / len(cleaned))
    
    # Thread confidence
    if schema.get('thread_re'):
        hits = sum(1 for l in cleaned if schema['thread_re'].search(l))
        score += weights['thread'] * (hits / len(cleaned))
    
    # Session confidence (presence only)
    if schema.get('session_keys'):
        for compiled, _ in schema['session_keys']:
            hits = sum(1 for l in cleaned if compiled.search(l))
            if hits > 0:
                score += weights['session']
                break
    
    # Multiline bonus/penalty
    if schema.get('is_multiline'):
        continuation_re = schema.get('continuation_re')
        if continuation_re:
            cont_hits = sum(1 for l in cleaned if continuation_re.match(l))
            cont_ratio = cont_hits / len(cleaned)
            
            # Ideal continuation ratio is 5-40%
            if 0.05 <= cont_ratio <= 0.4:
                score += 0.05  # Small bonus for good multiline detection
            elif cont_ratio > 0.6:
                score -= 0.1   # Penalty for over-matching continuations
    
    return max(0.0, min(1.0, score))

def _calculate_detection_confidence(schema: dict, sample_lines: list[str]) -> float:
    """
    Calculate confidence score (0.0 - 1.0) for detected schema.
    """
    score = 0.0
    weights = {'timestamp': 0.5, 'thread': 0.3, 'session': 0.2}
    
    cleaned = [_LINE_PREFIX_RE.sub('', l) for l in sample_lines[:200]]
    if not cleaned:
        return 0.0
    
    # Timestamp confidence
    if schema['timestamp_re']:
        hits = sum(1 for l in cleaned if schema['timestamp_re'].search(l))
        score += weights['timestamp'] * (hits / len(cleaned))
    
    # Thread confidence
    if schema['thread_re']:
        hits = sum(1 for l in cleaned if schema['thread_re'].search(l))
        score += weights['thread'] * (hits / len(cleaned))
    
    # Session confidence (presence only)
    if schema['session_keys']:
        for compiled, _ in schema['session_keys']:
            hits = sum(1 for l in cleaned if compiled.search(l))
            if hits > 0:
                score += weights['session']
                break
    
    return score


from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import statistics
from datetime import datetime, timedelta
import re

@dataclass
class ValidationResult:
    """Results from schema validation tests."""
    test_name: str
    passed: bool
    score: float  # 0.0 - 1.0
    details: str
    warnings: List[str] = field(default_factory=list)

@dataclass
class SchemaReliabilityReport:
    """Aggregate report of all validation tests."""
    overall_score: float
    is_reliable: bool
    results: List[ValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        self.results.append(result)
        if not result.passed:
            self.recommendations.append(f"⚠️  {result.details}")
    
    def __str__(self) -> str:
        status = "✅ RELIABLE" if self.is_reliable else "❌ UNRELIABLE"
        lines = [
            f"Schema Reliability: {status} (Score: {self.overall_score:.2f})",
            "=" * 60,
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"{icon} {r.test_name}: {r.score:.2f} - {r.details}")
        if self.recommendations:
            lines.append("\nRecommendations:")
            lines.extend(f"  • {rec}" for rec in self.recommendations)
        return "\n".join(lines)


class SchemaValidator:
    """Validates detected log schema reliability, including multiline support."""
    
    def __init__(self, schema: dict, sample_lines: List[str]):
        self.schema = schema
        self.sample_lines = sample_lines
        self.parsed_results = []  # Now stores logical entries, not physical lines
        self.is_multiline = schema.get('is_multiline', False)
        
    def run_all_tests(self) -> SchemaReliabilityReport:
        """Run complete validation suite."""
        report = SchemaReliabilityReport(overall_score=0.0, is_reliable=False)
        
        # First, parse all sample lines to get data for testing
        self._parse_samples()
        
        # Run individual tests
        tests = [
            self._test_timestamp_coverage,
            self._test_timestamp_consistency,
            self._test_thread_coverage,
            self._test_session_key_coverage,
            self._test_cross_validation,
            self._test_schema_stability,
            self._test_parse_errors,
        ]
        
        # Add multiline-specific tests if enabled
        if self.is_multiline:
            tests.extend([
                self._test_continuation_pattern_accuracy,
                self._test_entry_size_distribution,
                self._test_stack_trace_grouping,
            ])
        
        scores = []
        for test in tests:
            try:
                result = test()
                report.add_result(result)
                scores.append(result.score)
            except Exception as e:
                report.add_result(ValidationResult(
                    test_name=test.__name__,
                    passed=False,
                    score=0.0,
                    details=f"Test failed with exception: {str(e)}"
                ))
        
        # Calculate overall score (weighted average)
        if scores:
            # Weight multiline tests slightly lower since they're secondary
            if self.is_multiline:
                # First 7 tests are core, rest are multiline
                core_scores = scores[:7]
                multi_scores = scores[7:]
                core_avg = sum(core_scores) / len(core_scores) if core_scores else 0
                multi_avg = sum(multi_scores) / len(multi_scores) if multi_scores else 1.0
                report.overall_score = (core_avg * 0.8) + (multi_avg * 0.2)
            else:
                report.overall_score = sum(scores) / len(scores)
                
            report.is_reliable = report.overall_score >= 0.7
            
        return report
    
    def _parse_samples(self):
        """Parse all sample lines into logical entries for reuse across tests."""
        self.parsed_results = []
        
        if self.is_multiline and self.schema.get('continuation_re'):
            # Use grouping generator for multiline logs
            from io import StringIO
            file_like = StringIO('\n'.join(self.sample_lines))
            
            for entry_lines in group_log_entries(file_like, self.schema):
                ts, pk, full_text = _parse_entry(entry_lines, self.schema)
                self.parsed_results.append({
                    'entry_lines': entry_lines,  # List of physical lines
                    'line_count': len(entry_lines),
                    'timestamp': ts,
                    'primary_key': pk,
                    'cleaned': full_text,  # Full multiline text
                    'first_line': entry_lines[0] if entry_lines else '',
                })
        else:
            # Single-line mode: each physical line is a logical entry
            for line in self.sample_lines:
                # Wrap single line in list for _parse_entry compatibility
                entry_lines = [line]
                ts, pk, full_text = _parse_entry(entry_lines, self.schema)
                self.parsed_results.append({
                    'entry_lines': entry_lines,
                    'line_count': 1,
                    'timestamp': ts,
                    'primary_key': pk,
                    'cleaned': full_text,
                    'first_line': line,
                })
    
    def _test_timestamp_coverage(self) -> ValidationResult:
        """Test: What % of LOGICAL ENTRIES have valid timestamps?"""
        total = len(self.parsed_results)
        with_ts = sum(1 for r in self.parsed_results if r['timestamp'] is not None)
        coverage = with_ts / total if total > 0 else 0.0
        
        passed = coverage >= 0.8  # 80% threshold
        return ValidationResult(
            test_name="Timestamp Coverage",
            passed=passed,
            score=coverage,
            details=f"{coverage:.1%} of logical entries have parsed timestamps",
            warnings=[] if coverage > 0.5 else ["Low timestamp coverage may indicate wrong pattern"]
        )
    
    def _test_timestamp_consistency(self) -> ValidationResult:
        """Test: Are timestamps monotonically increasing? (Most logs are chronological)"""
        timestamps = [r['timestamp'] for r in self.parsed_results if r['timestamp'] is not None]
        
        if len(timestamps) < 10:
            return ValidationResult(
                test_name="Timestamp Consistency",
                passed=True,
                score=1.0,
                details="Insufficient timestamps for consistency check"
            )
        
        # Check for monotonic increase (allow small out-of-order for multi-threaded logs)
        out_of_order = 0
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1] - timedelta(seconds=1):  # 1s tolerance
                out_of_order += 1
        
        consistency = 1.0 - (out_of_order / len(timestamps))
        passed = consistency >= 0.9
        
        return ValidationResult(
            test_name="Timestamp Consistency",
            passed=passed,
            score=consistency,
            details=f"{consistency:.1%} timestamps in expected order ({out_of_order} out-of-order)",
            warnings=["Out-of-order timestamps may indicate multi-source log aggregation"] if out_of_order > 0 else []
        )
    
    def _test_thread_coverage(self) -> ValidationResult:
        """Test: If thread pattern detected, what % of LOGICAL ENTRIES have thread IDs?"""
        if self.schema.get('thread_re') is None:
            return ValidationResult(
                test_name="Thread Coverage",
                passed=True,
                score=1.0,
                details="No thread pattern in schema (N/A)"
            )
        
        total = len(self.parsed_results)
        with_thread = sum(1 for r in self.parsed_results if r['primary_key'] is not None)
        coverage = with_thread / total if total > 0 else 0.0
        
        passed = coverage >= 0.5  # 50% threshold (threads may not be on every line)
        return ValidationResult(
            test_name="Thread Coverage",
            passed=passed,
            score=coverage,
            details=f"{coverage:.1%} of logical entries have thread/session IDs",
            warnings=["Low thread coverage may indicate wrong thread pattern"] if coverage < 0.3 else []
        )
    
    def _test_session_key_coverage(self) -> ValidationResult:
        """Test: Session keys should appear in at least some LOGICAL ENTRIES if detected."""
        session_keys = self.schema.get('session_keys', [])
        
        if not session_keys:
            return ValidationResult(
                test_name="Session Key Coverage",
                passed=True,
                score=1.0,
                details="No session keys in schema (N/A)"
            )
        
        total = len(self.parsed_results)
        with_session = 0
        for r in self.parsed_results:
            for compiled, _ in session_keys:
                if compiled.search(r['cleaned']):  # Search full multiline text
                    with_session += 1
                    break
        
        coverage = with_session / total if total > 0 else 0.0
        passed = coverage > 0.01  # At least 1% should have session keys
        
        return ValidationResult(
            test_name="Session Key Coverage",
            passed=passed,
            score=min(1.0, coverage * 10),  # Scale up for visibility
            details=f"{coverage:.1%} of logical entries contain session/transaction keys",
            warnings=["Session keys detected but rarely appear"] if coverage < 0.05 else []
        )
    
    def _test_cross_validation(self) -> ValidationResult:
        """Test: Split sample, detect on first half, validate on second half."""
        if len(self.sample_lines) < 50:
            return ValidationResult(
                test_name="Cross Validation",
                passed=True,
                score=1.0,
                details="Insufficient lines for cross-validation"
            )
        
        mid = len(self.sample_lines) // 2
        train_lines = self.sample_lines[:mid]
        test_lines = self.sample_lines[mid:]
        
        # Re-detect schema on training set (with same multiline settings)
        train_schema = detect_log_structure_hybrid(
            train_lines, 
            use_llm_fallback=False,  # Keep deterministic for validation
            enable_multiline=self.is_multiline
        )
        
        # Parse test set with train schema
        test_hits = 0
        if train_schema.get('is_multiline') and train_schema.get('continuation_re'):
            # Use grouping for multiline test
            from io import StringIO
            file_like = StringIO('\n'.join(test_lines))
            for entry_lines in group_log_entries(file_like, train_schema):
                ts, pk, full_text = _parse_entry(entry_lines, train_schema)
                if ts is not None:
                    test_hits += 1
        else:
            # Single-line test
            for line in test_lines:
                ts, pk, clean = _parse_entry([line], train_schema)
                if ts is not None:
                    test_hits += 1
        
        generalization = test_hits / len(test_lines) if test_lines else 0.0
        passed = generalization >= 0.7
        
        return ValidationResult(
            test_name="Cross Validation",
            passed=passed,
            score=generalization,
            details=f"Schema generalizes to {generalization:.1%} of held-out lines",
            warnings=["Schema may be overfit to training samples"] if generalization < 0.5 else []
        )
    
    def _test_schema_stability(self) -> ValidationResult:
        """Test: Detect schema on different random samples, check consistency."""
        import random
        
        if len(self.sample_lines) < 100:
            return ValidationResult(
                test_name="Schema Stability",
                passed=True,
                score=1.0,
                details="Insufficient lines for stability test"
            )
        
        # Detect on 3 different random samples
        detected_formats = []
        detected_multiline = []
        
        for i in range(3):
            sample = random.sample(self.sample_lines, min(200, len(self.sample_lines)))
            schema = detect_log_structure_hybrid(
                sample, 
                use_llm_fallback=False,
                enable_multiline=self.is_multiline
            )
            fmt = schema.get('timestamp_fmt', 'NONE')
            detected_formats.append(fmt)
            detected_multiline.append(schema.get('is_multiline', False))
        
        # Check if all detected the same format AND same multiline setting
        unique_formats = set(detected_formats)
        unique_multiline = set(detected_multiline)
        
        format_stable = len(unique_formats) == 1
        multiline_stable = len(unique_multiline) == 1
        
        stability = 1.0 if (format_stable and multiline_stable) else (0.5 if format_stable else 0.0)
        passed = format_stable and multiline_stable
        
        return ValidationResult(
            test_name="Schema Stability",
            passed=passed,
            score=stability,
            details=f"Detected {len(unique_formats)} format(s) and {len(unique_multiline)} multiline setting(s) across samples",
            warnings=["Schema detection is unstable across samples"] if not passed else []
        )
    
    def _test_parse_errors(self) -> ValidationResult:
        """Test: Check for common parsing anti-patterns."""
        issues = []
        
        # Check for obviously wrong timestamps (e.g., year 1900 default)
        default_year_count = sum(
            1 for r in self.parsed_results 
            if r['timestamp'] and r['timestamp'].year == 1900
        )
        if default_year_count > len(self.parsed_results) * 0.5:
            issues.append("Many timestamps default to year 1900 (missing year in format)")
        
        # Check for empty primary keys when pattern exists
        if self.schema.get('thread_re'):
            empty_pk = sum(1 for r in self.parsed_results if r['primary_key'] is None)
            if empty_pk > len(self.parsed_results) * 0.8:
                issues.append("Thread pattern exists but rarely matches")
        
        passed = len(issues) == 0
        score = 1.0 - (len(issues) * 0.25)
        
        return ValidationResult(
            test_name="Parse Error Detection",
            passed=passed,
            score=max(0.0, score),
            details="No parsing anti-patterns detected" if passed else "; ".join(issues),
            warnings=issues
        )
    
    # -------------------------------------------------------------------------
    # NEW: Multiline-specific validation tests
    # -------------------------------------------------------------------------
    
    def _test_continuation_pattern_accuracy(self) -> ValidationResult:
        """Test: Does the continuation regex correctly identify continuation lines?"""
        if not self.is_multiline or not self.schema.get('continuation_re'):
            return ValidationResult(
                test_name="Continuation Pattern Accuracy",
                passed=True,
                score=1.0,
                details="Multiline not enabled or no continuation pattern (N/A)"
            )
        
        continuation_re = self.schema['continuation_re']
        timestamp_re = self.schema.get('timestamp_re')
        
        if not timestamp_re:
            return ValidationResult(
                test_name="Continuation Pattern Accuracy",
                passed=False,
                score=0.0,
                details="Cannot validate continuation pattern without timestamp regex"
            )
        
        # Analyze each entry: continuations should NOT have timestamps at start
        false_positives = 0  # Lines marked as continuation but have timestamp
        false_negatives = 0  # Lines without timestamp but NOT marked as continuation
        
        for entry in self.parsed_results:
            entry_lines = entry['entry_lines']
            if len(entry_lines) <= 1:
                continue  # Single-line entry, nothing to validate
            
            # First line should NOT match continuation (it's the start)
            first_line = entry_lines[0]
            if continuation_re.match(first_line):
                false_positives += 1
            
            # Subsequent lines SHOULD match continuation (or have no timestamp)
            for line in entry_lines[1:]:
                stripped = _LINE_PREFIX_RE.sub('', line).lstrip()
                has_timestamp = timestamp_re.search(stripped) and timestamp_re.search(stripped).start() <= 10
                is_marked_continuation = continuation_re.match(line)
                
                if has_timestamp and is_marked_continuation:
                    false_positives += 1  # Continuation regex matched a new entry start
                elif not has_timestamp and not is_marked_continuation:
                    false_negatives += 1  # Should have been marked as continuation
        
        total_checks = sum(len(e['entry_lines']) - 1 for e in self.parsed_results if len(e['entry_lines']) > 1)
        if total_checks == 0:
            return ValidationResult(
                test_name="Continuation Pattern Accuracy",
                passed=True,
                score=1.0,
                details="No multiline entries to validate continuation pattern"
            )
        
        accuracy = 1.0 - ((false_positives + false_negatives) / total_checks)
        passed = accuracy >= 0.85
        
        return ValidationResult(
            test_name="Continuation Pattern Accuracy",
            passed=passed,
            score=accuracy,
            details=f"{accuracy:.1%} accuracy ({false_positives} FP, {false_negatives} FN out of {total_checks} checks)",
            warnings=["Continuation pattern may need refinement"] if accuracy < 0.9 else []
        )
    
    def _test_entry_size_distribution(self) -> ValidationResult:
        """Test: Are logical entry sizes reasonable? (Not too many 1-liners or huge blobs)"""
        if not self.is_multiline:
            return ValidationResult(
                test_name="Entry Size Distribution",
                passed=True,
                score=1.0,
                details="Single-line mode (N/A)"
            )
        
        line_counts = [r['line_count'] for r in self.parsed_results]
        if not line_counts:
            return ValidationResult(
                test_name="Entry Size Distribution",
                passed=True,
                score=1.0,
                details="No entries to analyze"
            )
        
        avg_lines = statistics.mean(line_counts)
        median_lines = statistics.median(line_counts)
        max_lines = max(line_counts)
        
        # Heuristics for reasonable distribution:
        # - Median should be 1-5 (most entries are short)
        # - Average should be < 10 (not dominated by huge stack traces)
        # - Max should be < 100 (sanity check for runaway grouping)
        median_ok = 1 <= median_lines <= 5
        avg_ok = avg_lines < 10
        max_ok = max_lines < 100
        
        passed = median_ok and avg_ok and max_ok
        score = (int(median_ok) + int(avg_ok) + int(max_ok)) / 3.0
        
        return ValidationResult(
            test_name="Entry Size Distribution",
            passed=passed,
            score=score,
            details=f"Entry sizes: avg={avg_lines:.1f}, median={median_lines}, max={max_lines}",
            warnings=[
                f"Median entry size {median_lines} seems unusual" if not median_ok else None,
                f"Average entry size {avg_lines:.1f} is high" if not avg_ok else None,
                f"Max entry size {max_lines} is very large" if not max_ok else None,
            ]
        )
    
    def _test_stack_trace_grouping(self) -> ValidationResult:
        """Test: If stack traces exist, are they properly grouped with their parent entry?"""
        if not self.is_multiline or not self.schema.get('stack_trace_re'):
            return ValidationResult(
                test_name="Stack Trace Grouping",
                passed=True,
                score=1.0,
                details="No stack trace pattern or multiline disabled (N/A)"
            )
        
        stack_trace_re = self.schema['stack_trace_re']
        
        # Count entries that contain stack traces
        entries_with_traces = 0
        traces_properly_grouped = 0
        
        for entry in self.parsed_results:
            entry_lines = entry['entry_lines']
            if len(entry_lines) <= 1:
                continue
            
            # Check if any line in entry matches stack trace pattern
            has_trace = any(stack_trace_re.match(_LINE_PREFIX_RE.sub('', l)) for l in entry_lines)
            if not has_trace:
                continue
            
            entries_with_traces += 1
            
            # Stack traces should NOT be on the first line (they're continuations)
            first_line = _LINE_PREFIX_RE.sub('', entry_lines[0])
            if not stack_trace_re.match(first_line):
                traces_properly_grouped += 1
        
        if entries_with_traces == 0:
            return ValidationResult(
                test_name="Stack Trace Grouping",
                passed=True,
                score=1.0,
                details="No stack traces found in samples to validate"
            )
        
        grouping_rate = traces_properly_grouped / entries_with_traces
        passed = grouping_rate >= 0.9
        
        return ValidationResult(
            test_name="Stack Trace Grouping",
            passed=passed,
            score=grouping_rate,
            details=f"{grouping_rate:.1%} of stack traces properly grouped with parent entry",
            warnings=["Stack traces may be split across entries"] if grouping_rate < 0.95 else []
        )

# Assuming these are imported from your existing modules
# from your_module import detect_log_structure_hybrid, SchemaValidator

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for log analysis configuration.
    """
    parser = argparse.ArgumentParser(
        description="Detect and validate log structure from a sample log file."
    )
    
    # Positional argument for file path
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to the log file to analyze."
    )
    
    # Optional argument for detection sample size
    parser.add_argument(
        "--detect-lines",
        type=int,
        default=1000,
        help="Number of lines to use for structure detection (default: 1000)."
    )
    
    # Optional argument for validation sample size
    parser.add_argument(
        "--validate-lines",
        type=int,
        default=2000,
        help="Number of lines to use for schema validation (default: 2000)."
    )

    return parser.parse_args()

def read_log_lines(file_path: str, max_lines: int) -> List[str]:
    """
    Read up to max_lines from the specified file.
    
    Args:
        file_path: Path to the log file.
        max_lines: Maximum number of lines to read.
        
    Returns:
        List of log lines.
    """
    lines = []
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
        
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
    except Exception as e:
        raise IOError(f"Failed to read log file: {e}")
    
    return lines

def main():
    """
    Main entry point for log structure detection and validation.
    """
    args = parse_arguments()

    # Validate line counts
    if args.detect_lines <= 0 or args.validate_lines <= 0:
        print("Error: Line counts must be positive integers.")
        sys.exit(1)

    # Determine the total lines needed to read (optimization)
    # We read once up to the maximum required by either task
    total_lines_to_read = max(args.detect_lines, args.validate_lines)

    print(f"Analyzing: {args.log_path}")
    print(f"Reading up to {total_lines_to_read} lines...")

    try:
        all_sample_lines = read_log_lines(args.log_path, total_lines_to_read)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not all_sample_lines:
        print("Error: Log file is empty or could not be read.")
        sys.exit(1)

    # Slice lines for specific tasks
    detection_lines = all_sample_lines[:args.detect_lines]
    validation_lines = all_sample_lines[:args.validate_lines]

    print(f"Using {len(detection_lines)} lines for detection...")
    schema = detect_log_structure_hybrid(detection_lines, use_llm_fallback=True)
    
    if not schema:
        print("Failed to detect schema.")
        sys.exit(1)

    print("Detected Schema:")
    print(schema)

    print(f"Using {len(validation_lines)} lines for validation...")
    validator = SchemaValidator(schema, validation_lines)
    report = validator.run_all_tests()
    
    print("\nSchema Validation Report:")
    print(report)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
