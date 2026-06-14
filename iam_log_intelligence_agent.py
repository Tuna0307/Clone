"""
Legacy v1 IAM log agent.

This file keeps the original LangChain tool-calling/RAG implementation for
reference. The current Streamlit app uses iam_log_intelligence_agent_hybridChunking2.py
instead, because hybrid/request-aware extraction gives better log citations.
"""

import os
import json
from typing import Optional, Iterator
from tqdm import tqdm

from llm_factory import get_llm, get_embeddings
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================================
# LLM and Embeddings Configuration (provider-agnostic via llm_factory)
# ============================================================================

llm = get_llm()
embeddings = get_embeddings()

# Generic text splitter used by the legacy v1 agent. The current production
# pipeline avoids this splitter so thread/request context is preserved.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=500,
    separators=["\n\n", "\n", " ", ""]
)

# Global variable to store current index path
current_index_path = None

# ============================================================================
# Log Indexing Functions (Optimized for Large Scale)
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
        
    log_files = []
    # Extensions to look for
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
    """Return human-readable file size (e.g., '524.6 MB')."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def stream_file_lines(file_path: str) -> Iterator[str]:
    """
    Generator that yields lines from a file one at a time.
    Memory-efficient for extremely large files.
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            yield line


def index_files_incrementally(file_paths: list[str]) -> str:
    """
    Memory-safe incremental indexing for multiple/large files.
    Reads, chunks, and indexes files one by one to minimize RAM usage.
    Now includes progress bars for visibility on large files.
    
    Args:
        file_paths: List of absolute file paths to index
        
    Returns:
        Path to the saved FAISS index
    """
    global current_index_path
    
    vectorstore = None
    total_chunks = 0
    total_files = len(file_paths)
    
    print(f"\n[Indexer] Starting incremental indexing for {total_files} files...")
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        print(f"  ({i+1}/{total_files}) Processing: {file_name} ({format_file_size(file_size)})")
        
        try:
            # Count lines first for accurate progress (fast scan)
            print(f"    Counting lines...", end=" ", flush=True)
            line_count = 0
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for _ in f:
                    line_count += 1
            print(f"{line_count:,} lines")
            
            # Stage 1: Read file with progress
            content_parts = []
            with tqdm(total=line_count, desc="    Reading", unit=" lines", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for line in stream_file_lines(file_path):
                    content_parts.append(line)
                    pbar.update(1)
            
            content = ''.join(content_parts)
            del content_parts  # Free memory
            
            if not content:
                print(f"    Skipping empty file.")
                continue

            # Stage 2: Split content into chunks
            print(f"    Chunking...", end=" ", flush=True)
            chunks = splitter.split_text(content)
            print(f"{len(chunks):,} chunks created")
            
            del content  # Free memory after chunking
            
            # Stage 3: Create documents and index with batch progress
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"file": file_name, "source": file_path, "chunk_id": idx}
                )
                for idx, chunk in enumerate(chunks)
            ]
            
            del chunks  # Free memory
            
            # Index with batch processing and progress bar
            batch_size = 100  # Process 100 documents at a time for better progress visibility
            
            with tqdm(total=len(docs), desc="    Indexing", unit=" docs",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                if vectorstore is None:
                    # First batch creates the store
                    first_batch = docs[:batch_size]
                    vectorstore = FAISS.from_documents(first_batch, embeddings)
                    pbar.update(len(first_batch))
                    remaining_docs = docs[batch_size:]
                else:
                    remaining_docs = docs
                
                # Add remaining documents in batches
                for j in range(0, len(remaining_docs), batch_size):
                    batch = remaining_docs[j:j + batch_size]
                    vectorstore.add_documents(batch)
                    pbar.update(len(batch))
                
            total_chunks += len(docs)
            print(f"    ✓ Indexed {len(docs):,} chunks from {file_name}")
            
            del docs  # Force cleanup
            
        except Exception as e:
            print(f"    [Error] Failed to process {file_name}: {e}")
            continue

    if vectorstore is None:
        raise ValueError("No valid log content found to index.")

    # Save final index
    index_name = "faiss_index_combined" if len(file_paths) > 1 else f"faiss_index_{os.path.basename(file_paths[0])}"
    
    print(f"\n[Indexer] Saving index with {total_chunks:,} total chunks...")
    vectorstore.save_local(index_name)
    current_index_path = index_name
    
    print(f"[Indexer] Index saved to '{index_name}'")
    return index_name

# Legacy wrapper for compatibility (redirects to new system)
def index_logs(log_text: str, file_name: str = "logs") -> str:
    # Save text to a temp file to use the new robust pipeline
    temp_path = f"temp_{file_name}"
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(log_text)
    return index_files_incrementally([os.path.abspath(temp_path)])

def index_multiple_logs(log_files: dict) -> str:
    # Not used in new main flow, but kept for library compatibility
    # Writes dict content to temp files
    temp_paths = []
    for name, content in log_files.items():
        p = f"temp_{name}"
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        temp_paths.append(os.path.abspath(p))
    return index_files_incrementally(temp_paths)

# ============================================================================
# Agent Tools
# ============================================================================

@tool
def retrieve_log_chunks(query: str, top_k: int = 50) -> str:
    """
    Retrieve relevant log chunks from the vector index using semantic search.
    Use this tool to find log entries related to a specific query.
    
    Args:
        query: Search query describing what log entries to find
        top_k: Number of top matching chunks to retrieve (default: 50)
        
    Returns:
        Concatenated relevant log chunks with metadata AND CITATION IDs
    """
    global current_index_path
    
    if not current_index_path:
        return "Error: No log index available. Please index logs first."
    
    try:
        vectorstore = FAISS.load_local(
            current_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        results = vectorstore.similarity_search(query, k=top_k)
        
        output = []
        for i, r in enumerate(results):
            # Create a unique, verifiable citation ID
            citation_id = f"REF_{r.metadata.get('chunk_id', 'X')}_{i}"
            
            # Format: [ID] Content (File)
            formatted_chunk = (
                f"[{citation_id}]\n"
                f"Source: {r.metadata.get('file', 'unknown')}\n"
                f"Content: {r.page_content}\n"
            )
            output.append(formatted_chunk)
        
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Error retrieving chunks: {str(e)}"


@tool
def parse_log_structure(sample_query: str = "log format timestamp thread") -> str:
    """
    Parse and analyze the structure of the indexed logs.
    Retrieves sample log entries and identifies format, fields, thread IDs, timestamps.
    
    Args:
        sample_query: Query to retrieve representative log samples
        
    Returns:
        Structured description of the log format
    """
    # Retrieve sample log entries
    samples = retrieve_log_chunks.invoke({"query": sample_query, "top_k": 20})
    
    if samples.startswith("Error"):
        return samples
    
    prompt = f"""Analyze the following log samples and describe the log structure.

LOG SAMPLES:
{samples}

Provide a structured analysis including:
1. **Log Format Pattern**: Describe the logging format/pattern used (e.g., timestamp format, field order)
2. **Fields Identified**: List all distinct fields (timestamp, log level, thread ID, component, message, etc.)
3. **Timestamp Format**: Exact format of timestamps (e.g., ISO 8601, custom format)
4. **Thread/Session Identifiers**: How threads or sessions are identified
5. **Component/Class Names**: Pattern for component or class identifiers
6. **Log Levels Used**: Which log levels appear (INFO, WARN, ERROR, DEBUG, etc.)

Be specific and use examples from the logs."""

    response = llm.invoke(prompt)
    return response.content


@tool
def aggregate_logs(aggregation_query: str, structure_description: str = "") -> str:
    """
    Aggregate and group related log entries into sessions or requests.
    Groups scattered log lines by thread ID, session ID, or transaction ID.
    
    Args:
        aggregation_query: Query describing what to aggregate (e.g., "user authentication session")
        structure_description: Optional description of log structure from parse_log_structure
        
    Returns:
        Aggregated log groups with session/thread identification
    """
    # Retrieve relevant log chunks for aggregation
    chunks = retrieve_log_chunks.invoke({"query": aggregation_query, "top_k": 30})
    
    if chunks.startswith("Error"):
        return chunks
    
    prompt = f"""Analyze the following log entries and aggregate them into logical groups.

{f"LOG STRUCTURE: {structure_description}" if structure_description else ""}

LOG ENTRIES:
{chunks}

Tasks:
1. **Identify Grouping Criteria**: Determine how logs should be grouped (by thread ID, session ID, user, transaction, etc.)
2. **Group Related Entries**: Organize log entries into their logical groups
3. **Sequence Events**: Within each group, order events chronologically
4. **Summarize Each Group**: Provide a brief summary of what each group represents

Output format:
For each identified group:
- Group ID/Name
- Time range
- Number of entries
- Key events in sequence
- Brief summary"""

    response = llm.invoke(prompt)
    return response.content


@tool
def classify_log_types(classification_query: str = "authentication token MFA access") -> str:
    """
    Classify log entries into different IAM operation types.
    Labels groups as authentication, token validation, MFA, authorization, etc.
    
    Args:
        classification_query: Query to find logs for classification
        
    Returns:
        Classification of log entries by IAM operation type
    """
    # Retrieve relevant log chunks
    chunks = retrieve_log_chunks.invoke({"query": classification_query, "top_k": 25})
    
    if chunks.startswith("Error"):
        return chunks
    
    prompt = f"""Classify the following IAM log entries into operation categories.

LOG ENTRIES:
{chunks}

Classify each distinct operation/event into one of these IAM categories:
1. **Authentication**: Login attempts, password validation, SSO
2. **Token Operations**: Token creation, validation, refresh, revocation
3. **MFA/2FA**: Multi-factor authentication events
4. **Authorization**: Access control decisions, permission checks
5. **Session Management**: Session creation, timeout, termination
6. **User Management**: User CRUD operations, profile updates
7. **Audit/Logging**: Audit trail entries, security events
8. **Error/Exception**: Errors, failures, exceptions
9. **System/Health**: System status, health checks, startup/shutdown

For each category found:
- List example log entries
- Count of occurrences
- Notable patterns or concerns"""

    response = llm.invoke(prompt)
    return response.content


@tool
def detect_anomalies(anomaly_query: str = "error warning failed exception timeout") -> str:
    """
    Detect and score anomalies in the log entries.
    Iteratively flags abnormal patterns via targeted retrieval.
    
    Args:
        anomaly_query: Query to find potentially anomalous log entries
        
    Returns:
        List of detected anomalies with severity scores and descriptions
    """
    # Retrieve potentially anomalous log chunks
    # We broaden the search to catch crypto issues
    chunks = retrieve_log_chunks.invoke({"query": "SecurityException Encryption Decryption Certificate KeyStore " + anomaly_query, "top_k": 40})
    
    if chunks.startswith("Error"):
        return chunks
    
    prompt = f"""Analyze the following log entries for anomalies.
PRIORITIZE Security/Crypto failures over Network/Database errors.

LOG ENTRIES:
{chunks}

Detect and evaluate:

1. **CRITICAL: Cryptographic Failures**:
   - Decryption/Encryption errors
   - Certificate/Key issues
   - Signature validation failures

2. **HIGH: Authentication/Security**:
   - Login failures
   - Token validation issues

3. **MEDIUM: System/Network**:
   - Timeouts
   - Database connection errors

For each anomaly detected, provide:
- **Severity**: CRITICAL / HIGH / MEDIUM / LOW
- **Category**: Error/Security/Performance/Operational
- **Description**: What was detected
- **Evidence**: Relevant log excerpts (Must cite REF_ID)
- **Impact**: Potential consequences
- **Recommendation**: Suggested action"""

    response = llm.invoke(prompt)
    return response.content


# ============================================================================
# Generate Incident Report Tool (Commented out as requested)
# ============================================================================

# @tool
# def generate_incident_report(anomalies: str, context_summary: str = "") -> str:
#     """
#     Generate a final incident report based on detected anomalies.
#     Produces a concise, actionable report for flagged anomalies.
#     
#     Args:
#         anomalies: Anomaly detection results from detect_anomalies
#         context_summary: Optional additional context
#         
#     Returns:
#         Formatted incident report
#     """
#     prompt = f"""Generate a professional incident report based on the following analysis.
# 
# DETECTED ANOMALIES:
# {anomalies}
# 
# {f"ADDITIONAL CONTEXT: {context_summary}" if context_summary else ""}
# 
# Create an incident report with:
# 
# 1. **Executive Summary**: Brief overview (2-3 sentences)
# 
# 2. **Incident Classification**:
#    - Severity Level
#    - Category
#    - Affected Systems/Components
# 
# 3. **Timeline**: Chronological sequence of events
# 
# 4. **Root Cause Analysis**: Identified or suspected root cause
# 
# 5. **Impact Assessment**:
#    - Users/Systems affected
#    - Business impact
#    - Data implications
# 
# 6. **Recommendations**:
#    - Immediate actions
#    - Short-term mitigations
#    - Long-term improvements
# 
# 7. **Appendix**: Key log excerpts as evidence
# 
# Format the report professionally for stakeholder review."""
# 
#     response = llm.invoke(prompt)
#     return response.content


# ============================================================================
# Manual Agent Loop (Reliable)
# ============================================================================

# Define available tools map
tools_map = {
    "retrieve_log_chunks": retrieve_log_chunks,
    "parse_log_structure": parse_log_structure,
    "aggregate_logs": aggregate_logs,
    "classify_log_types": classify_log_types,
    "detect_anomalies": detect_anomalies,
}

tools_list = list(tools_map.values())

# Bind tools to the model
model_with_tools = llm.bind_tools(tools_list)

import re
import ast

def parse_pseudo_tool_call(text: str):
    """
    Fallback: Parse tool calls from text if the model writes code instead of using native tools.
    """
    for tool_name in tools_map.keys():
        # Look for tool_name(args)
        pattern = re.compile(re.escape(tool_name) + r"\((.*?)\)", re.DOTALL)
        match = pattern.search(text)
        if match:
            args_str = match.group(1)
            try:
                # Try to parse arguments as python dict or keyword args
                # This is a simple heuristic
                if "=" in args_str:
                    # Convert key="value" to dict
                    # This is tricky without full parsing, but we can try
                    # strict key=value parsing or just use the whole string as the first arg if simple
                    pass
                
                return tool_name, args_str
            except:
                pass
    return None, None

def run_agent_loop(initial_messages: list) -> str:
    """
    Manually execute the agent loop to ensure tools are called and results returned.
    """
    messages = list(initial_messages)
    max_steps = 10
    
    print("\n[Starting Analysis Loop]")
    
    for step in range(max_steps):
        # Invoke model
        response = model_with_tools.invoke(messages)
        messages.append(response)
        
        tool_calls_to_make = []
        
        # 1. Native Tool Calls
        if response.tool_calls:
            print(f"  Step {step+1}: AI requesting {len(response.tool_calls)} tool(s) (Native)...")
            for tc in response.tool_calls:
                tool_calls_to_make.append({
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"]
                })
        
        # 2. Fallback: Parse text for tool calls if native failed
        elif response.content:
            # Check for known tool names in content
            found_pseudo = False
            for tool_name in tools_map.keys():
                if f"{tool_name}(" in response.content:
                    print(f"  Step {step+1}: Detected pseudo-tool call for '{tool_name}' in text...")
                    
                    # Simple regex to get the content inside quotes if present
                    # Use raw strings to avoid SyntaxWarnings
                    pattern = re.compile(re.escape(tool_name) + r"\s*\(\s*(?:.*?=\s*)?['\"](.*?)['\"]")
                    match = pattern.search(response.content)
                    args = {}
                    if match:
                        if tool_name == "retrieve_log_chunks": args = {"query": match.group(1)}
                        elif tool_name == "parse_log_structure": args = {"sample_query": match.group(1)}
                        elif tool_name == "aggregate_logs": args = {"aggregation_query": match.group(1)}
                        elif tool_name == "classify_log_types": args = {"classification_query": match.group(1)}
                        elif tool_name == "detect_anomalies": args = {"anomaly_query": match.group(1)}
                    
                    # Generate a synthetic call ID
                    synthetic_id = f"call_{step}_{tool_name}"
                    
                    tool_calls_to_make.append({
                        "name": tool_name,
                        "args": args,
                        "id": synthetic_id
                    })
                    
                    # CRITICAL FIX: Patch the last message to look like a real tool call
                    # This satisfies Bedrock's validation requirement
                    # We must also CLEAR the content to avoid "content and tool_calls in same turn" error
                    messages[-1].content = "" 
                    messages[-1].tool_calls = [{
                        "name": tool_name,
                        "args": args,
                        "id": synthetic_id,
                        "type": "tool_call"
                    }]
                    
                    found_pseudo = True
                    break 
            
            if not found_pseudo and "Summary of findings" in response.content:
                 print("  Analysis complete (Summary detected).")
                 return response.content

        if tool_calls_to_make:
            for tc in tool_calls_to_make:
                tool_name = tc["name"]
                tool_args = tc["args"]
                call_id = tc["id"]
                
                print(f"    -> Executing '{tool_name}' with args {tool_args}...")
                
                # Execute tool
                if tool_name in tools_map:
                    try:
                        tool_func = tools_map[tool_name]
                        # Invoke tool
                        tool_output = tool_func.invoke(tool_args)
                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"
                else:
                    tool_output = f"Error: Tool '{tool_name}' not found."
                
                # Append tool result
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))
        else:
            # No more tools, return final answer
            print("  Analysis complete.")
            return response.content
            
    return response.content

# ============================================================================
# Map-Reduce Architecture for Large Scale Log Analysis
# ============================================================================

import shutil
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def load_search_config(config_path: str = "search_config.json") -> dict:
    """
    Load search configuration from JSON file.
    Returns default config if file is missing or invalid.
    """
    default_config = {
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

def export_to_pdf(report_text: str, filename: str = "IAM_Forensic_Report.pdf"):
    """
    Export the final text report to a professional PDF file.
    """
    print(f"\n[Export] Generating PDF report: {filename}...")
    
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom Styles
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
            borderPadding=5
        )

        lines = report_text.split('\n')
        
        story.append(Paragraph("IAM Forensic Investigation Report", title_style))
        story.append(Spacer(1, 12))
        
        buffer_text = []
        
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
        print(f"[Export] PDF saved successfully to: {os.path.abspath(filename)}")
        return os.path.abspath(filename)
        
    except Exception as e:
        print(f"[Export] Failed to generate PDF: {e}")
        return None

def analyze_single_file(file_path: str) -> dict:
    """
    [MAP STEP] Analyze a single log file in isolation.
    Indexes the file, runs a targeted hunt for errors using Configured Buckets,
    and returns a structured finding report.
    """
    file_name = os.path.basename(file_path)
    print(f"\n[MAP] Analyzing file: {file_name}...")
    
    # 1. Index just this file
    index_path = ""
    try:
        index_path = index_files_incrementally([file_path])
    except Exception as e:
        print(f"  [Error] Indexing failed: {e}")
        return {"file": file_name, "error": str(e), "findings": []}

    # 2. Targeted Hunter Prompts (Diversity Search)
    config = load_search_config()
    findings = []
    
    print(f"  [MAP] executing {len(config['buckets'])} search buckets...")
    
    for bucket in config["buckets"]:
        bucket_name = bucket.get("name", "Unknown")
        query = bucket.get("query", "Error")
        top_k = bucket.get("top_k", 10)
        
        results = retrieve_log_chunks.invoke({"query": query, "top_k": top_k})
        
        if "Exception" in results or "Error" in results or "Failed" in results or "Refused" in results:
            findings.append(f"### Verified Evidence (Category: {bucket_name}):\n{results}")
        
    # 3. Cleanup
    try:
        if index_path and os.path.exists(index_path):
            import gc
            gc.collect() 
            shutil.rmtree(index_path)
            print(f"  [Clean] Removed temp index: {index_path}")
    except Exception as e:
        print(f"  [Warning] Failed to clean up index {index_path}: {e}")
    
    print(f"  [MAP] Finished {file_name}. Extracted evidence from {len(findings)} categories.")
    
    return {
        "file": file_name,
        "findings": findings
    }

def consolidate_reports(all_findings: list[dict]) -> str:
    """
    [REDUCE STEP] Synthesize findings from all files into a final report.
    Includes smart summarization to prevent context window overflow.
    """
    print(f"\n[REDUCE] Consolidating findings from {len(all_findings)} files...")
    
    # Compile the "Case File" with strict length limits per file
    compiled_evidence = ""
    critical_findings_count = 0
    
    for report in all_findings:
        if not report["findings"]:
            continue
            
        file_name = report['file']
        findings_text = "\n".join(report["findings"])
        
        # Smart Truncation: Limit each file's evidence to ~2000 chars to save tokens
        # We prioritize the first few matches as they usually contain the key error
        if len(findings_text) > 2000:
            findings_text = findings_text[:2000] + "\n... [Evidence Truncated for Conciseness] ..."
            
        compiled_evidence += f"\n\n=== EVIDENCE FROM {file_name} ===\n"
        compiled_evidence += findings_text
        critical_findings_count += 1

    if not compiled_evidence.strip():
        return "No critical anomalies detected in any files."

    print(f"  Consolidated evidence from {critical_findings_count} files.")

    # Final Agent Run
    system_text = """You are a Lead Forensic Investigator.
You have received evidence files from multiple log sources.
Your job is to correlate them and write the Final Incident Report.

# REPORT STRUCTURE:
## 1. Executive Summary
(What is the root cause? How widespread is it?)

## 2. Root Cause Analysis
(Combine clues. e.g., "File A showed a timeout, which matches the crash in File B.")

## 3. Verified Evidence
(Quote specific error messages from the provided evidence texts. USE THE REF_ID provided in the text.)

## 4. Recommendations
(Technical fixes.)
"""

    user_prompt = f"""Here is the compiled evidence from the forensic team (Truncated for brevity):
{compiled_evidence}

Generate the Final Forensic Report.
"""

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating final report: {str(e)}\n\nPartial Evidence Collected:\n{compiled_evidence[:5000]}"

# ============================================================================
# Pipeline Execution
# ============================================================================

def run_pipeline(path_input: str) -> str:
    """
    Run the Map-Reduce pipeline.
    """
    # 1. Discovery
    log_files = get_log_files_from_path(path_input)
    if not log_files:
        return "Error: No log files found."

    # 2. Map (Analyze each file)
    all_findings = []
    for f in log_files:
        report = analyze_single_file(f)
        all_findings.append(report)

    # 3. Reduce (Consolidate)
    final_report = consolidate_reports(all_findings)
    
    # 4. Export to PDF
    export_to_pdf(final_report)
    
    return final_report


def interactive_mode():
    """
    Run the agent in interactive mode for ad-hoc queries.
    """
    print("\n" + "="*60)
    print("IAM Log Intelligence Agent - Interactive Mode")
    print("="*60)
    print("Type 'exit' or 'quit' to end the session.")
    print("="*60 + "\n")
    
    history = [
        SystemMessage(content="You are an expert IAM Log Intelligence Agent. Use tools to answer user queries.")
    ]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            history.append(HumanMessage(content=user_input))
            
            # Run loop with copy of history to allow recursion without polluting main history logic too much
            # (Simplification: in this manual loop, we just pass the history)
            response_content = run_agent_loop(history)
            
            # We append the final answer to history, but the intermediate tool calls are inside run_agent_loop
            # For a simple chat, we just append the AI's final text
            history.append(AIMessage(content=response_content))
            
            print(f"\nAgent: {response_content}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("IAM Log Intelligence Agent")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        # User provided a path (File OR Folder)
        path_input = sys.argv[1]
        
        if os.path.exists(path_input):
            try:
                result = run_pipeline(path_input)
                print("\n" + "="*60)
                print("ANALYSIS COMPLETE")
                print("="*60)
                print(result)
            except Exception as e:
                print(f"Error during execution: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Error: Path not found: {path_input}")
            sys.exit(1)
            
    else:
        # Interactive mode
        print("No log file/folder provided. Starting interactive mode...")
        print("You can index logs using the following commands in interactive mode:")
        print("  - 'analyze <filepath>' to analyze a specific log file")
        print("  - Or ask questions about previously indexed logs\n")
        interactive_mode()
