"""PDF export and report consolidation for the IAM pipeline.

Extracted from iam_log_intelligence_agent_hybridChunking2.py as part of a
conservative modular refactor.
"""

import html
import os
import re
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from artifact_paths import ensure_parent_dir, report_path
from langchain_core.messages import HumanMessage, SystemMessage
from llm_factory import get_llm
from pipeline.constants import (
    BENIGN_CHUNK_MAX_CHARS,
    REDUCE_EVIDENCE_BUDGET_CHARS,
    REDUCE_PER_FILE_CAP_CHARS,
)
from pipeline.progress import emit_ui_progress
from pipeline.prompts_api import build_reduce_system_prompt, build_reduce_user_message
from pipeline.references import (
    _format_server_monitoring_reduce_report,
    _replace_chunk_refs_with_original_references,
)

llm = get_llm()


def _finalize_reduce_report(report_text: str, all_findings: list[dict], *, mode: str) -> str:
    """Apply mode-appropriate citation post-processing to the reduce LLM output."""
    if mode == "server_monitoring":
        return _format_server_monitoring_reduce_report(report_text)
    return _replace_chunk_refs_with_original_references(report_text, all_findings)


def _markdown_links_to_reportlab(text: str) -> str:
    """
    Convert Markdown links into ReportLab paragraph links safely.

    Args:
        text: Markdown-ish report line

    Returns:
        ReportLab-safe paragraph markup
    """
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    result_parts: list[str] = []
    last_end = 0

    for match in pattern.finditer(text):
        result_parts.append(html.escape(text[last_end:match.start()]))
        label = html.escape(match.group(1))
        href = html.escape(match.group(2), quote=True)
        result_parts.append(f'<a href="{href}" color="blue"><u>{label}</u></a>')
        last_end = match.end()

    result_parts.append(html.escape(text[last_end:]))
    return ''.join(result_parts)


def consolidate_reports(all_findings: list[dict], *, mode: str = "api_request") -> str:
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
    emit_ui_progress(f"[REDUCE] Consolidating findings from {len(all_findings)} file(s)...")

    # Compile evidence from all files
    compiled_evidence = ""
    contributing_files = 0
    failed_files = 0

    for report in all_findings:
        if report.get("query_valid", True) is False:
            failed_files += 1
            print(f"  [WARNING] Skipping {report['file']} - invalid query window")
            continue

        findings_text = report.get("findings", "")
        if not findings_text.strip():
            failed_files += 1
            print(f"  [WARNING] Skipping {report['file']} - no findings (likely LLM error)")
            continue

        file_name = report['file']
        chunk_count = report.get('chunk_count', 0)
        high_count = report.get('high_anomaly_count', 0)
        category = report.get('category', 'unclassified')
        subcategory = report.get('subcategory', 'unclassified')
        status = report.get('status', 'ok')

        # Smart truncation: cap per-file evidence to stay within context window
        if len(findings_text) > REDUCE_PER_FILE_CAP_CHARS:
            findings_text = findings_text[:REDUCE_PER_FILE_CAP_CHARS] + "\n... [Findings Truncated for Conciseness] ..."

        compiled_evidence += (
            f"\n\n{'='*50}\n"
            f"=== FILE: {file_name} | Category: {category}/{subcategory} | Status: {status} | Chunks: {chunk_count} | High-anomaly: {high_count} ===\n"
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
    if mode == "api_request":
        emit_ui_progress(
            f"Compiled evidence from {contributing_files} file(s) ({failed_files} failed)."
        )

    # ---- Final LLM consolidation ----
    system_text = build_reduce_system_prompt(mode)
    user_prompt = build_reduce_user_message(compiled_evidence)

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_prompt),
    ]

    try:
        print(f"  [DEBUG] Sending {len(compiled_evidence):,} chars to final LLM")
        if mode == "api_request":
            emit_ui_progress(f"  [DEBUG] Sending {len(compiled_evidence):,} chars to final LLM")
        response = llm.invoke(messages)
        print(f"  [DEBUG] Final LLM returned {len(response.content):,} chars")
        if mode == "api_request":
            emit_ui_progress(f"  [DEBUG] Final LLM returned {len(response.content):,} chars")
        return _finalize_reduce_report(response.content, all_findings, mode=mode)
    except Exception as e:
        error_msg = str(e)
        print(f"  [ERROR] Final LLM call failed: {error_msg}")
        # Return a proper error report instead of trying to format the error as findings
        error_report = (
            f"# AGENT ERROR - Report Generation Failed\n\n"
            f"The final report could not be generated due to an LLM timeout or error:\n"
            f"`{error_msg}`\n\n"
            f"This is an infrastructure issue with the AI service, not an analysis result.\n\n"
            f"## Partial Evidence Summary\n"
            f"The following files were analyzed but could not be consolidated:\n"
            f"{compiled_evidence[:BENIGN_CHUNK_MAX_CHARS]}"
        )
        return _finalize_reduce_report(error_report, all_findings, mode=mode)


def export_to_pdf(report_text: str, filename: str = "IAM_Forensic_Report.pdf") -> Optional[str]:
    """
    Export the final text report to a professional PDF file.

    Args:
        report_text: Markdown-formatted report string
        filename:    Output PDF filename

    Returns:
        Absolute path to the generated PDF, or None on failure
    """
    output_path = report_path(filename)
    print(f"\n[Export] Generating PDF report: {output_path}...")

    try:
        ensure_parent_dir(output_path)
        doc = SimpleDocTemplate(output_path, pagesize=letter)
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
                    story.append(Paragraph(_markdown_links_to_reportlab(" ".join(buffer_text)), body_style))
                    buffer_text = []
                    story.append(Spacer(1, 6))

                clean_header = line.replace('#', '').strip()
                story.append(Paragraph(_markdown_links_to_reportlab(clean_header), heading_style))
                story.append(Spacer(1, 6))

            elif line.startswith('*') or line.startswith('-') or '[REF_' in line:
                if buffer_text:
                    story.append(Paragraph(_markdown_links_to_reportlab(" ".join(buffer_text)), body_style))
                    buffer_text = []
                story.append(Paragraph(_markdown_links_to_reportlab(line), code_style))
                story.append(Spacer(1, 4))

            else:
                buffer_text.append(line)

        if buffer_text:
            story.append(Paragraph(_markdown_links_to_reportlab(" ".join(buffer_text)), body_style))

        doc.build(story)
        abs_path = os.path.abspath(output_path)
        print(f"[Export] PDF saved successfully to: {abs_path}")
        return abs_path

    except Exception as e:
        print(f"[Export] Failed to generate PDF: {e}")
        return None
