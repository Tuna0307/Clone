"""
Centralized runtime artifact paths.

All generated files should live below outputs/ instead of the repository root.
Keeping this logic in one module prevents the Streamlit app, CLI pipeline, and
follow-up retrieval code from drifting to different artifact locations.
"""

from __future__ import annotations

import os
from pathlib import Path


# Root for generated runtime artifacts. This directory is intentionally ignored
# by Git because it can contain uploaded logs, debug prompts, vector indexes,
# and generated PDF reports.
OUTPUT_ROOT = Path("outputs")
DEBUG_DIR = OUTPUT_ROOT / "debug"
FAISS_DIR = OUTPUT_ROOT / "faiss"
REPORT_DIR = OUTPUT_ROOT / "reports"
UPLOAD_DIR = OUTPUT_ROOT / "uploads"


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    """
    Create the parent directory for a path.

    Args:
        path: File path whose parent should exist
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def debug_evidence_path(file_name: str) -> str:
    """
    Return the debug evidence path for a source file.

    Args:
        file_name: Source file base name

    Returns:
        Debug evidence file path
    """
    return str(DEBUG_DIR / f"debug_evidence_{file_name}.txt")


def faiss_index_dir(file_name: str) -> str:
    """
    Return the FAISS artifact directory for a source file.

    Args:
        file_name: Source file base name

    Returns:
        FAISS artifact directory path
    """
    file_stem = os.path.splitext(file_name)[0]
    return str(FAISS_DIR / f"faiss_index_{file_stem}")


def report_path(filename: str = "IAM_Forensic_Report.pdf") -> str:
    """
    Return the report output path.

    Args:
        filename: Report filename

    Returns:
        Report file path
    """
    return str(REPORT_DIR / filename)


def upload_session_dir(session_id: str) -> str:
    """
    Return the upload directory for a Streamlit session.

    Args:
        session_id: Session identifier

    Returns:
        Upload directory path
    """
    return str(UPLOAD_DIR / session_id)
