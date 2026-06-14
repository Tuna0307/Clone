"""
Helpers for Streamlit-uploaded log files.

Uploaded logs are copied into a session-specific outputs/uploads/ directory so
the pipeline can treat them like normal files and report stable source paths.
Only known log-like extensions are accepted.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


VALID_LOG_EXTENSIONS = {".log", ".txt", ".out", ".err", ".msg"}
UPLOAD_COPY_CHUNK_SIZE = 8 * 1024 * 1024


class UploadedLogFile(Protocol):
    name: str

    def read(self, size: int = -1) -> bytes:
        """
        Read uploaded file bytes.

        Returns:
            File content bytes
        """

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the upload stream cursor."""


def _safe_upload_name(file_name: str) -> str:
    """
    Return a filesystem-safe uploaded filename.

    Args:
        file_name: Original uploaded filename

    Returns:
        Sanitized base filename
    """
    # Browsers normally submit only a filename, but normalize defensively in
    # case an uploaded object contains a client-side path segment.
    normalized = file_name.replace("\\", "/")
    return os.path.basename(normalized).strip()


def save_uploaded_files(uploaded_files: list[UploadedLogFile], upload_dir: str) -> list[str]:
    """
    Save uploaded log files into a clean local directory.

    Args:
        uploaded_files: Streamlit uploaded file objects
        upload_dir: Target upload directory

    Returns:
        Absolute paths for valid saved log files
    """
    target_dir = Path(upload_dir).resolve()
    upload_root = (Path.cwd() / "outputs" / "uploads").resolve()
    if target_dir != upload_root and upload_root not in target_dir.parents:
        raise ValueError(f"Upload directory must be inside {upload_root}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    used_names: set[str] = set()

    for uploaded_file in uploaded_files:
        safe_name = _safe_upload_name(str(uploaded_file.name))
        if not safe_name:
            continue

        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix.lower()
        if suffix not in VALID_LOG_EXTENSIONS:
            continue

        # Preserve the visible filename when possible; add a numeric suffix only
        # when the same file name is uploaded more than once in one session.
        candidate_name = safe_name
        counter = 2
        while candidate_name.lower() in used_names:
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate_name.lower())

        target_path = target_dir / candidate_name
        uploaded_file.seek(0)
        with target_path.open("wb") as output_file:
            while chunk := uploaded_file.read(UPLOAD_COPY_CHUNK_SIZE):
                output_file.write(chunk)
        saved_paths.append(str(target_path))

    return saved_paths
