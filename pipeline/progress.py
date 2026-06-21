"""Optional UI progress callbacks for Streamlit (CLI runs remain print-only)."""

from __future__ import annotations

import html
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, Iterator

ProgressCallback = Callable[[str], None]

_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "progress_callback",
    default=None,
)


def get_progress_callback() -> ProgressCallback | None:
    """Return the active UI progress callback, if any."""
    return _progress_callback.get()


def emit_ui_progress(message: str) -> None:
    """Emit a user-facing progress line to the optional UI callback."""
    callback = _progress_callback.get()
    if callback is not None:
        callback(message)


@contextmanager
def progress_callback_scope(callback: ProgressCallback | None) -> Iterator[None]:
    """Temporarily register a UI progress callback for the current context."""
    token: Token | None = None
    if callback is not None:
        token = _progress_callback.set(callback)
    try:
        yield
    finally:
        if token is not None:
            _progress_callback.reset(token)


def format_progress_details_block(lines: list[str], *, summary_label: str = "Pipeline progress") -> str:
    """Render accumulated progress lines inside a collapsible HTML details block."""
    if not lines:
        return ""

    label = summary_label
    body = html.escape("\n".join(lines))
    return (
        f'<details class="pipeline-progress-details">'
        f"<summary>{html.escape(label)}</summary>"
        f"<pre>{body}</pre>"
        f"</details>"
    )