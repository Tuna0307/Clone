"""Tests for optional UI progress callbacks."""

from pipeline.progress import (
    emit_ui_progress,
    format_progress_details_block,
    get_progress_callback,
    progress_callback_scope,
)


def test_emit_ui_progress_noop_without_callback():
    collected: list[str] = []

    class Collector:
        def append(self, value: str) -> None:
            collected.append(value)

    emit_ui_progress("should not appear")
    assert collected == []
    assert get_progress_callback() is None


def test_progress_callback_scope_delivers_messages():
    collected: list[str] = []

    def on_line(message: str) -> None:
        collected.append(message)

    with progress_callback_scope(on_line):
        assert get_progress_callback() is on_line
        emit_ui_progress("line one")
        emit_ui_progress("line two")

    assert collected == ["line one", "line two"]
    assert get_progress_callback() is None


def test_nested_progress_callback_scope_restores_outer_callback():
    outer: list[str] = []
    inner: list[str] = []

    with progress_callback_scope(lambda msg: outer.append(msg)):
        emit_ui_progress("outer-before")
        with progress_callback_scope(lambda msg: inner.append(msg)):
            emit_ui_progress("inner")
        emit_ui_progress("outer-after")

    assert outer == ["outer-before", "outer-after"]
    assert inner == ["inner"]


def test_format_progress_details_block_renders_collapsible_html():
    block = format_progress_details_block(
        ["Sampling lines for schema detection...", "Timestamp detected: True | Thread detected: True"],
        summary_label="Pipeline progress",
    )
    assert 'class="pipeline-progress-details"' in block
    assert "Pipeline progress (2 steps)" in block
    assert "Sampling lines for schema detection..." in block