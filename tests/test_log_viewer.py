from pathlib import Path

from log_viewer import build_log_reference_key, parse_line_reference_start, read_log_line_window


def test_parse_line_reference_start_handles_single_and_range():
    assert parse_line_reference_start("line 1152") == 1152
    assert parse_line_reference_start("lines 1152") == 1152
    assert parse_line_reference_start("SystemOut.log, lines 1152-1158") == 1152


def test_read_log_line_window_returns_context_and_marks_target(tmp_path):
    log_path = tmp_path / "sample.log"
    log_path.write_text(
        "\n".join(f"line {index}" for index in range(1, 11)) + "\n",
        encoding="utf-8",
    )

    window = read_log_line_window(str(log_path), target_line=5, context_radius=2)

    assert window.target_line == 5
    assert [(line.line_number, line.is_target) for line in window.lines] == [
        (3, False),
        (4, False),
        (5, True),
        (6, False),
        (7, False),
    ]
    assert [line.text for line in window.lines] == [
        "line 3",
        "line 4",
        "line 5",
        "line 6",
        "line 7",
    ]


def test_build_log_reference_key_is_stable_and_scoped():
    first = build_log_reference_key("C:/logs/a.log", 42, "message_1", 8)
    second = build_log_reference_key("C:/logs/a.log", 42, "message_1", 8)
    other_message = build_log_reference_key("C:/logs/a.log", 42, "message_2", 8)

    assert first == second
    assert first != other_message
    assert first.startswith("view_log_ref_")
