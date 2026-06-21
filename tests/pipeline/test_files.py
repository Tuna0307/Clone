import os
import tempfile

import pipeline.files as pf


def test_get_log_files_from_path_with_temp_dir():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.log"), "w").close()
        open(os.path.join(d, "b.txt"), "w").close()
        open(os.path.join(d, "c.py"), "w").close()
        result = pf.get_log_files_from_path(d)
        assert len(result) == 2
        assert any("a.log" in r for r in result)
        assert any("b.txt" in r for r in result)


def test_format_file_size():
    assert " B" in pf.format_file_size(512)
    assert "KB" in pf.format_file_size(2048)
    assert "MB" in pf.format_file_size(5 * 1024 * 1024)


def test_stream_file_lines(sample_log_file):
    lines = list(pf.stream_file_lines(sample_log_file))
    assert len(lines) == 8
    assert "INFO Server started" in lines[0]


def test_stream_file_lines_respects_byte_bounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.log")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("line1\nline2\nline3\nline4\nline5\n")

        # Start at byte offset of "line3\n" (after "line1\nline2\n" = 12 bytes)
        lines = list(pf.stream_file_lines(path, start_offset=12, end_offset=18))
        assert lines == ["line3\n"], f"Got {lines}"

        # Start at beginning, end before last line
        lines2 = list(pf.stream_file_lines(path, start_offset=0, end_offset=18))
        assert lines2 == ["line1\n", "line2\n", "line3\n"], f"Got {lines2}"
