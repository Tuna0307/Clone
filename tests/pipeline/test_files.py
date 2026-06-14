import os
import tempfile

import pipeline.files as f


def test_get_log_files_from_path_with_temp_dir():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "a.log"), "w").close()
        open(os.path.join(d, "b.txt"), "w").close()
        open(os.path.join(d, "c.py"), "w").close()
        result = f.get_log_files_from_path(d)
        assert len(result) == 2
        assert any("a.log" in r for r in result)
        assert any("b.txt" in r for r in result)


def test_format_file_size():
    assert " B" in f.format_file_size(512)
    assert "KB" in f.format_file_size(2048)
    assert "MB" in f.format_file_size(5 * 1024 * 1024)


def test_stream_file_lines(sample_log_file):
    lines = list(f.stream_file_lines(sample_log_file))
    assert len(lines) == 8
    assert "INFO Server started" in lines[0]
