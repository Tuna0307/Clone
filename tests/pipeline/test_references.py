import pipeline.references as r


def test_compact_line_ranges():
    assert r._compact_line_ranges([1, 2, 3, 5, 6]) == "1-3, 5-6"
    assert r._compact_line_ranges([1]) == "1"


def test_line_reference_from_metadata():
    meta = {"source_file": "test.log", "line_ranges": "1-3"}
    ref = r._line_reference_from_metadata(meta)
    assert "lines 1-3" == ref


def test_replace_chunk_refs_with_original_references():
    findings = [
        {
            "source_reference_map": [
                {
                    "ref_id": "REF_001",
                    "source_file": "a.log",
                    "source_path": "/tmp/a.log",
                    "line_reference": "lines 1-2",
                    "vscode_uri": "",
                    "file_uri": "",
                }
            ]
        }
    ]
    text = "See [REF_001] for details."
    result = r._replace_chunk_refs_with_original_references(text, findings)
    assert "REF_001" not in result
    assert "a.log" in result
    assert "Path:" not in result


def test_format_original_reference_hides_path_behind_icon():
    ref = {
        "source_file": "a.log",
        "source_path": "/tmp/a.log",
        "line_reference": "lines 1-2",
        "vscode_uri": "vscode://file//tmp/a.log:1",
        "file_uri": "file:////tmp/a.log",
    }

    result = r._format_original_reference_markdown(ref)

    assert result == [
        "- Original Log Reference: [a.log, lines 1-2](vscode://file//tmp/a.log:1) [📄](file:////tmp/a.log)"
    ]
