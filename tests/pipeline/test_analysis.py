import pipeline.analysis as a


def test_analyze_single_file_smoke(mock_llm, mock_embeddings, sample_log_file, sample_schema):
    # Extend the fixture schema with the key added by detect_log_structure so
    # analyze_single_file does not KeyError on schema["session_keys"].
    patched_schema = {**sample_schema, "session_keys": []}

    original = a.detect_log_structure
    a.detect_log_structure = lambda p: patched_schema
    try:
        result = a.analyze_single_file(sample_log_file)
        assert "file" in result
        assert "status" in result
    finally:
        a.detect_log_structure = original
