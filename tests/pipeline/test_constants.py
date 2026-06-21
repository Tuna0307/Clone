# tests/pipeline/test_constants.py
import pipeline.constants as c


def test_map_evidence_budget_chars():
    assert c.MAP_EVIDENCE_BUDGET_CHARS == 800_000


def test_max_log_file_size_matches_streamlit_5gb_upload_limit():
    assert c.MAX_LOG_FILE_SIZE_BYTES == 5 * 1024 * 1024 * 1024


def test_default_iam_critical_keywords_exist():
    assert "CryptoService" in c._DEFAULT_IAM_CRITICAL_KEYWORDS
    assert "WrapAEK" in c._DEFAULT_IAM_CRITICAL_KEYWORDS


def test_default_error_keywords_exist():
    assert "ERROR" in c._DEFAULT_ERROR_KEYWORDS
    assert "Exception" in c._DEFAULT_ERROR_KEYWORDS


def test_default_noise_patterns_compile():
    import re
    for pat in c._DEFAULT_NOISE_PATTERNS:
        re.compile(pat, re.IGNORECASE)


def test_query_datetime_formats_nonempty():
    assert len(c._QUERY_DATETIME_FORMATS) > 0


