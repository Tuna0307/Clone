import pipeline.chunking as c
from pipeline.constants import _DEFAULT_API_REQUEST_BOUNDARIES
from pipeline.query import load_retrieval_signals


def test_extract_api_request_docs_on_server_log(sample_log_file, sample_schema):
    retrieval_signals = load_retrieval_signals()
    docs = c.extract_api_request_docs_deterministic(
        sample_log_file,
        sample_schema,
        _DEFAULT_API_REQUEST_BOUNDARIES,
        retrieval_signals,
        query_context=None,
    )
    assert isinstance(docs, list)


def test_decode_url_encoded_errors():
    result = c.decode_url_encoded_errors("Error%3A%20timeout")
    assert "Error: timeout" in result
