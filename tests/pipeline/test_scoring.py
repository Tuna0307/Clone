from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

import pipeline.scoring as s


def test_score_anomalies_with_mock_embeddings(mock_embeddings):
    docs = [
        Document(page_content="normal info line", metadata={"source_file": "a.log", "line_ranges": [1]}),
        Document(page_content="ERROR CryptoService failed", metadata={"source_file": "a.log", "line_ranges": [2]}),
    ]
    scored = s.score_anomalies(docs)
    assert len(scored) == len(docs)
    assert all("anomaly_score" in d.metadata for d in scored)
