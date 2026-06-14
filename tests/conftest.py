"""Shared pytest fixtures for the IAM pipeline test suite."""

import hashlib
import importlib
import os
import re
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest


class FakeMessage:
    """Narrow stand-in for an LLM message response."""

    def __init__(self, content: str):
        self.content = content


def _safe_patch_module(monkeypatch, module_name: str, attr_name: str, value):
    """Import *module_name* if it exists and patch *attr_name* on it."""
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return
    monkeypatch.setattr(mod, attr_name, value, raising=False)


@pytest.fixture
def mock_llm(monkeypatch):
    """Return a mock LLM that echoes the last user message (truncated to 200 chars)."""
    fake = MagicMock()

    def _invoke(messages, **kwargs):
        text = "Mock LLM response"
        if messages:
            last = messages[-1]
            if isinstance(last, tuple):
                text = str(last[1])[:200]
            elif hasattr(last, "content"):
                text = str(getattr(last, "content", text))[:200]
        return FakeMessage(content=text)

    fake.invoke = _invoke
    monkeypatch.setattr("llm_factory.get_llm", lambda: fake)

    # Patch production LLM module-level references if the modules exist.
    for mod_name, attr in [
        ("iam_log_intelligence_agent_hybridChunking2", "llm"),
        ("followup_retrieval", "_FOLLOWUP_LLM"),
        ("pipeline.query", "llm"),
        ("pipeline.analysis", "llm"),
        ("pipeline.reporting", "llm"),
        ("followup.intent", "_FOLLOWUP_LLM"),
        ("followup.answer", "_FOLLOWUP_LLM"),
    ]:
        _safe_patch_module(monkeypatch, mod_name, attr, fake)

    return fake


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Return a mock embeddings client that returns deterministic vectors."""
    fake = MagicMock()
    _cache = {}

    def _embed_documents(texts):
        results = []
        for t in texts:
            if t in _cache:
                results.append(_cache[t])
                continue
            # Deterministic per-text vector: seed from hash, then draw once
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
            vec = np.random.default_rng(seed).random(384).astype(np.float32).tolist()
            _cache[t] = vec
            results.append(vec)
        return results

    def _embed_query(text):
        if text in _cache:
            return _cache[text]
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2 ** 32)
        vec = np.random.default_rng(seed).random(384).astype(np.float32).tolist()
        _cache[text] = vec
        return vec

    fake.embed_documents = _embed_documents
    fake.embed_query = _embed_query
    monkeypatch.setattr("llm_factory.get_embeddings", lambda: fake)

    # Patch production embeddings module-level references if the modules exist.
    for mod_name, attr in [
        ("iam_log_intelligence_agent_hybridChunking2", "embeddings"),
        ("followup_retrieval", "_FOLLOWUP_EMBEDDINGS"),
        ("pipeline.query", "embeddings"),
        ("pipeline.analysis", "embeddings"),
        ("pipeline.scoring", "embeddings"),
        ("followup.intent", "_FOLLOWUP_EMBEDDINGS"),
        ("followup.answer", "_FOLLOWUP_EMBEDDINGS"),
    ]:
        _safe_patch_module(monkeypatch, mod_name, attr, fake)

    return fake


@pytest.fixture
def sample_log_file():
    """Yield a temporary log file with realistic server-monitoring lines."""
    lines = [
        "2024-01-15 09:23:45.123 [main] INFO Server started on port 8080",
        "2024-01-15 09:24:01.456 [worker-1] ERROR Connection timeout after 30000ms",
        "2024-01-15 09:24:02.789 [worker-1] ERROR Exception in thread worker-1: java.net.SocketTimeoutException",
        "2024-01-15 09:24:03.001 [worker-1]     at com.example.Server.handleRequest(Server.java:142)",
        "2024-01-15 09:24:03.002 [worker-1]     ... 5 more",
        "2024-01-15 09:25:10.333 [main] INFO Health check OK",
        "2024-01-15 09:26:00.000 [worker-2] WARN High memory usage: 87%",
        "2024-01-15 09:27:00.000 [worker-2] ERROR NullPointerException in CryptoService.decryptValueAsBinary",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.log")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        yield path


@pytest.fixture
def sample_schema():
    """Return a pre-built schema dict for tests."""
    return {
        "timestamp_re": re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"),
        "timestamp_fmt": "%Y-%m-%d %H:%M:%S.%f",
        "thread_re": re.compile(r"\[(\S+)\]"),
        "session_re": None,
        "stack_trace_re": re.compile(r"^\s*(at\s+|\.\.\.\s+\d+\s+more)"),
        "timestamp_group": 1,
        "thread_group": 1,
        "session_group": None,
        "has_timestamp": True,
        "has_thread": True,
        "has_session": False,
        "is_api_request_log": False,
        "is_server_monitoring": True,
    }
