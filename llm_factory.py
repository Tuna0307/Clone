"""
Provider-agnostic LLM and embeddings factory.

Usage:
    from llm_factory import get_llm, get_embeddings

    llm = get_llm()               # Returns ChatBedrockConverse or ChatOpenAI
    embeddings = get_embeddings()  # Returns BedrockEmbeddings or OpenAIEmbeddings

The provider is selected via the LLM_PROVIDER env var (see config.py).
Provider-specific imports are intentionally kept inside this module so the
pipeline can switch providers without changing analysis code.
"""

from __future__ import annotations

from config import (
    LLM_PROVIDER,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_ENDPOINT_URL,
    EMBEDDING_MODEL_ID,
    AWS_DEFAULT_REGION,
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    OPENAI_ORG_ID,
    OPENAI_REASONING_EFFORT,
)


def get_llm():
    """Return a configured chat LLM for the active provider.

    Returns:
        ChatBedrockConverse (Bedrock) or ChatOpenAI (OpenAI)
    """
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse

        kwargs: dict = {
            "model_id": LLM_MODEL_ID,
            "region_name": AWS_DEFAULT_REGION,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        if LLM_ENDPOINT_URL:
            kwargs["endpoint_url"] = LLM_ENDPOINT_URL
        return ChatBedrockConverse(**kwargs)

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": LLM_MODEL_ID,
            "temperature": LLM_TEMPERATURE,
            "max_completion_tokens": LLM_MAX_TOKENS,
        }
        if OPENAI_API_KEY:
            kwargs["api_key"] = OPENAI_API_KEY
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        if OPENAI_ORG_ID:
            kwargs["organization"] = OPENAI_ORG_ID
        if OPENAI_REASONING_EFFORT:
            kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
        return ChatOpenAI(**kwargs)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. "
            "Supported values: 'bedrock', 'openai'."
        )


def get_embeddings():
    """Return a configured embeddings client for the active provider.

    Returns:
        BedrockEmbeddings (Bedrock) or OpenAIEmbeddings (OpenAI)
    """
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import BedrockEmbeddings

        kwargs: dict = {
            "model_id": EMBEDDING_MODEL_ID,
            "region_name": AWS_DEFAULT_REGION,
        }
        if LLM_ENDPOINT_URL:
            kwargs["endpoint_url"] = LLM_ENDPOINT_URL
        return BedrockEmbeddings(**kwargs)

    elif LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {
            "model": EMBEDDING_MODEL_ID,
        }
        if OPENAI_API_KEY:
            kwargs["api_key"] = OPENAI_API_KEY
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        return OpenAIEmbeddings(**kwargs)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. "
            "Supported values: 'bedrock', 'openai'."
        )
