"""
Central configuration for the IAM Log Intelligence Agent.

Environment-driven settings are loaded once here. Runtime modules should import
configuration values from this file instead of reading environment variables
directly or hardcoding provider/model details.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import boto3
from dotenv import load_dotenv


# Load .env silently when it exists. CI and deployed environments can still
# provide the same variables directly through the process environment.
load_dotenv()

# load_dotenv() keeps blank entries such as OPENAI_API_BASE= in os.environ.
# The OpenAI SDK treats a blank base URL as an explicit, invalid endpoint, so
# remove empty optional OpenAI variables and let the SDK use its defaults.
_OPENAI_ENV_VARS = (
    "OPENAI_API_BASE",
    "OPENAI_ORG_ID",
    "OPENAI_API_KEY",
    "OPENAI_API_TYPE",
    "OPENAI_API_VERSION",
    "OPENAI_PROXY",
)
for _var in _OPENAI_ENV_VARS:
    if _var in os.environ and not os.environ[_var].strip():
        del os.environ[_var]


# Provider selection
# Supported values: "openai", "bedrock".
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()


# AWS credentials (used only when LLM_PROVIDER == "bedrock")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_SERVICE_NAME: str = os.getenv("AWS_SERVICE_NAME", "bedrock-runtime")


# OpenAI credentials (used only when LLM_PROVIDER == "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE: str | None = os.getenv("OPENAI_API_BASE") or None
OPENAI_ORG_ID: str | None = os.getenv("OPENAI_ORG_ID") or None
OPENAI_REASONING_EFFORT: str = os.getenv(
    "OPENAI_REASONING_EFFORT",
    "none" if os.getenv("LLM_MODEL_ID", "gpt-5.5").lower().startswith("gpt-5") else "",
).strip().lower()


# Model configuration. These defaults match .env.example and the current team
# direction; Bedrock users should override both model IDs in .env.
LLM_MODEL_ID: str = os.getenv("LLM_MODEL_ID", "gpt-5.5")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "16384"))
EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")


# Optional provider endpoint override. Leave empty for the default endpoint.
LLM_ENDPOINT_URL: str | None = os.getenv("LLM_ENDPOINT_URL") or None


def _validate() -> None:
    """
    Warn when credentials for the active provider are missing.

    The function exits only for an unknown provider. Missing credentials remain
    a warning so import-time validation does not hide unrelated test failures.
    """
    missing: list[str] = []

    if LLM_PROVIDER == "bedrock":
        if not AWS_ACCESS_KEY_ID:
            missing.append("AWS_ACCESS_KEY_ID")
        if not AWS_SECRET_ACCESS_KEY:
            missing.append("AWS_SECRET_ACCESS_KEY")
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
    else:
        print(
            f"[ERROR] Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Supported values: 'openai', 'bedrock'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if missing:
        print(
            f"[WARNING] Missing environment variable(s) for {LLM_PROVIDER}: "
            f"{', '.join(missing)}. Set them in .env before running the agent.",
            file=sys.stderr,
        )


_validate()


def get_bedrock_client() -> Any:
    """
    Initialize an AWS Bedrock Runtime client from environment configuration.

    Returns:
        boto3 Bedrock Runtime client

    Raises:
        Exception: Re-raises boto3 initialization failures after logging context
    """
    try:
        client_kwargs: dict[str, str] = {
            "service_name": AWS_SERVICE_NAME,
            "region_name": AWS_DEFAULT_REGION,
        }
        if AWS_ACCESS_KEY_ID:
            client_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        if AWS_SECRET_ACCESS_KEY:
            client_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY

        bedrock_runtime = boto3.client(**client_kwargs)
        print("[OK] AWS Bedrock client initialized successfully")
        return bedrock_runtime
    except Exception as exc:
        print(f"[X] Failed to initialize Bedrock client: {exc}")
        raise
