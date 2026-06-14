"""Helpers for parsing structured JSON from LLM responses in server_sql nodes."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from LLM output."""
    if not text:
        return None

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text.strip()]

    for candidate in candidates:
        start = candidate.find("{")
        if start < 0:
            continue
        depth = 0
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = candidate[start:idx + 1]
                    try:
                        parsed = json.loads(blob)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
    return None