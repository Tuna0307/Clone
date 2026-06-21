"""Load and cache agent prompts from prompts/0N_*.md section files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_FILES: tuple[Path, ...] = (
    _PROMPTS_DIR / "01_api_request.md",
    _PROMPTS_DIR / "02_server_monitoring.md",
    _PROMPTS_DIR / "03_follow_up_chat.md",
    _PROMPTS_DIR / "04_schema_fallback.md",
    _PROMPTS_DIR / "05_reference_appendices.md",
)
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
_SECTIONS: dict[str, str] | None = None
_SECTION_SOURCES: dict[str, Path] | None = None


_PROMPT_BLOCK_RE = re.compile(
    r"^---\s*\n"
    r"(?P<frontmatter>.*?)"
    r"^---\s*\n"
    r"(?P<body>.*?)"
    r"(?=^---\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SECTION_NAV_RE = re.compile(
    r'\n<a id="section-\d+[^"]*"></a>\n## \d+\.[^\n]*\s*$',
)


def _parse_sections(content: str) -> dict[str, str]:
    """Parse section-delimited markdown with YAML frontmatter per prompt."""
    sections: dict[str, str] = {}
    for match in _PROMPT_BLOCK_RE.finditer(content):
        frontmatter = match.group("frontmatter")
        body = match.group("body").strip()
        prompt_id = None
        for line in frontmatter.splitlines():
            if line.startswith("id:"):
                prompt_id = line.split(":", 1)[1].strip()
                break
        if not prompt_id:
            continue
        body = _SECTION_NAV_RE.sub("", body).rstrip()
        sections[prompt_id] = body
    return sections


def _get_sections() -> dict[str, str]:
    global _SECTIONS, _SECTION_SOURCES
    if _SECTIONS is None:
        merged: dict[str, str] = {}
        sources: dict[str, Path] = {}
        for path in _PROMPT_FILES:
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            for prompt_id, body in _parse_sections(path.read_text(encoding="utf-8")).items():
                if prompt_id in merged:
                    prior = sources[prompt_id].name
                    raise ValueError(
                        f"Duplicate prompt id '{prompt_id}' in {path.name} and {prior}"
                    )
                merged[prompt_id] = body
                sources[prompt_id] = path
        _SECTIONS = merged
        _SECTION_SOURCES = sources
    return _SECTIONS


def list_prompt_ids() -> list[str]:
    """Return all registered prompt section IDs."""
    return sorted(_get_sections().keys())


def load_fragment(prompt_id: str) -> str:
    """Load a static prompt section (no placeholder substitution)."""
    sections = _get_sections()
    if prompt_id not in sections:
        raise KeyError(f"Prompt id not found: {prompt_id}")
    return sections[prompt_id]


def load_prompt(prompt_id: str, **values: str) -> str:
    """Load a prompt section and substitute {{placeholders}}."""
    text = load_fragment(prompt_id)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            logger.warning("Unmatched placeholder %s in prompt %s", key, prompt_id)
            return match.group(0)
        return values[key]

    return _PLACEHOLDER_RE.sub(_replace, text)


def reload_prompts() -> None:
    """Clear the parse cache (for tests)."""
    global _SECTIONS, _SECTION_SOURCES
    _SECTIONS = None
    _SECTION_SOURCES = None