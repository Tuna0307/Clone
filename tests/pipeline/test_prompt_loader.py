"""Tests for centralized prompt loading."""

from __future__ import annotations

from pathlib import Path

from followup.prompts import PROMPT_REGISTRY as FOLLOWUP_PROMPT_REGISTRY
from pipeline.prompt_loader import list_prompt_ids, load_fragment, load_prompt, reload_prompts
from pipeline.prompts_api import PROMPT_REGISTRY as API_PROMPT_REGISTRY
from pipeline.server_sql.prompts import PROMPT_REGISTRY as SERVER_PROMPT_REGISTRY

EXPECTED_REFERENCE_IDS = [
    "reference.duckdb_schema",
    "reference.uam5_dictionary",
    "reference.archetype_taxonomy",
    "reference.sql_fence_rules",
    "reference.archetype.global_runtime_stall.investigation_focus",
    "reference.archetype.global_runtime_stall.red_herrings",
    "reference.archetype.high_volume_cardinality.investigation_focus",
    "reference.archetype.high_volume_cardinality.red_herrings",
    "reference.archetype.thread_pool_pressure.investigation_focus",
    "reference.archetype.thread_pool_pressure.red_herrings",
    "reference.archetype.db_connection_pressure.investigation_focus",
    "reference.archetype.db_connection_pressure.red_herrings",
    "reference.archetype.mixed_compound.investigation_focus",
    "reference.archetype.mixed_compound.red_herrings",
]

PROMPT_REGISTRY = sorted({
    *API_PROMPT_REGISTRY,
    *FOLLOWUP_PROMPT_REGISTRY,
    *SERVER_PROMPT_REGISTRY,
    *EXPECTED_REFERENCE_IDS,
})

PROMPT_SECTION_FILES = [
    "01_api_request.md",
    "02_server_monitoring.md",
    "03_follow_up_chat.md",
    "04_schema_fallback.md",
    "05_reference_appendices.md",
]


def test_all_prompt_section_files_exist():
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    for filename in PROMPT_SECTION_FILES:
        assert (prompts_dir / filename).is_file(), filename
    assert not (prompts_dir / "AGENT_PROMPTS.md").exists()


def test_prompt_registry_has_no_duplicates():
    assert len(PROMPT_REGISTRY) == len(set(PROMPT_REGISTRY))


def test_all_registered_prompt_ids_exist():
    reload_prompts()
    loaded = set(list_prompt_ids())
    missing = [prompt_id for prompt_id in PROMPT_REGISTRY if prompt_id not in loaded]
    assert not missing, f"Missing prompt IDs: {missing}"
    assert len(loaded) == len(PROMPT_REGISTRY)


def test_api_map_substitution():
    reload_prompts()
    guardrail = load_fragment("api_request.map.guardrail")
    system = load_prompt("api_request.map.system", api_map_guardrail_text=guardrail)
    assert "IAM Forensic Evidence Analyst" in system
    assert "deterministic API-request fast-path" in system
    assert "{{api_map_guardrail_text}}" not in system

    user = load_prompt(
        "api_request.map.user",
        file_name="sample.log",
        category="api_request",
        subcategory="api_request",
        evidence_profile_json='{"total_lines": 1}',
        evidence_text="[REF_1] error",
    )
    assert "sample.log" in user
    assert "[REF_1] error" in user


def test_prompt_bodies_exclude_section_navigation_markers():
    reload_prompts()
    toc_section_titles = [
        "## 1. API request (map, reduce)",
        "## 2. Server monitoring (LangGraph + follow-up SQL)",
        "## 3. Follow-up chat (intent, answer)",
        "## 4. Schema fallback",
        "## 5. Reference appendices",
    ]
    for prompt_id in PROMPT_REGISTRY:
        body = load_fragment(prompt_id)
        assert '<a id="section-' not in body, prompt_id
        for title in toc_section_titles:
            assert title not in body, prompt_id


def test_builders_smoke():
    from pipeline.prompts_api import build_api_map_messages, build_reduce_system_prompt

    reload_prompts()
    system, user = build_api_map_messages(
        file_name="f.log",
        category="api_request",
        subcategory="api_request",
        evidence_profile={"total_lines": 10},
        evidence_text="evidence",
    )
    assert system and user
    assert "deterministic API-request extraction path" in build_reduce_system_prompt("api_request")
    assert "UAM server monitoring" in build_reduce_system_prompt("server_monitoring")