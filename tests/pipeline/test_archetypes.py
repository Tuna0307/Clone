"""Tests for server_monitoring archetype taxonomy."""

from pipeline.server_sql.archetypes import (
    ALL_ARCHETYPES,
    ARCHETYPE_TAXONOMY,
    format_archetype_taxonomy_for_prompt,
    get_investigation_focus,
)


def test_all_archetypes_have_definitions():
    for archetype in ALL_ARCHETYPES:
        defn = ARCHETYPE_TAXONOMY[archetype]
        assert defn["key_signals"]
        assert defn["typical_symptoms"]
        assert defn["common_red_herrings"]
        assert defn["investigation_focus"]
        assert defn["competing_archetypes"]


def test_format_taxonomy_for_prompt_includes_all_archetypes():
    text = format_archetype_taxonomy_for_prompt()
    for archetype in ALL_ARCHETYPES:
        assert archetype in text


def test_investigation_focus_returns_list():
    focus = get_investigation_focus("global_runtime_stall")
    assert isinstance(focus, list)
    assert len(focus) >= 2