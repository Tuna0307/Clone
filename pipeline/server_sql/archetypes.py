"""Authoritative incident archetype taxonomy for the server_monitoring workflow.

All phases reference this module to prevent prompt drift and inconsistent handling.
"""

from __future__ import annotations

from typing import Literal, TypedDict

IncidentArchetype = Literal[
    "global_runtime_stall",
    "high_volume_cardinality",
    "thread_pool_pressure",
    "db_connection_pressure",
    "mixed_compound",
]

ALL_ARCHETYPES: tuple[IncidentArchetype, ...] = (
    "global_runtime_stall",
    "high_volume_cardinality",
    "thread_pool_pressure",
    "db_connection_pressure",
    "mixed_compound",
)


class ArchetypeDefinition(TypedDict):
    key_signals: list[str]
    typical_symptoms: list[str]
    common_red_herrings: list[str]
    investigation_focus: list[str]
    competing_archetypes: list[IncidentArchetype]


ARCHETYPE_TAXONOMY: dict[IncidentArchetype, ArchetypeDefinition] = {
    "global_runtime_stall": {
        "key_signals": [
            "Significant gaps in log_events output while server_metrics snapshots continue",
            "High jvm.threadCount with flat or low am.tomcat.thread.busy.count",
            "Rising am.auth.responseTime without a single dominant log signature",
            "Process-wide degradation across unrelated endpoints",
        ],
        "typical_symptoms": [
            "Connection pool saturation appearing after stall onset",
            "Queue buildup and rejected tasks as downstream effects",
            "Elevated response times across many request types",
        ],
        "common_red_herrings": [
            "DBCP saturation that begins only after the stall window",
            "Scheduled maintenance jobs at fixed cadence",
            "Steady-state background polling at constant cost",
        ],
        "investigation_focus": [
            "Log output gap analysis and metric-vs-log rate divergence",
            "JVM thread count vs Tomcat busy thread correlation",
            "Response time trends before infrastructure saturation",
            "Breadth of affected signatures in log_events",
        ],
        "competing_archetypes": [
            "high_volume_cardinality",
            "thread_pool_pressure",
            "mixed_compound",
        ],
    },
    "high_volume_cardinality": {
        "key_signals": [
            "Large Count/rows/returned values in log_events",
            "Tight bursts of repeated method signatures (N+1 patterns)",
            "Single dominant endpoint or operation driving line-rate spikes",
            "Extreme per-operation latencies tied to one operation family",
        ],
        "typical_symptoms": [
            "CPU saturation during burst windows",
            "Elevated Hibernate session counts during the burst",
            "Thread pool pressure as a secondary effect",
        ],
        "common_red_herrings": [
            "Hourly scheduled indexing jobs at identical cost",
            "Post-onset pool saturation caused by the burst itself",
            "Large but steady-state cache sizes unrelated to the incident",
        ],
        "investigation_focus": [
            "First onset of large counts and method bursts in log_events",
            "Per-record loop signatures and their timestamps",
            "Correlation between burst windows and metric spikes",
            "Affected users or request identifiers in raw lines",
        ],
        "competing_archetypes": [
            "global_runtime_stall",
            "db_connection_pressure",
            "mixed_compound",
        ],
    },
    "thread_pool_pressure": {
        "key_signals": [
            "eventManager.threadPoolQueueSize approaching threadPoolMaxQueueSize",
            "Rising eventManager.threadPoolRejectedCount or Tomcat busy threads",
            "eventManager.threadPoolActiveCount near threadPoolMaxSize",
            "Queue buildup preceding response time increases",
        ],
        "typical_symptoms": [
            "Delivery/event manager backlog",
            "Request latency increases under load",
            "Rejected task counters incrementing",
        ],
        "common_red_herrings": [
            "Queue size spikes that occur only after a primary trigger (cardinality or stall)",
            "Steady-state queue levels below capacity",
            "Tomcat busy count rises without queue pressure",
        ],
        "investigation_focus": [
            "Tomcat and eventManager pool metrics over time",
            "Queue size vs active count vs rejected count",
            "Onset timing relative to log bursts or runtime gaps",
            "Whether pool pressure precedes or follows other signals",
        ],
        "competing_archetypes": [
            "global_runtime_stall",
            "high_volume_cardinality",
            "db_connection_pressure",
        ],
    },
    "db_connection_pressure": {
        "key_signals": [
            "dbcp.ActiveConnections near dbcp.MaxActive",
            "Rising hibernate.sessionCount with long-lived sessions",
            "Slow LDAP/jdbc/SQL operations in log_events",
            "Connection wait or pool exhaustion indicators",
        ],
        "typical_symptoms": [
            "Elevated authentication or persistence latencies",
            "Cache miss spikes under connection contention",
            "Thread blocking while waiting for connections",
        ],
        "common_red_herrings": [
            "Pool saturation that is clearly post-onset to a cardinality burst",
            "Steady-state connection counts below max",
            "Scheduled batch jobs using connections at fixed intervals",
        ],
        "investigation_focus": [
            "DBCP active/idle/max metrics over the incident window",
            "Hibernate session count trends",
            "Slow database/LDAP log lines and their onset",
            "Whether connection pressure precedes or follows other archetypes",
        ],
        "competing_archetypes": [
            "high_volume_cardinality",
            "global_runtime_stall",
            "thread_pool_pressure",
        ],
    },
    "mixed_compound": {
        "key_signals": [
            "Strong signals from two or more archetypes with overlapping timelines",
            "A primary trigger (e.g. cardinality burst) followed by secondary runtime effects",
            "Onset of one archetype clearly precedes symptoms of another",
        ],
        "typical_symptoms": [
            "High-volume query triggers thread pool and connection exhaustion",
            "Runtime stall coincides with but does not explain cardinality bursts",
            "Layered degradation across JVM, pools, and persistence",
        ],
        "common_red_herrings": [
            "Treating downstream pool saturation as the root cause when a burst preceded it",
            "Attributing everything to a single archetype when timelines diverge",
        ],
        "investigation_focus": [
            "Timeline ordering: which archetype signals appear first",
            "Symptom vs cause classification per signal family",
            "Evidence for trigger vs downstream effect chains",
            "Test each constituent archetype hypothesis independently",
        ],
        "competing_archetypes": [
            "global_runtime_stall",
            "high_volume_cardinality",
            "thread_pool_pressure",
            "db_connection_pressure",
        ],
    },
}


def format_archetype_taxonomy_for_prompt() -> str:
    """Render the taxonomy as markdown for injection into LLM prompts."""
    lines = ["## Incident Archetype Taxonomy (authoritative)\n"]
    for archetype in ALL_ARCHETYPES:
        defn = ARCHETYPE_TAXONOMY[archetype]
        lines.append(f"### {archetype}")
        lines.append("**Key signals:** " + "; ".join(defn["key_signals"]))
        lines.append("**Typical symptoms:** " + "; ".join(defn["typical_symptoms"]))
        lines.append("**Common red herrings:** " + "; ".join(defn["common_red_herrings"]))
        lines.append("**Investigation focus:** " + "; ".join(defn["investigation_focus"]))
        lines.append("**Competing archetypes to test:** " + ", ".join(defn["competing_archetypes"]))
        lines.append("")
    return "\n".join(lines)


def get_investigation_focus(archetype: IncidentArchetype) -> list[str]:
    return list(ARCHETYPE_TAXONOMY[archetype]["investigation_focus"])


def get_common_red_herrings(archetype: IncidentArchetype) -> list[str]:
    return list(ARCHETYPE_TAXONOMY[archetype]["common_red_herrings"])