"""Pydantic v2 models for the structured server monitoring workflow.

This is the canonical, production-grade state definition for Option B
(structured workflow for the DuckDB + agentic SQL path).

Models are:
- Fully validated (construction + assignment)
- 100% serializable (perfect for the .sql_trace.jsonl artifact)
- Designed for both lightweight FSM today and easy LangGraph upgrade tomorrow
- Support archetype-aware classification, onset analysis, and competing hypotheses
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from pipeline.constants import (
    EVIDENCE_GATHERING_MAX_CRITIC_RETRY_LOOPS,
    SERVER_SQL_MAX_RECLASSIFICATIONS,
    SERVER_SQL_MAX_STEPS,
)
from pipeline.server_sql.archetypes import IncidentArchetype

# =============================================================================
# Supporting Models (rich, frozen, provenance-heavy)
# =============================================================================

class LogLineRef(BaseModel):
    """Precise citation to an original log line (for user-facing output and audit)."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime | None = None
    thread: str | None = None
    raw_line: str
    source_file: str | None = None
    line_number: int | None = None


class StructuralSignal(BaseModel):
    """Balanced deterministic pre-screen result from broad diagnostic SQL."""
    model_config = ConfigDict(frozen=True)

    signal_id: str
    signal_family: Literal[
        "log_gap",
        "metric_correlation",
        "endpoint_breadth",
        "high_volume_indicator",
        "runtime_stall_indicator",
        "other",
    ]
    summary: str
    sql_query: str = ""
    observations: list[str] = Field(default_factory=list)
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None


class ArchetypeHypothesis(BaseModel):
    """One scored incident archetype hypothesis."""
    model_config = ConfigDict(frozen=True)

    archetype: IncidentArchetype
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_signals: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None


class ArchetypeClassification(BaseModel):
    """Multi-label archetype classification from broad diagnostic."""
    model_config = ConfigDict()

    primary: ArchetypeHypothesis
    secondary: ArchetypeHypothesis | None = None
    rejected_hypotheses: list[ArchetypeHypothesis] = Field(default_factory=list)
    classification_method: Literal[
        "deterministic_only", "llm_synthesis", "critic_reclassification"
    ] = "llm_synthesis"
    rationale: str = ""


class OnsetRecord(BaseModel):
    """Per-signal onset and symptom/cause classification."""
    model_config = ConfigDict(frozen=True)

    signal_name: str
    onset_time: datetime | None = None
    onset_shape: Literal["abrupt", "gradual", "unknown"] = "unknown"
    role: Literal["likely_cause", "confirmed_effect", "ambiguous"] = "ambiguous"
    evidence: list[str] = Field(default_factory=list)


class OnsetAnalysis(BaseModel):
    """Onset timing and symptom vs cause discrimination."""
    model_config = ConfigDict()

    degradation_start: datetime | None = None
    onset_shape_overall: Literal["abrupt", "gradual", "unknown"] = "unknown"
    signal_records: list[OnsetRecord] = Field(default_factory=list)


class HighSignalEvent(BaseModel):
    """Pre-detected application outlier signal (regex pre-scan or targeted SQL)."""
    model_config = ConfigDict(frozen=True)

    timestamp: datetime | None = None
    signal_type: Literal[
        "high_result_count",
        "extreme_latency",
        "authz_loop_candidate",
        "heavy_repository_op",
        "large_cache_event",
        "slow_ldap_or_db",
        "write_rate_spike",
        "repetition_burst",
        "other",
    ]
    captured_value: int | float | None = Field(
        default=None, description="The large number (Count=6891, lapse=124517, etc.) when present"
    )
    snippet: str = Field(..., min_length=1)
    raw_line: str
    discovery_method: Literal["pre_scan", "phase0_sql", "llm_sql", "deterministic"] = "pre_scan"
    first_onset: bool = Field(default=False, description="True if this is the earliest occurrence of this pattern")


class CriticalWindow(BaseModel):
    """A focused time window containing the key causal activity (onset + peak)."""
    model_config = ConfigDict(frozen=True)

    start_time: datetime
    end_time: datetime
    label: str = Field(..., description="e.g. 'Count=6891 N+1 onset on DATCKPW2'")

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v, info):
        if "start_time" in info.data and v < info.data["start_time"]:
            raise ValueError("end_time must be >= start_time")
        return v


class RedHerringRejection(BaseModel):
    """Explicit, auditable rejection of a non-causal signal (rich provenance for audit + UI)."""
    model_config = ConfigDict(frozen=True, json_encoders={datetime: lambda v: v.isoformat() if v else None})

    signal_description: str = Field(..., description="What was observed (e.g. 'createIndexAllAvailableCredentialTO scheduled job')")
    rejection_category: Literal[
        "cadence_scheduled", "post_onset_symptom", "steady_state", "out_of_scope", "low_impact", "background_polling", "other"
    ]
    rejection_reason: str = Field(..., description="Why this is not causal (tied to ground truth disciplines)")
    evidence: list[str] = Field(default_factory=list, description="Supporting raw excerpts or SQL results")
    confidence: Literal["CERTAIN", "STRONG", "INFERRED", "WEAK"] = "STRONG"


class EvidencePackage(BaseModel):
    """Self-contained, citable hypothesis bundle produced by one phase (richer for synthesis)."""
    model_config = ConfigDict()

    package_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    hypothesis: str
    category: IncidentArchetype | Literal["other"] = "other"

    critical_windows: list[CriticalWindow] = Field(default_factory=list)
    red_herring_rejections: list[RedHerringRejection] = Field(default_factory=list)
    raw_line_refs: list[LogLineRef] = Field(default_factory=list)
    sql_queries: list[str] = Field(default_factory=list)
    metric_snapshots: list[dict[str, Any]] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_cost_claim: str | None = None
    created_in_phase: str = "evidence_gathering"
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class TraceStep(BaseModel):
    """One atomic step in the agent's execution history — the canonical unit for the .sql_trace.jsonl artifact."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() if v else None})

    step: int | str = Field(..., description="-1=init, 'phase0', 0,1,..., 'rescue-3', 'refinement-0'")
    phase: str = Field(..., description="initialize | high_volume_diagnostic | onset_analysis | red_herring_filter | evidence_gathering | critic | report_synthesis | ticket_refinement | finalize | ...")
    node: str | None = Field(default=None, description="FSM / LangGraph node that produced this step")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    llm_output: str | None = None
    sql_blocks: list[str] = Field(default_factory=list, description="Legacy/artifact key name (preferred for writer compat)")
    sql_proposed: list[str] = Field(default_factory=list)  # kept for transition; prefer sql_blocks in new code
    observations: list[str] = Field(default_factory=list)

    critic_verdict: str | None = None
    decision: Literal[
        "continue", "finalize", "retry", "error",
        "skip_no_conn", "strong_partial", "error_fallback", "partial_fallback",
        "finalize_with_ticket"  # emitted by ticket_refinement_node on success path
    ] | None = None

    new_high_volume_signals: list[HighSignalEvent] = Field(default_factory=list)
    new_critical_windows: list[CriticalWindow] = Field(default_factory=list)
    new_red_herring_rejections: list[RedHerringRejection] = Field(default_factory=list)
    evidence_package_updates: list[str] = Field(default_factory=list)

    queries_executed_so_far: int = 0
    phases_completed_so_far: list[str] = Field(default_factory=list)


# =============================================================================
# Main State
# =============================================================================

class ServerMonitoringState(BaseModel):
    """The single source of truth for one server_monitoring file analysis.

    This is the typed state that will flow through the structured workflow
    (lightweight FSM today, LangGraph StateGraph tomorrow).

    Fully validated, serializable, rich provenance. Private runtime resources
    (DuckDB conn, original schema) are excluded from dumps.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda v: v.isoformat() if v else None},
    )

    # Identity & inputs
    file_name: str
    file_path: str
    query_context: dict[str, Any] | None = None
    ticket_text: str | None = Field(default=None, max_length=12000)

    # Serializable runtime detail for LangGraph reliability (file-backed DuckDB path).
    # Enables nodes after initialize_node to re-open the DB after any state reconstruction /
    # model_validate / superstep handoff. Not a PrivateAttr (must survive serialization).
    db_path: str | None = None

    # Control / FSM
    current_phase: str = "initialize"
    phases_completed: set[str] = Field(default_factory=set)
    queries_executed: int = 0
    steps_taken: int = 0
    max_steps: int = Field(default=SERVER_SQL_MAX_STEPS, ge=3)
    evidence_critic_retry_loops: int = 0
    max_evidence_critic_retry_loops: int = Field(
        default=EVIDENCE_GATHERING_MAX_CRITIC_RETRY_LOOPS, ge=0
    )

    # Evidence
    high_volume_signals: list[HighSignalEvent] = Field(default_factory=list)
    structural_signals: list[StructuralSignal] = Field(default_factory=list)
    archetype_classification: ArchetypeClassification | None = None
    onset_analysis: OnsetAnalysis | None = None
    competing_hypotheses: list[ArchetypeHypothesis] = Field(default_factory=list)
    reclassification_count: int = 0
    max_reclassifications: int = Field(default=SERVER_SQL_MAX_RECLASSIFICATIONS, ge=0)
    critical_windows: list[CriticalWindow] = Field(default_factory=list)
    raw_line_samples: list[LogLineRef] = Field(default_factory=list)
    red_herring_rejections: list[RedHerringRejection] = Field(default_factory=list)
    evidence_packages: dict[str, EvidencePackage] = Field(default_factory=dict)

    # Reasoning / conversation (LangChain message compatibility + history)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    report_draft: str | None = None
    critic_feedback_history: list[dict[str, Any]] = Field(default_factory=list)

    # Full provenance (becomes the .sql_trace.jsonl gold artifact)
    trace: list[TraceStep] = Field(default_factory=list)

    # Outputs
    final_findings: str | None = None
    status: Literal["running", "success", "error"] = "running"

    # Real row counts (populated by initialize_node after load; fixes the "0" contract gap)
    metric_row_count: int = 0
    log_event_row_count: int = 0

    # Internal runtime-only (never serialized)
    _duckdb_conn: Any = PrivateAttr(default=None)
    _schema: dict[str, Any] = PrivateAttr(default_factory=dict)

    broad_diagnostic_cache: list[dict] = Field(default_factory=list)

    #To track whether the autonomous summary was generated
    autonomous_summary_generated: bool = Field(default=False)


    # ------------------------------------------------------------------
    # model_post_init + helpers (auto-trace + safe private attr handling)
    # ------------------------------------------------------------------
    def model_post_init(self, __context: Any) -> None:
        if not self.trace:
            self.add_trace_step(
                step=-1,
                phase="initialization",
                node="constructor",
                llm_output=None,
                sql_blocks=[],
                observations=[f"State created for {self.file_name}"],
                decision="continue",
            )

    # ------------------------------------------------------------------
    # Typed mutators (the preferred way to update inside nodes — keeps validation + trace)
    # ------------------------------------------------------------------
    def add_high_volume_signal(self, event: dict | HighSignalEvent | None = None, **kwargs) -> HighSignalEvent:
        if event is None and kwargs:
            event = kwargs
        if isinstance(event, dict):
            if "discovery_method" not in event:
                event["discovery_method"] = "pre_scan"
            event = HighSignalEvent.model_validate(event)
        elif isinstance(event, HighSignalEvent):
            pass
        else:
            if "discovery_method" not in kwargs:
                kwargs["discovery_method"] = "pre_scan"
            event = HighSignalEvent(**kwargs)
        self.high_volume_signals.append(event)
        return event

    def add_critical_window(self, window: dict | CriticalWindow | None = None, **kwargs) -> CriticalWindow:
        if isinstance(window, dict):
            window = CriticalWindow.model_validate(window)
        elif window is None:
            window = CriticalWindow(**kwargs)
        self.critical_windows.append(window)
        return window

    def add_red_herring(self, rej: dict | RedHerringRejection | None = None, **kwargs) -> RedHerringRejection:
        if isinstance(rej, dict):
            rej = RedHerringRejection.model_validate(rej)
        elif rej is None:
            rej = RedHerringRejection(**kwargs)
        self.red_herring_rejections.append(rej)
        return rej

    def add_structural_signal(self, signal: dict | StructuralSignal | None = None, **kwargs) -> StructuralSignal:
        if isinstance(signal, dict):
            signal = StructuralSignal.model_validate(signal)
        elif signal is None:
            signal = StructuralSignal(**kwargs)
        self.structural_signals.append(signal)
        return signal

    def set_archetype_classification(self, classification: dict | ArchetypeClassification) -> ArchetypeClassification:
        if isinstance(classification, dict):
            classification = ArchetypeClassification.model_validate(classification)
        self.archetype_classification = classification
        return classification

    def update_onset_analysis(self, analysis: dict | OnsetAnalysis) -> OnsetAnalysis:
        if isinstance(analysis, dict):
            analysis = OnsetAnalysis.model_validate(analysis)
        self.onset_analysis = analysis
        return analysis

    def upsert_evidence_package(self, package: dict | EvidencePackage) -> EvidencePackage:
        if isinstance(package, dict):
            package = EvidencePackage.model_validate(package)
        self.evidence_packages[package.package_id] = package
        return package

    def add_trace_step(self, **kwargs) -> TraceStep:
        # Back-compat: if caller passes sql_proposed but not sql_blocks, mirror it
        if "sql_proposed" in kwargs and "sql_blocks" not in kwargs:
            kwargs["sql_blocks"] = kwargs["sql_proposed"]
        if "queries_executed_so_far" not in kwargs:
            kwargs["queries_executed_so_far"] = self.queries_executed
        step = TraceStep(**kwargs)
        self.trace.append(step)
        self.queries_executed = max(self.queries_executed, step.queries_executed_so_far)
        self.steps_taken = max(self.steps_taken, len(self.trace))
        return step

    def mark_phase_complete(self, phase: str) -> None:
        self.phases_completed.add(phase)
        # Auto-trace the milestone for auditability
        self.add_trace_step(
            step=f"phase-complete-{phase}",
            phase=phase,
            node="mark_phase_complete",
            decision="continue",
            phases_completed_so_far=sorted(self.phases_completed),
        )

    def to_serializable_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"_duckdb_conn", "_schema"})

    def to_trace_jsonl_lines(self) -> list[str]:
        return [step.model_dump_json() for step in self.trace]

    # Convenience for nodes that need the live connection or original schema
    @property
    def duckdb_conn(self) -> Any:
        return self._duckdb_conn

    @duckdb_conn.setter
    def duckdb_conn(self, conn: Any) -> None:
        object.__setattr__(self, "_duckdb_conn", conn)

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    @schema.setter
    def schema(self, s: dict[str, Any]) -> None:
        object.__setattr__(self, "_schema", s or {})

    @property
    def llm(self) -> Any:
        return getattr(self, "_llm", None)

    @llm.setter
    def llm(self, llm_instance: Any) -> None:
        object.__setattr__(self, "_llm", llm_instance)
