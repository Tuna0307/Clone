# Reference Appendices

> Edit prompts here. Loaded by `pipeline.prompt_loader`.
> Placeholders use `{{snake_case}}`. Use single `{` in log examples.

<a id="section-5-reference-appendices"></a>
## 5. Reference appendices

---
id: reference.duckdb_schema
role: fragment
workflow: reference
---

**DuckDB table schemas (exact column names):**
- `log_events`: `timestamp`, `thread`, `raw_line`, plus **pre-computed categorical flags**:
  `has_latency`, `has_jdbc`, `has_ldap`, `has_hibernate`, `has_connection_wait`,
  `has_staleobject`, `has_sql`, `has_count_rows`, `has_entry_authz`, `has_rest`,
  `has_scheduled`, plus **extracted values** `method_sig`, `latency_ms`, `result_count`,
  `scheduled_op_name`. Use these flags instead of `LIKE '%...%'` on `raw_line` when possible.
  Always use `timestamp` (not `ts`) for time filtering.
- `server_metrics` (raw EAV): `timestamp`, `thread`, `metric_name`, `metric_value`, `category`, `raw_line`.
  Each metric snapshot is stored as multiple rows. Pivot with `CASE WHEN metric_name = '...' THEN metric_value END` or query `server_metrics_wide`.
- `server_metrics_wide` (preferred for time-series): one row per snapshot with `timestamp`, `ts`, `thread_count`, `tomcat_busy_threads`, `tomcat_current_threads`, `dbcp_active_connections`, `dbcp_idle_connections`, `response_time_ms`, `hibernate_sessions`, and related pool columns.

**SQL rules:**
- On `log_events`, always filter with `timestamp` (e.g. `le.timestamp BETWEEN ...`).
- **Prefer pre-computed flag columns** (e.g. `has_latency = TRUE`, `latency_ms DESC`) over
  `raw_line LIKE '%lapse(ms)%'` or `regexp_extract(raw_line, ...)`.
- For pivoted metrics, prefer `FROM server_metrics_wide` instead of inventing wide columns on `server_metrics`.
- `server_metrics_wide.ts` is an alias of `timestamp` for convenience.

---
id: reference.uam5_dictionary
role: fragment
workflow: reference
---

### System Information
- am.serverName (String): Server ID of the particular UAM Instance being monitored
- am.cachedSession (Integer): Session that is cached in memory
- am.auth.responseTime (Double): Average Authentication Response Time (last 1000 samples)
- am.auth.responseTime90th (Double): Authentication Response Time 90th percentile
- jvm.freeMemory (Long): JVM Free Memory in bytes
- jvm.threadCount (Integer): Current number of live threads
- jvm.maxMemory (Long): Maximum JVM memory (-Xmx)
- am.e2eeNonExpiredSessionCache (Integer): Count of E2EE sessions not yet expired
- am.serverTime (Long): Server timestamp of the snapshot (epoch ms)
- eventManager.threadPoolMaxSize (Integer)
- eventManager.threadPoolQueueSize (Integer)
- eventManager.threadPoolMaxQueueSize (Integer)
- eventManager.threadPoolActiveCount (Integer)
- eventManager.threadPoolRejectedCount (Long)
- eventManager.threadPoolRejectedCountInTimeWindow (Long)

### Tomcat
- am.tomcat.connector.name (String): Tomcat Connector Name
- am.tomcat.thread.current.count (Integer): Total threads in pool
- am.tomcat.thread.busy.count (Integer): Busy threads serving requests

### Hibernate
- hibernate.sessionCount (Integer): Active Hibernate sessions
- hibernate.relation2.cache.hitCount / missCount / elementInMemory
- hibernate.baseobject.cache.hitCount / missCount / elementInMemory
- hibernate.attr.cache.hitCount / missCount / elementInMemory

### DBCP Connection Pool
- dbcp.ActiveConnections (Integer)
- dbcp.AllConnections (Integer)
- dbcp.IdleConnections (Integer)

### deliveryManager (Email Gateway)
- deliveryManager.MRQ-EMAIL-GW.threadPoolActiveCount / MaxQueueSize / MaxSize / QueueSize / RejectedCount / RejectedCountInTimeWindow

### deliveryManager (SMS Gateway)
- deliveryManager.MRQ-SMS-GW.threadPoolActiveCount / MaxQueueSize / MaxSize / QueueSize / RejectedCount / RejectedCountInTimeWindow

---
id: reference.archetype_taxonomy
role: fragment
workflow: reference
---

## Incident Archetype Taxonomy (authoritative)

### global_runtime_stall
**Key signals:** Significant gaps in log_events output while server_metrics snapshots continue; High jvm.threadCount with flat or low am.tomcat.thread.busy.count; Rising am.auth.responseTime without a single dominant log signature; Process-wide degradation across unrelated endpoints
**Typical symptoms:** Connection pool saturation appearing after stall onset; Queue buildup and rejected tasks as downstream effects; Elevated response times across many request types
**Common red herrings:** DBCP saturation that begins only after the stall window; Scheduled maintenance jobs at fixed cadence; Steady-state background polling at constant cost
**Investigation focus:** Log output gap analysis and metric-vs-log rate divergence; JVM thread count vs Tomcat busy thread correlation; Response time trends before infrastructure saturation; Breadth of affected signatures in log_events
**Competing archetypes to test:** high_volume_cardinality, thread_pool_pressure, mixed_compound

### high_volume_cardinality
**Key signals:** Large Count/rows/returned values in log_events; Tight bursts of repeated method signatures (N+1 patterns); Single dominant endpoint or operation driving line-rate spikes; Extreme per-operation latencies tied to one operation family
**Typical symptoms:** CPU saturation during burst windows; Elevated Hibernate session counts during the burst; Thread pool pressure as a secondary effect
**Common red herrings:** Hourly scheduled indexing jobs at identical cost; Post-onset pool saturation caused by the burst itself; Large but steady-state cache sizes unrelated to the incident
**Investigation focus:** First onset of large counts and method bursts in log_events; Per-record loop signatures and their timestamps; Correlation between burst windows and metric spikes; Affected users or request identifiers in raw lines
**Competing archetypes to test:** global_runtime_stall, db_connection_pressure, mixed_compound

### thread_pool_pressure
**Key signals:** eventManager.threadPoolQueueSize approaching threadPoolMaxQueueSize; Rising eventManager.threadPoolRejectedCount or Tomcat busy threads; eventManager.threadPoolActiveCount near threadPoolMaxSize; Queue buildup preceding response time increases
**Typical symptoms:** Delivery/event manager backlog; Request latency increases under load; Rejected task counters incrementing
**Common red herrings:** Queue size spikes that occur only after a primary trigger (cardinality or stall); Steady-state queue levels below capacity; Tomcat busy count rises without queue pressure
**Investigation focus:** Tomcat and eventManager pool metrics over time; Queue size vs active count vs rejected count; Onset timing relative to log bursts or runtime gaps; Whether pool pressure precedes or follows other signals
**Competing archetypes to test:** global_runtime_stall, high_volume_cardinality, db_connection_pressure

### db_connection_pressure
**Key signals:** dbcp.ActiveConnections near dbcp.MaxActive; Rising hibernate.sessionCount with long-lived sessions; Slow LDAP/jdbc/SQL operations in log_events; Connection wait or pool exhaustion indicators
**Typical symptoms:** Elevated authentication or persistence latencies; Cache miss spikes under connection contention; Thread blocking while waiting for connections
**Common red herrings:** Pool saturation that is clearly post-onset to a cardinality burst; Steady-state connection counts below max; Scheduled batch jobs using connections at fixed intervals
**Investigation focus:** DBCP active/idle/max metrics over the incident window; Hibernate session count trends; Slow database/LDAP log lines and their onset; Whether connection pressure precedes or follows other archetypes
**Competing archetypes to test:** high_volume_cardinality, global_runtime_stall, thread_pool_pressure

### mixed_compound
**Key signals:** Strong signals from two or more archetypes with overlapping timelines; A primary trigger (e.g. cardinality burst) followed by secondary runtime effects; Onset of one archetype clearly precedes symptoms of another
**Typical symptoms:** High-volume query triggers thread pool and connection exhaustion; Runtime stall coincides with but does not explain cardinality bursts; Layered degradation across JVM, pools, and persistence
**Common red herrings:** Treating downstream pool saturation as the root cause when a burst preceded it; Attributing everything to a single archetype when timelines diverge
**Investigation focus:** Timeline ordering: which archetype signals appear first; Symptom vs cause classification per signal family; Evidence for trigger vs downstream effect chains; Test each constituent archetype hypothesis independently
**Competing archetypes to test:** global_runtime_stall, high_volume_cardinality, thread_pool_pressure, db_connection_pressure

---
id: reference.sql_fence_rules
role: fragment
workflow: reference
---

SQL FORMAT RULES (required for automatic execution):
- Put each query in its own fenced ```sql block.
- Start each block directly with WITH or SELECT. Do NOT prefix blocks with -- or /* comment lines.
- Put brief purpose/labels outside the fence, not inside it.
- Only read-only SELECT/WITH queries are permitted.

---
id: reference.archetype.global_runtime_stall.investigation_focus
role: fragment
workflow: reference
---

- Log output gap analysis and metric-vs-log rate divergence
- JVM thread count vs Tomcat busy thread correlation
- Response time trends before infrastructure saturation
- Breadth of affected signatures in log_events

---
id: reference.archetype.global_runtime_stall.red_herrings
role: fragment
workflow: reference
---

- DBCP saturation that begins only after the stall window
- Scheduled maintenance jobs at fixed cadence
- Steady-state background polling at constant cost

---
id: reference.archetype.high_volume_cardinality.investigation_focus
role: fragment
workflow: reference
---

- First onset of large counts and method bursts in log_events
- Per-record loop signatures and their timestamps
- Correlation between burst windows and metric spikes
- Affected users or request identifiers in raw lines

---
id: reference.archetype.high_volume_cardinality.red_herrings
role: fragment
workflow: reference
---

- Hourly scheduled indexing jobs at identical cost
- Post-onset pool saturation caused by the burst itself
- Large but steady-state cache sizes unrelated to the incident

---
id: reference.archetype.thread_pool_pressure.investigation_focus
role: fragment
workflow: reference
---

- Tomcat and eventManager pool metrics over time
- Queue size vs active count vs rejected count
- Onset timing relative to log bursts or runtime gaps
- Whether pool pressure precedes or follows other signals

---
id: reference.archetype.thread_pool_pressure.red_herrings
role: fragment
workflow: reference
---

- Queue size spikes that occur only after a primary trigger (cardinality or stall)
- Steady-state queue levels below capacity
- Tomcat busy count rises without queue pressure

---
id: reference.archetype.db_connection_pressure.investigation_focus
role: fragment
workflow: reference
---

- DBCP active/idle/max metrics over the incident window
- Hibernate session count trends
- Slow database/LDAP log lines and their onset
- Whether connection pressure precedes or follows other archetypes

---
id: reference.archetype.db_connection_pressure.red_herrings
role: fragment
workflow: reference
---

- Pool saturation that is clearly post-onset to a cardinality burst
- Steady-state connection counts below max
- Scheduled batch jobs using connections at fixed intervals

---
id: reference.archetype.mixed_compound.investigation_focus
role: fragment
workflow: reference
---

- Timeline ordering: which archetype signals appear first
- Symptom vs cause classification per signal family
- Evidence for trigger vs downstream effect chains
- Test each constituent archetype hypothesis independently

---
id: reference.archetype.mixed_compound.red_herrings
role: fragment
workflow: reference
---

- Treating downstream pool saturation as the root cause when a burst preceded it
- Attributing everything to a single archetype when timelines diverge
