"""Relational repository layer with SQLite default and optional PostgreSQL adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Protocol, Tuple

from aml_engine.models import (
    ADVISORY_NOTICE,
    Alert,
    AuditEvent,
    Case,
    CaseDisposition,
    CaseStatus,
    EvidenceRecord,
    PrioritizationResult,
    ReasonContribution,
    ReviewPriority,
    RunDiagnostics,
)


class CaseRepository(Protocol):
    def save_alert_and_case(self, alert: Alert, case: Case) -> Tuple[bool, Case]:
        """Idempotently persist Alert and Case. Returns (created_new, persisted_case)."""
        ...

    def update_case(self, case: Case) -> None:
        ...

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        ...

    def list_alerts(self, priority: Optional[str] = None) -> List[Alert]:
        ...

    def get_case(self, case_id: str) -> Optional[Case]:
        ...

    def list_cases(self, status: Optional[str] = None) -> List[Case]:
        ...

    def save_run_diagnostics(self, diagnostics: RunDiagnostics) -> None:
        ...

    def list_runs(self) -> List[Dict[str, Any]]:
        ...


def _alert_from_dict(data: Dict[str, Any]) -> Alert:
    p = data["prioritization"]
    ev = data["evidence"]
    prioritization = PrioritizationResult(
        rule_score=float(p["rule_score"]),
        ml_raw_score=float(p["ml_raw_score"]) if p.get("ml_raw_score") is not None else None,
        ml_calibrated_probability=(
            float(p["ml_calibrated_probability"])
            if p.get("ml_calibrated_probability") is not None
            else None
        ),
        combined_score=float(p["combined_score"]),
        priority=ReviewPriority(p["priority"]),
        policy_version=str(p["policy_version"]),
        model_version=p.get("model_version"),
        feature_contributions=tuple(
            ReasonContribution(
                code=c["code"],
                category=c["category"],
                weight=float(c["weight"]),
                description=c["description"],
                evidence_metrics=dict(c.get("evidence_metrics", {})),
            )
            for c in p.get("feature_contributions", [])
        ),
    )
    evidence = EvidenceRecord(
        evidence_id=ev["evidence_id"],
        alert_id=ev["alert_id"],
        config_hash=ev["config_hash"],
        feature_version=ev["feature_version"],
        policy_version=ev["policy_version"],
        model_version=ev.get("model_version"),
        source_tx_hashes=tuple(ev["source_tx_hashes"]),
        involved_tx_ids=tuple(ev["involved_tx_ids"]),
        graph_path=tuple(ev["graph_path"]),
        entity_mapping_warnings=tuple(ev.get("entity_mapping_warnings", [])),
        provenance_hash=ev["provenance_hash"],
    )
    reasons = tuple(
        ReasonContribution(
            code=r["code"],
            category=r["category"],
            weight=float(r["weight"]),
            description=r["description"],
            evidence_metrics=dict(r.get("evidence_metrics", {})),
        )
        for r in data.get("reasons", [])
    )
    return Alert(
        alert_id=data["alert_id"],
        subject_account_id=data["subject_account_id"],
        subject_entity_id=data.get("subject_entity_id"),
        scenarios=tuple(data["scenarios"]),
        trigger_timestamp_iso=data["trigger_timestamp_iso"],
        involved_tx_ids=tuple(data["involved_tx_ids"]),
        prioritization=prioritization,
        reasons=reasons,
        evidence=evidence,
        advisory_notice=data.get("advisory_notice", ADVISORY_NOTICE),
    )


def _case_from_dict(data: Dict[str, Any]) -> Case:
    audit_events = [
        AuditEvent(
            event_id=e["event_id"],
            case_id=e["case_id"],
            sequence_num=int(e["sequence_num"]),
            timestamp_iso=e["timestamp_iso"],
            actor_id=e["actor_id"],
            action=e["action"],
            previous_status=e.get("previous_status"),
            new_status=e["new_status"],
            previous_priority=e.get("previous_priority"),
            new_priority=e["new_priority"],
            is_override=bool(e["is_override"]),
            override_reason_code=e.get("override_reason_code"),
            rationale=e["rationale"],
            prev_hash=e["prev_hash"],
            event_hash=e["event_hash"],
        )
        for e in data.get("audit_trail", [])
    ]
    return Case(
        case_id=data["case_id"],
        alert_id=data["alert_id"],
        subject_account_id=data["subject_account_id"],
        subject_entity_id=data.get("subject_entity_id"),
        status=CaseStatus(data["status"]),
        disposition=CaseDisposition(data["disposition"]),
        system_priority=ReviewPriority(data["system_priority"]),
        current_priority=ReviewPriority(data["current_priority"]),
        created_at_iso=data["created_at_iso"],
        updated_at_iso=data["updated_at_iso"],
        assigned_analyst=data.get("assigned_analyst"),
        override_count=int(data.get("override_count", 0)),
        advisory_notice=data.get("advisory_notice", ADVISORY_NOTICE),
        audit_trail=audit_events,
    )


class SQLiteCaseRepository:
    """Durable SQLite repository supporting idempotent replay and restart recovery."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS monitoring_runs (
                run_id TEXT PRIMARY KEY,
                config_version TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                model_version TEXT,
                diagnostics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                subject_account_id TEXT NOT NULL,
                subject_entity_id TEXT,
                trigger_timestamp_iso TEXT NOT NULL,
                priority TEXT NOT NULL,
                combined_score REAL NOT NULL,
                provenance_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                alert_id TEXT NOT NULL UNIQUE,
                subject_account_id TEXT NOT NULL,
                subject_entity_id TEXT,
                status TEXT NOT NULL,
                disposition TEXT NOT NULL,
                system_priority TEXT NOT NULL,
                current_priority TEXT NOT NULL,
                assigned_analyst TEXT,
                override_count INTEGER NOT NULL DEFAULT 0,
                created_at_iso TEXT NOT NULL,
                updated_at_iso TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_audit_events (
                event_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp_iso TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                action TEXT NOT NULL,
                is_override INTEGER NOT NULL,
                override_reason_code TEXT,
                rationale TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE (case_id, sequence_num)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_alert_and_case(self, alert: Alert, case: Case) -> Tuple[bool, Case]:
        existing = self.get_case(case.case_id)
        if existing is not None:
            return False, existing

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO alerts (
                alert_id, subject_account_id, subject_entity_id,
                trigger_timestamp_iso, priority, combined_score,
                provenance_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.alert_id,
                alert.subject_account_id,
                alert.subject_entity_id,
                alert.trigger_timestamp_iso,
                alert.prioritization.priority.value,
                alert.prioritization.combined_score,
                alert.evidence.provenance_hash,
                json.dumps(alert.to_dict(), sort_keys=True),
            ),
        )
        cur.execute(
            """
            INSERT INTO cases (
                case_id, alert_id, subject_account_id, subject_entity_id,
                status, disposition, system_priority, current_priority,
                assigned_analyst, override_count, created_at_iso, updated_at_iso,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case.case_id,
                case.alert_id,
                case.subject_account_id,
                case.subject_entity_id,
                case.status.value,
                case.disposition.value,
                case.system_priority.value,
                case.current_priority.value,
                case.assigned_analyst,
                case.override_count,
                case.created_at_iso,
                case.updated_at_iso,
                json.dumps(case.to_dict(), sort_keys=True),
            ),
        )
        for ev in case.audit_trail:
            cur.execute(
                """
                INSERT OR IGNORE INTO case_audit_events (
                    event_id, case_id, sequence_num, timestamp_iso,
                    actor_id, action, is_override, override_reason_code,
                    rationale, prev_hash, event_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev.event_id,
                    ev.case_id,
                    ev.sequence_num,
                    ev.timestamp_iso,
                    ev.actor_id,
                    ev.action,
                    1 if ev.is_override else 0,
                    ev.override_reason_code,
                    ev.rationale,
                    ev.prev_hash,
                    ev.event_hash,
                    json.dumps(ev.to_dict(), sort_keys=True),
                ),
            )
        self._conn.commit()
        return True, case

    def update_case(self, case: Case) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE cases
            SET status = ?, disposition = ?, current_priority = ?,
                assigned_analyst = ?, override_count = ?, updated_at_iso = ?,
                payload_json = ?
            WHERE case_id = ?
            """,
            (
                case.status.value,
                case.disposition.value,
                case.current_priority.value,
                case.assigned_analyst,
                case.override_count,
                case.updated_at_iso,
                json.dumps(case.to_dict(), sort_keys=True),
                case.case_id,
            ),
        )
        for ev in case.audit_trail:
            cur.execute(
                """
                INSERT OR IGNORE INTO case_audit_events (
                    event_id, case_id, sequence_num, timestamp_iso,
                    actor_id, action, is_override, override_reason_code,
                    rationale, prev_hash, event_hash, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev.event_id,
                    ev.case_id,
                    ev.sequence_num,
                    ev.timestamp_iso,
                    ev.actor_id,
                    ev.action,
                    1 if ev.is_override else 0,
                    ev.override_reason_code,
                    ev.rationale,
                    ev.prev_hash,
                    ev.event_hash,
                    json.dumps(ev.to_dict(), sort_keys=True),
                ),
            )
        self._conn.commit()

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        row = self._conn.execute(
            "SELECT payload_json FROM alerts WHERE alert_id = ?", (alert_id,)
        ).fetchone()
        if row is None:
            return None
        return _alert_from_dict(json.loads(row["payload_json"]))

    def list_alerts(self, priority: Optional[str] = None) -> List[Alert]:
        if priority:
            rows = self._conn.execute(
                "SELECT payload_json FROM alerts WHERE priority = ? ORDER BY combined_score DESC, alert_id ASC",
                (priority,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM alerts ORDER BY combined_score DESC, alert_id ASC"
            ).fetchall()
        return [_alert_from_dict(json.loads(r["payload_json"])) for r in rows]

    def get_case(self, case_id: str) -> Optional[Case]:
        row = self._conn.execute(
            "SELECT payload_json FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return _case_from_dict(json.loads(row["payload_json"]))

    def list_cases(self, status: Optional[str] = None) -> List[Case]:
        if status:
            rows = self._conn.execute(
                "SELECT payload_json FROM cases WHERE status = ? ORDER BY created_at_iso ASC, case_id ASC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload_json FROM cases ORDER BY created_at_iso ASC, case_id ASC"
            ).fetchall()
        return [_case_from_dict(json.loads(r["payload_json"])) for r in rows]

    def save_run_diagnostics(self, diagnostics: RunDiagnostics) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO monitoring_runs (
                run_id, config_version, config_hash, feature_version,
                policy_version, model_version, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                diagnostics.run_id,
                diagnostics.config_version,
                diagnostics.config_hash,
                diagnostics.feature_version,
                diagnostics.policy_version,
                diagnostics.model_version,
                json.dumps(diagnostics.to_dict(), sort_keys=True),
            ),
        )
        self._conn.commit()

    def list_runs(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT diagnostics_json FROM monitoring_runs ORDER BY rowid ASC"
        ).fetchall()
        return [json.loads(r["diagnostics_json"]) for r in rows]


class PostgresCaseRepository:
    """Optional PostgreSQL adapter implementing the CaseRepository protocol.

    Requires `psycopg` (v3) and a live PostgreSQL DSN. Schema DDL is located in
    `src/aml_engine/storage/schema_postgres.sql`.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Optional dependency 'psycopg' is required for PostgresCaseRepository. "
                "Install with: pip install -r requirements-optional.txt"
            ) from exc
        self._conn = psycopg.connect(dsn, autocommit=False)
        schema_sql = (Path(__file__).resolve().parent / "schema_postgres.sql").read_text(encoding="utf-8")
        with self._conn.cursor() as cur:
            cur.execute(schema_sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_alert_and_case(self, alert: Alert, case: Case) -> Tuple[bool, Case]:
        existing = self.get_case(case.case_id)
        if existing is not None:
            return False, existing
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alerts (
                    alert_id, subject_account_id, subject_entity_id,
                    trigger_timestamp_iso, priority, combined_score,
                    provenance_hash, payload_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (alert_id) DO NOTHING
                """,
                (
                    alert.alert_id,
                    alert.subject_account_id,
                    alert.subject_entity_id,
                    alert.trigger_timestamp_iso,
                    alert.prioritization.priority.value,
                    alert.prioritization.combined_score,
                    alert.evidence.provenance_hash,
                    json.dumps(alert.to_dict(), sort_keys=True),
                ),
            )
            cur.execute(
                """
                INSERT INTO cases (
                    case_id, alert_id, subject_account_id, subject_entity_id,
                    status, disposition, system_priority, current_priority,
                    assigned_analyst, override_count, created_at_iso, updated_at_iso,
                    payload_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    case.case_id,
                    case.alert_id,
                    case.subject_account_id,
                    case.subject_entity_id,
                    case.status.value,
                    case.disposition.value,
                    case.system_priority.value,
                    case.current_priority.value,
                    case.assigned_analyst,
                    case.override_count,
                    case.created_at_iso,
                    case.updated_at_iso,
                    json.dumps(case.to_dict(), sort_keys=True),
                ),
            )
            for ev in case.audit_trail:
                cur.execute(
                    """
                    INSERT INTO case_audit_events (
                        event_id, case_id, sequence_num, timestamp_iso,
                        actor_id, action, is_override, override_reason_code,
                        rationale, prev_hash, event_hash, payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        ev.event_id,
                        ev.case_id,
                        ev.sequence_num,
                        ev.timestamp_iso,
                        ev.actor_id,
                        ev.action,
                        ev.is_override,
                        ev.override_reason_code,
                        ev.rationale,
                        ev.prev_hash,
                        ev.event_hash,
                        json.dumps(ev.to_dict(), sort_keys=True),
                    ),
                )
        self._conn.commit()
        return True, case

    def update_case(self, case: Case) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE cases
                SET status = %s, disposition = %s, current_priority = %s,
                    assigned_analyst = %s, override_count = %s, updated_at_iso = %s,
                    payload_json = %s::jsonb
                WHERE case_id = %s
                """,
                (
                    case.status.value,
                    case.disposition.value,
                    case.current_priority.value,
                    case.assigned_analyst,
                    case.override_count,
                    case.updated_at_iso,
                    json.dumps(case.to_dict(), sort_keys=True),
                    case.case_id,
                ),
            )
        self._conn.commit()

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT payload_json FROM alerts WHERE alert_id = %s", (alert_id,))
            row = cur.fetchone()
        return _alert_from_dict(row[0] if isinstance(row[0], dict) else json.loads(row[0])) if row else None

    def list_alerts(self, priority: Optional[str] = None) -> List[Alert]:
        with self._conn.cursor() as cur:
            if priority:
                cur.execute(
                    "SELECT payload_json FROM alerts WHERE priority = %s ORDER BY combined_score DESC, alert_id ASC",
                    (priority,),
                )
            else:
                cur.execute("SELECT payload_json FROM alerts ORDER BY combined_score DESC, alert_id ASC")
            rows = cur.fetchall()
        return [_alert_from_dict(r[0] if isinstance(r[0], dict) else json.loads(r[0])) for r in rows]

    def get_case(self, case_id: str) -> Optional[Case]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT payload_json FROM cases WHERE case_id = %s", (case_id,))
            row = cur.fetchone()
        return _case_from_dict(row[0] if isinstance(row[0], dict) else json.loads(row[0])) if row else None

    def list_cases(self, status: Optional[str] = None) -> List[Case]:
        with self._conn.cursor() as cur:
            if status:
                cur.execute(
                    "SELECT payload_json FROM cases WHERE status = %s ORDER BY created_at_iso ASC, case_id ASC",
                    (status,),
                )
            else:
                cur.execute("SELECT payload_json FROM cases ORDER BY created_at_iso ASC, case_id ASC")
            rows = cur.fetchall()
        return [_case_from_dict(r[0] if isinstance(r[0], dict) else json.loads(r[0])) for r in rows]

    def save_run_diagnostics(self, diagnostics: RunDiagnostics) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monitoring_runs (
                    run_id, config_version, config_hash, feature_version,
                    policy_version, model_version, diagnostics_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE SET diagnostics_json = EXCLUDED.diagnostics_json
                """,
                (
                    diagnostics.run_id,
                    diagnostics.config_version,
                    diagnostics.config_hash,
                    diagnostics.feature_version,
                    diagnostics.policy_version,
                    diagnostics.model_version,
                    json.dumps(diagnostics.to_dict(), sort_keys=True),
                ),
            )
        self._conn.commit()

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT diagnostics_json FROM monitoring_runs ORDER BY created_at ASC")
            rows = cur.fetchall()
        return [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]
