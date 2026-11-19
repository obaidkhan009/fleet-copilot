"""Evidence ledger: the backbone of grounding and traceability.

Every row a read tool returns to the model is registered here first and gets a
short id (``E1``, ``E2``...). The model must cite these ids; the verifier
checks that each citation exists and that every device/employee the answer
names is backed by evidence; and the action policy requires proposals to cite
evidence about the *same* device. Each record keeps the tool name, arguments,
and the underlying snapshot refs (``<device_id>@<timestamp>``) so any claim can
be traced back to raw telemetry lines.
"""

from __future__ import annotations

from typing import Any


class EvidenceLedger:
    def __init__(self, records: dict[str, dict] | None = None):
        self.records: dict[str, dict] = dict(records or {})
        self.new_ids: list[str] = []

    def cite(self, *, tool: str, tenant_id: str, claim: str, device_id: str | None = None,
             employee_id: str | None = None, snapshot_refs: list[str] | None = None,
             data: dict[str, Any] | None = None, devices: list[str] | None = None,
             args: dict | None = None) -> str:
        eid = f"E{len(self.records) + 1}"
        self.records[eid] = {
            "id": eid,
            "tool": tool,
            "args": args or {},
            "tenant_id": tenant_id,
            "device_id": device_id,
            "employee_id": employee_id,
            # aggregate evidence (e.g. fleet overview) can cover several devices
            "devices": sorted(set(devices or ([device_id] if device_id else []))),
            "claim": claim,
            "snapshot_refs": list(snapshot_refs or []),
            "data": data or {},
        }
        self.new_ids.append(eid)
        return eid

    def get(self, eid: str) -> dict | None:
        return self.records.get(eid)

    def is_aggregate(self, eid: str) -> bool:
        """Aggregate ("scope") records support counts and "none matched" statements, not per-device facts."""
        r = self.records.get(eid)
        return bool(r) and not r.get("device_id")

    def devices_covered(self) -> set[str]:
        """Devices with at least one device-specific record. Aggregate records list every device in scope,
        so counting them would make any device id look "supported"."""
        return {r["device_id"] for r in self.records.values() if r.get("device_id")}

    def employees_covered(self) -> set[str]:
        return {r["employee_id"] for r in self.records.values() if r.get("employee_id")}
