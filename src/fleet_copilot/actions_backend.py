"""Mock downstream systems (ticketing, procurement, notifications).

Approved actions are written to an outbox (JSONL). Each action has an
idempotency key derived from (tenant, tool, canonical args) so re-running an
approved plan, or a LangGraph resume replaying a node, never double-submits.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def idempotency_key(tenant_id: str, tool: str, args: dict) -> str:
    material = json.dumps({"t": tenant_id, "tool": tool, "args": args}, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


class ActionBackend:
    PREFIX = {
        "create_upgrade_order": "PO",
        "open_remediation_ticket": "TKT",
        "flag_device_for_replacement": "RPL",
        "notify_employee": "MSG",
    }

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with open(self.path, encoding="utf-8") as fh:
                    self._records = [json.loads(l) for l in fh if l.strip()]

    @property
    def records(self) -> list[dict]:
        return list(self._records)

    def find(self, tenant_id: str, tool: str, **match) -> list[dict]:
        return [r for r in self._records if r["tenant_id"] == tenant_id and r["tool"] == tool
                and all(r["args"].get(k) == v for k, v in match.items())]

    def submit(self, tenant_id: str, tool: str, args: dict, approved_by: str, proposal_id: str) -> dict:
        key = idempotency_key(tenant_id, tool, args)
        with self._lock:
            for r in self._records:
                if r["idempotency_key"] == key:
                    return {**r, "duplicate": True}
            rec = {
                "external_id": f"{self.PREFIX.get(tool, 'ACT')}-{key[:8].upper()}",
                "idempotency_key": key,
                "tenant_id": tenant_id,
                "tool": tool,
                "args": args,
                "approved_by": approved_by,
                "proposal_id": proposal_id,
                "status": "submitted",
            }
            self._records.append(rec)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
            return {**rec, "duplicate": False}
