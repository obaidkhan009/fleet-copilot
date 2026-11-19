"""Tenant-partitioned telemetry store.

Tenant isolation is enforced *structurally*: the agent and its tools never get
a handle on the whole store, only on a ``TenantView`` that physically contains
one company's snapshots. There is no code path from a ``TenantView`` back to
another tenant's data, so a prompt-injected or confused model cannot reach it
no matter which ids it asks for.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from .models import Snapshot


class UnknownTenantError(KeyError):
    pass


class TenantView:
    """Read-only access to a single tenant's snapshots."""

    __slots__ = ("_tenant_id", "_by_device", "_employees")

    def __init__(self, tenant_id: str, snapshots: list[Snapshot]):
        assert all(s.company_id == tenant_id for s in snapshots), "tenant partition violated"
        by_device: dict[str, list[Snapshot]] = defaultdict(list)
        for s in snapshots:
            by_device[s.device_id].append(s)
        self._tenant_id = tenant_id
        self._by_device = MappingProxyType(
            {d: tuple(sorted(v, key=lambda s: s.collected_at)) for d, v in sorted(by_device.items())}
        )
        self._employees = frozenset(s.employee_id for s in snapshots)

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def device_ids(self) -> list[str]:
        return list(self._by_device)

    def has_device(self, device_id: str) -> bool:
        return device_id in self._by_device

    def has_employee(self, employee_id: str) -> bool:
        return employee_id in self._employees

    def employee_ids(self) -> list[str]:
        return sorted(self._employees)

    def history(self, device_id: str) -> tuple[Snapshot, ...]:
        """All snapshots for a device, oldest first. Raises KeyError if not in tenant."""
        return self._by_device[device_id]

    def latest(self, device_id: str) -> Snapshot:
        return self._by_device[device_id][-1]

    def latest_all(self) -> list[Snapshot]:
        return [h[-1] for h in self._by_device.values()]

    def devices_of(self, employee_id: str) -> list[str]:
        return [d for d, h in self._by_device.items() if h[-1].employee_id == employee_id]

    @property
    def as_of(self) -> datetime:
        return max(h[-1].collected_at for h in self._by_device.values())

    @property
    def window_start(self) -> datetime:
        return min(h[0].collected_at for h in self._by_device.values())


class TelemetryStore:
    def __init__(self, snapshots: list[Snapshot]):
        parts: dict[str, list[Snapshot]] = defaultdict(list)
        for s in snapshots:
            parts[s.company_id].append(s)
        self._views = {t: TenantView(t, v) for t, v in parts.items()}

    @classmethod
    def load(cls, path: str | Path) -> "TelemetryStore":
        snaps = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    snaps.append(Snapshot.from_record(json.loads(line)))
        return cls(snaps)

    def tenant_ids(self) -> list[str]:
        return sorted(self._views)

    def scoped(self, tenant_id: str) -> TenantView:
        try:
            return self._views[tenant_id]
        except KeyError:
            raise UnknownTenantError(tenant_id) from None
