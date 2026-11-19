"""Normalised, immutable view of one telemetry snapshot.

Raw NDJSON records are parsed once into ``Snapshot`` objects with derived
metrics (percentages, OS version keys, device type). Every snapshot has a
stable ``ref`` (``<device_id>@<collected_at>``) which is the atomic unit of
evidence that answers cite back to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

DESKTOP_MODELS = ("Mac mini", "iMac", "Mac Studio", "Mac Pro")


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def os_version_key(product_name: str, version: str) -> tuple[int, ...]:
    """Comparable key for an OS version within one OS family.

    macOS "14.5" -> (14, 5); Windows "11 23H2" -> (11, 23, 2).
    """
    if product_name == "Windows":
        m = re.match(r"(\d+)\s+(\d+)H(\d)", version)
        if m:
            return tuple(int(x) for x in m.groups())
    return tuple(int(x) for x in re.findall(r"\d+", version))


@dataclass(frozen=True)
class Battery:
    condition: str
    cycle_count: int
    full_charge_capacity: int
    percentage: int
    charging_status: str


@dataclass(frozen=True)
class Snapshot:
    device_id: str
    company_id: str
    employee_id: str
    collected_at: datetime
    hostname: str
    os_family: str          # "macOS" | "Windows"
    os_version: str         # e.g. "14.5", "11 23H2"
    os_key: tuple[int, ...]
    model_name: str
    processor: str
    device_type: str        # "laptop" | "desktop"
    ram_bytes: int
    mem_used_pct: float
    disk_size_bytes: int
    disk_free_bytes: int
    disk_free_pct: float
    disk_encrypted: bool
    battery: Battery | None  # None when the block is absent from the snapshot
    compliance: dict[str, tuple[str, str]]  # check_id -> (status, severity)
    software: tuple[str, ...]
    network_reported: bool
    raw: dict[str, Any] = field(repr=False, compare=False, hash=False)

    @property
    def ref(self) -> str:
        return f"{self.device_id}@{self.collected_at.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    @property
    def ram_gb(self) -> int:
        return round(self.ram_bytes / 2**30)

    @property
    def disk_size_gb(self) -> int:
        return round(self.disk_size_bytes / 2**30)

    @property
    def failing_checks(self) -> list[tuple[str, str]]:
        return sorted((cid, sev) for cid, (st, sev) in self.compliance.items() if st == "fail")

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> "Snapshot":
        os_ = r["os"]
        ident = r["device_identity"]
        mem = r["memory"]
        vols = r.get("disk_volumes") or []
        root = next((v for v in vols if v.get("mount_point") in ("/", "C:\\", "C:")), vols[0] if vols else None)
        size = int(root["size_bytes"]) if root else 0
        avail = int(root["available_bytes"]) if root else 0
        b = r.get("battery")
        battery = None
        if b and b.get("battery_present"):
            battery = Battery(
                condition=b.get("condition", "Unknown"),
                cycle_count=int(b.get("cycle_count", 0)),
                full_charge_capacity=int(b.get("full_charge_capacity", 0)),
                percentage=int(b.get("percentage", 0)),
                charging_status=b.get("charging_status", "unknown"),
            )
        model = ident["model_name"]
        total_mem = mem.get("total_memory_bytes") or mem.get("ram_bytes") or 1
        return cls(
            device_id=r["device_id"],
            company_id=r["company_id"],
            employee_id=r["employee_id"],
            collected_at=parse_ts(r["collected_at"]),
            hostname=os_.get("hostname", ""),
            os_family=os_["product_name"],
            os_version=os_["product_version"],
            os_key=os_version_key(os_["product_name"], os_["product_version"]),
            model_name=model,
            processor=ident.get("processor", ""),
            device_type="desktop" if model.startswith(DESKTOP_MODELS) else "laptop",
            ram_bytes=int(mem.get("ram_bytes") or total_mem),
            mem_used_pct=round(100.0 * mem["used_memory_bytes"] / total_mem, 2),
            disk_size_bytes=size,
            disk_free_bytes=avail,
            disk_free_pct=round(100.0 * avail / size, 2) if size else 0.0,
            disk_encrypted=bool(root and root.get("encrypted")),
            battery=battery,
            compliance={c["check_id"]: (c["status"], c["severity"]) for c in r.get("compliance_results", [])},
            software=tuple(s["name"] for s in r.get("installed_software", [])),
            network_reported="network" in r,
            raw=r,
        )
