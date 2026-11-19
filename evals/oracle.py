"""Independent ground-truth oracle.

Deliberately re-implemented from the raw NDJSON with plain ``json`` and NO
imports from ``fleet_copilot`` - if the oracle reused the package's own
parsing or detectors, a bug there would be graded as correct. Definitions
mirror the documented policy (README > Grounding strategy) but are written
independently.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "device-telemetry-dataset.ndjson"


@lru_cache(maxsize=1)
def _rows() -> tuple[dict, ...]:
    with open(DATA, encoding="utf-8") as fh:
        return tuple(json.loads(l) for l in fh if l.strip())


def history(tenant: str) -> dict[str, list[dict]]:
    h: dict[str, list[dict]] = defaultdict(list)
    for r in _rows():
        if r["company_id"] == tenant:
            h[r["device_id"]].append(r)
    return {d: sorted(v, key=lambda r: r["collected_at"]) for d, v in h.items()}


def latest(tenant: str) -> dict[str, dict]:
    return {d: v[-1] for d, v in history(tenant).items()}


def tenants() -> list[str]:
    return sorted({r["company_id"] for r in _rows()})


def all_device_ids(tenant: str) -> set[str]:
    return set(history(tenant))


def all_employee_ids(tenant: str) -> set[str]:
    return {r["employee_id"] for r in _rows() if r["company_id"] == tenant}


def _free(r):
    v = r["disk_volumes"][0]
    return v["available_bytes"] / v["size_bytes"]


def _mem(r):
    m = r["memory"]
    return m["used_memory_bytes"] / m["total_memory_bytes"]


def _is_laptop(r):
    return not r["device_identity"]["model_name"].startswith("Mac mini")


def _status(r, check):
    return next((c["status"] for c in r["compliance_results"] if c["check_id"] == check), None)


def _ver(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


# --- point-in-time ----------------------------------------------------------

def disk_low(tenant, pct=15.0):
    return {d for d, r in latest(tenant).items() if _free(r) * 100 < pct}


def mem_high(tenant, pct=85.0):
    return {d for d, r in latest(tenant).items() if _mem(r) * 100 > pct}


def macos_older_than(tenant, major_minor="15"):
    bound = _ver(major_minor)
    out = set()
    for d, r in latest(tenant).items():
        if r["os"]["product_name"] == "macOS":
            v = _ver(r["os"]["product_version"])
            n = max(len(v), len(bound))
            if v + (0,) * (n - len(v)) < bound + (0,) * (n - len(bound)):
                out.add(d)
    return out


def failing(tenant, check=None, severity=None, laptops_only=False):
    out = set()
    for d, r in latest(tenant).items():
        if laptops_only and not _is_laptop(r):
            continue
        for c in r["compliance_results"]:
            if c["status"] == "fail" and (check is None or c["check_id"] == check) and (
                    severity is None or c["severity"] == severity):
                out.add(d)
    return out


def windows_devices(tenant):
    return {d for d, r in latest(tenant).items() if r["os"]["product_name"] == "Windows"}


# --- trends -------------------------------------------------------------------

def ram_constrained(tenant, pct=85.0, share=0.7):
    return {d for d, h in history(tenant).items() if sum(_mem(r) * 100 > pct for r in h) / len(h) >= share}


def storage_constrained(tenant, pct=15.0, share=0.7):
    return {d for d, h in history(tenant).items()
            if sum(_free(r) * 100 < pct for r in h) / len(h) >= share or _free(h[-1]) * 100 < pct}


def battery_eol(tenant, cycle_limit=1000):
    out = set()
    for d, h in history(tenant).items():
        bats = [r["battery"] for r in h if r.get("battery", {}).get("battery_present")]
        if bats and (bats[-1]["condition"] != "Normal" or bats[-1]["cycle_count"] >= cycle_limit):
            out.add(d)
    return out


def compliance_regressions(tenant, check="screen_lock"):
    out = set()
    for d, h in history(tenant).items():
        st = [_status(r, check) for r in h]
        if st[-1] == "fail" and any(a == "pass" and b == "fail" for a, b in zip(st, st[1:])):
            out.add(d)
    return out


RISKY = {"uTorrent", "TeamViewer", "CleanMyMac X"}


def unapproved_software(tenant):
    return {d for d, r in latest(tenant).items() if RISKY & {s["name"] for s in r["installed_software"]}}


def upgradeable(model: str, component: str) -> bool:
    table = {"Dell XPS 15": {"ram", "storage", "battery"}, "ThinkPad X1 Carbon": {"storage", "battery"},
             "MacBook Pro": {"battery"}, "MacBook Air": {"battery"}, "Mac mini": set()}
    return component in table.get(model, set())


def model_of(tenant, device):
    return latest(tenant)[device]["device_identity"]["model_name"]


ORACLES = {
    "disk_low": disk_low, "mem_high": mem_high, "macos_older_than": macos_older_than, "failing": failing,
    "windows_devices": windows_devices, "ram_constrained": ram_constrained,
    "storage_constrained": storage_constrained, "battery_eol": battery_eol,
    "compliance_regressions": compliance_regressions, "unapproved_software": unapproved_software,
}


def resolve(tenant: str, spec) -> set[str]:
    """spec: list of ids | {"oracle": name, "args": {...}} | {"empty": true}"""
    if isinstance(spec, list):
        return set(spec)
    if spec.get("empty"):
        return set()
    return set(ORACLES[spec["oracle"]](tenant, **spec.get("args", {})))
