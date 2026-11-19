"""Deterministic insight & trend detectors over a tenant's time series.

Design choice: trend detection is done in code, not by the LLM. Asking a model
to eyeball 750 snapshots for "consistently constrained" is slow, expensive and
unverifiable. Instead each detector applies an explicit, documented rule and
returns ``Finding`` objects that carry (a) the metrics that triggered the rule
and (b) the exact snapshot refs that prove it. The LLM's job is to choose which
detectors to run, interpret and prioritise the findings, and recommend actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean

from ..config import HARDWARE_SERVICEABILITY, RISKY_SOFTWARE, Thresholds
from ..data.models import Snapshot
from ..data.store import TenantView

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


@dataclass
class Finding:
    kind: str
    device_id: str
    employee_id: str
    severity: str
    title: str
    explanation: str
    metrics: dict = field(default_factory=dict)
    snapshot_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _days(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def slope_per_day(points: list[tuple[datetime, float]]) -> float:
    """Ordinary least-squares slope (units per day)."""
    if len(points) < 2:
        return 0.0
    t0 = points[0][0]
    xs = [_days(t0, t) for t, _ in points]
    ys = [v for _, v in points]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _p95(vals: list[float]) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def _date(s: Snapshot) -> str:
    return s.collected_at.strftime("%Y-%m-%d")


def serviceable(model_name: str, component: str) -> bool | None:
    prof = HARDWARE_SERVICEABILITY.get(model_name)
    return None if prof is None else prof.get(component)


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

def storage_constrained(view: TenantView, th: Thresholds) -> list[Finding]:
    out = []
    for d in view.device_ids():
        h = view.history(d)
        latest = h[-1]
        low = [s for s in h if s.disk_free_pct < th.disk_low_free_pct]
        frac = len(low) / len(h)
        sustained = frac >= th.sustained_fraction
        emerging = not sustained and latest.disk_free_pct < th.disk_low_free_pct
        if not (sustained or emerging):
            continue
        worst = min(h, key=lambda s: s.disk_free_pct)
        slope = slope_per_day([(s.collected_at, s.disk_free_pct) for s in h])
        if latest.disk_free_pct < th.disk_critical_free_pct:
            sev = "critical" if sustained else "high"
        else:
            sev = "high" if sustained else "medium"
        upg = serviceable(latest.model_name, "storage")
        out.append(Finding(
            kind="storage_constrained",
            device_id=d,
            employee_id=latest.employee_id,
            severity=sev,
            title=("Consistently low on storage" if sustained else "Recently low on storage")
            + f": {latest.disk_free_pct:.1f}% free",
            explanation=(
                f"Free disk space was below {th.disk_low_free_pct:.0f}% in {len(low)}/{len(h)} snapshots "
                f"({_pct(len(low), len(h))}%) between {_date(h[0])} and {_date(latest)}; minimum "
                f"{worst.disk_free_pct:.1f}% on {_date(worst)}, latest {latest.disk_free_pct:.1f}% "
                f"({latest.disk_free_bytes / 2**30:.0f} GB free of {latest.disk_size_gb} GB). "
                f"Storage on a {latest.model_name} is "
                + ("field-upgradeable." if upg else "not field-upgradeable (soldered)." if upg is False else "of unknown serviceability.")
            ),
            metrics={
                "latest_free_pct": latest.disk_free_pct,
                "min_free_pct": worst.disk_free_pct,
                "mean_free_pct": round(mean(s.disk_free_pct for s in h), 2),
                "share_of_snapshots_below_threshold_pct": _pct(len(low), len(h)),
                "snapshots": len(h),
                "trend_pct_points_per_day": round(slope, 3),
                "disk_size_gb": latest.disk_size_gb,
                "storage_upgradeable": upg,
                "sustained": sustained,
            },
            snapshot_refs=list(dict.fromkeys([h[0].ref, worst.ref, latest.ref])),
        ))
    return out


def ram_constrained(view: TenantView, th: Thresholds) -> list[Finding]:
    out = []
    for d in view.device_ids():
        h = view.history(d)
        latest = h[-1]
        vals = [s.mem_used_pct for s in h]
        high = [s for s in h if s.mem_used_pct > th.ram_high_used_pct]
        frac = len(high) / len(h)
        if frac < th.sustained_fraction:
            continue
        peak = max(h, key=lambda s: s.mem_used_pct)
        upg = serviceable(latest.model_name, "ram")
        out.append(Finding(
            kind="ram_constrained",
            device_id=d,
            employee_id=latest.employee_id,
            severity="high" if frac >= 0.9 else "medium",
            title=f"Consistently RAM constrained: {mean(vals):.0f}% average memory use on {latest.ram_gb} GB",
            explanation=(
                f"Memory utilisation exceeded {th.ram_high_used_pct:.0f}% in {len(high)}/{len(h)} snapshots "
                f"({_pct(len(high), len(h))}%) between {_date(h[0])} and {_date(latest)}; mean {mean(vals):.1f}%, "
                f"p95 {_p95(vals):.1f}%, peak {peak.mem_used_pct:.1f}% on {_date(peak)}. "
                f"RAM on a {latest.model_name} is "
                + ("upgradeable." if upg else "soldered and cannot be upgraded." if upg is False else "of unknown serviceability.")
            ),
            metrics={
                "ram_gb": latest.ram_gb,
                "mean_used_pct": round(mean(vals), 2),
                "p95_used_pct": round(_p95(vals), 2),
                "max_used_pct": peak.mem_used_pct,
                "share_of_snapshots_above_threshold_pct": _pct(len(high), len(h)),
                "snapshots": len(h),
                "ram_upgradeable": upg,
            },
            snapshot_refs=list(dict.fromkeys([h[0].ref, peak.ref, latest.ref])),
        ))
    return out


def battery_end_of_life(view: TenantView, th: Thresholds) -> list[Finding]:
    out = []
    for d in view.device_ids():
        h = view.history(d)
        with_bat = [s for s in h if s.battery is not None]
        if len(with_bat) < 2:
            continue
        first, latest = with_bat[0], with_bat[-1]
        b0, b1 = first.battery, latest.battery
        span = max(_days(first.collected_at, latest.collected_at), 1e-9)
        cycles_per_day = (b1.cycle_count - b0.cycle_count) / span
        cap_drop = 100.0 * (b0.full_charge_capacity - b1.full_charge_capacity) / b0.full_charge_capacity
        degraded_since = next((s for s in with_bat if s.battery.condition != "Normal"), None)
        reasons = []
        if b1.condition != "Normal":
            reasons.append(f"OS reports battery condition '{b1.condition}' since {_date(degraded_since)}")
        if b1.cycle_count >= th.battery_cycle_limit:
            reasons.append(f"cycle count {b1.cycle_count} exceeds the {th.battery_cycle_limit}-cycle rated life")
        days_to_limit = None
        if b1.cycle_count < th.battery_cycle_limit and cycles_per_day > 0:
            days_to_limit = round((th.battery_cycle_limit - b1.cycle_count) / cycles_per_day)
            if days_to_limit <= th.battery_eol_horizon_days:
                reasons.append(
                    f"at {cycles_per_day:.2f} cycles/day it reaches {th.battery_cycle_limit} cycles in ~{days_to_limit} days"
                )
        if cap_drop >= th.battery_capacity_drop_pct:
            reasons.append(
                f"full-charge capacity fell {cap_drop:.1f}% ({b0.full_charge_capacity} -> {b1.full_charge_capacity} mAh) in {span:.0f} days"
            )
        if not reasons:
            continue
        eol = b1.condition != "Normal" or b1.cycle_count >= th.battery_cycle_limit
        sev = "critical" if (b1.condition != "Normal" and b1.cycle_count >= th.battery_cycle_limit) else (
            "high" if eol else "medium")
        refs = [first.ref] + ([degraded_since.ref] if degraded_since else []) + [latest.ref]
        out.append(Finding(
            kind="battery_end_of_life",
            device_id=d,
            employee_id=latest.employee_id,
            severity=sev,
            title=("Battery at end of life" if eol else "Battery approaching end of life")
            + f": {b1.cycle_count} cycles, condition '{b1.condition}'",
            explanation="; ".join(reasons) + ".",
            metrics={
                "status": "end_of_life" if eol else "approaching_end_of_life",
                "cycle_count": b1.cycle_count,
                "cycle_count_window_start": b0.cycle_count,
                "cycles_per_day": round(cycles_per_day, 3),
                "projected_days_to_cycle_limit": days_to_limit,
                "condition": b1.condition,
                "degraded_since": _date(degraded_since) if degraded_since else None,
                "full_charge_capacity_mah": b1.full_charge_capacity,
                "capacity_drop_pct_in_window": round(cap_drop, 2),
                "battery_serviceable": serviceable(latest.model_name, "battery"),
            },
            snapshot_refs=list(dict.fromkeys(refs)),
        ))
    return out


def compliance_drift(view: TenantView, th: Thresholds) -> list[Finding]:
    """Per-device pass->fail regressions (and fail->pass remediations) per check."""
    out = []
    for d in view.device_ids():
        h = view.history(d)
        latest = h[-1]
        checks = sorted({c for s in h for c in s.compliance})
        for cid in checks:
            seq = [(s, s.compliance.get(cid, (None, None))) for s in h]
            sev = next((v[1] for _, v in seq if v[1]), "unknown")
            transitions = []
            for (sa, (st_a, _)), (sb, (st_b, _)) in zip(seq, seq[1:]):
                if st_a and st_b and st_a != st_b:
                    transitions.append((sa, sb, st_a, st_b))
            if not transitions:
                continue
            last_a, last_b, frm, to = transitions[-1]
            current = latest.compliance.get(cid, (None,))[0]
            regressed = frm == "pass" and to == "fail" and current == "fail"
            days_failing = round(_days(last_b.collected_at, latest.collected_at), 1)
            if regressed:
                out.append(Finding(
                    kind="compliance_drift",
                    device_id=d,
                    employee_id=latest.employee_id,
                    severity="high" if sev == "high" else "medium",
                    title=f"Compliance regression: '{cid}' went pass -> fail on {_date(last_b)}",
                    explanation=(
                        f"'{cid}' ({sev} severity) passed through {last_a.collected_at:%Y-%m-%d %H:%M} UTC and has failed in "
                        f"every snapshot since {last_b.collected_at:%Y-%m-%d %H:%M} UTC ({days_failing} days)."
                    ),
                    metrics={"check_id": cid, "check_severity": sev, "direction": "regression",
                             "failing_since": _date(last_b), "days_failing": days_failing,
                             "transitions": len(transitions)},
                    snapshot_refs=[last_a.ref, last_b.ref, latest.ref],
                ))
            elif to == "pass" and current == "pass":
                out.append(Finding(
                    kind="compliance_drift",
                    device_id=d,
                    employee_id=latest.employee_id,
                    severity="info",
                    title=f"Compliance remediated: '{cid}' fail -> pass on {_date(last_b)}",
                    explanation=f"'{cid}' failed through {_date(last_a)} and has passed since {_date(last_b)}.",
                    metrics={"check_id": cid, "check_severity": sev, "direction": "remediation",
                             "passing_since": _date(last_b), "transitions": len(transitions)},
                    snapshot_refs=[last_a.ref, last_b.ref, latest.ref],
                ))
    return out


def compliance_trend(view: TenantView) -> dict:
    """Tenant-level daily failing-device counts per check (first vs last day + series)."""
    by_day: dict[str, dict[str, set[str]]] = {}
    checks: dict[str, str] = {}
    for d in view.device_ids():
        for s in view.history(d):
            day = _date(s)
            for cid, (st, sev) in s.compliance.items():
                checks[cid] = sev
                by_day.setdefault(day, {}).setdefault(cid, set())
                if st == "fail":
                    by_day[day][cid].add(d)
    days = sorted(by_day)
    n = len(view.device_ids())
    trend = {}
    for cid in sorted(checks):
        series = [len(by_day[day].get(cid, ())) for day in days]
        trend[cid] = {
            "severity": checks[cid],
            "devices_failing_first_day": series[0],
            "devices_failing_last_day": series[-1],
            "failure_rate_first_day_pct": _pct(series[0], n),
            "failure_rate_last_day_pct": _pct(series[-1], n),
            "daily_failing_devices": dict(zip(days, series)),
        }
    return {"devices": n, "first_day": days[0], "last_day": days[-1], "checks": trend}


def unapproved_software(view: TenantView, th: Thresholds) -> list[Finding]:
    out = []
    for d in view.device_ids():
        h = view.history(d)
        latest = h[-1]
        for app, risk in sorted(RISKY_SOFTWARE.items()):
            first_idx = next((i for i, s in enumerate(h) if app in s.software), None)
            if first_idx is None:
                continue
            first = h[first_idx]
            present_now = app in latest.software
            refs = ([h[first_idx - 1].ref] if first_idx > 0 else []) + [first.ref, latest.ref]
            out.append(Finding(
                kind="unapproved_software",
                device_id=d,
                employee_id=latest.employee_id,
                severity="high" if present_now else "low",
                title=f"Unapproved software '{app}' " + ("installed" if present_now else "was installed"),
                explanation=(
                    f"'{app}' ({risk}) first appeared on {_date(first)}"
                    + (f" (absent in the previous snapshot on {_date(h[first_idx - 1])})" if first_idx > 0 else "")
                    + (" and is still installed in the latest snapshot." if present_now else " and has since been removed.")
                ),
                metrics={"software": app, "risk": risk, "first_seen": _date(first), "present_in_latest": present_now},
                snapshot_refs=list(dict.fromkeys(refs)),
            ))
    return out


def telemetry_gaps(view: TenantView, th: Thresholds) -> list[Finding]:
    """Data-quality findings: missing blocks that must NOT be read as real conditions."""
    out = []
    for d in view.device_ids():
        h = view.history(d)
        latest = h[-1]
        if latest.device_type == "laptop":
            missing = [s for s in h if s.battery is None]
            if missing and len(missing) < len(h):
                out.append(Finding(
                    kind="telemetry_gaps",
                    device_id=d,
                    employee_id=latest.employee_id,
                    severity="low",
                    title=f"Battery telemetry missing in {len(missing)}/{len(h)} snapshots",
                    explanation=(
                        "This laptop reports a battery in most snapshots, so the missing battery block is a "
                        "reporting gap, not evidence that the device has no battery. Battery trends for this "
                        "device are computed only over snapshots that include the block."
                    ),
                    metrics={"missing_battery_snapshots": len(missing), "snapshots": len(h)},
                    snapshot_refs=[s.ref for s in missing[:5]],
                ))
        no_net = [s for s in h if not s.network_reported]
        if no_net:
            out.append(Finding(
                kind="telemetry_gaps",
                device_id=d,
                employee_id=latest.employee_id,
                severity="info",
                title=f"Network block missing in {len(no_net)}/{len(h)} snapshots",
                explanation="Agent submitted snapshots without network interfaces (likely offline at collection time).",
                metrics={"missing_network_snapshots": len(no_net), "snapshots": len(h)},
                snapshot_refs=[s.ref for s in no_net[:5]],
            ))
    return out


DETECTORS = {
    "battery_end_of_life": battery_end_of_life,
    "ram_constrained": ram_constrained,
    "storage_constrained": storage_constrained,
    "compliance_drift": compliance_drift,
    "unapproved_software": unapproved_software,
    "telemetry_gaps": telemetry_gaps,
}
INSIGHT_KINDS = tuple(DETECTORS)


def run_detectors(view: TenantView, th: Thresholds, kinds: list[str] | None = None,
                  device_id: str | None = None) -> list[Finding]:
    kinds = list(kinds or INSIGHT_KINDS)
    findings: list[Finding] = []
    for k in kinds:
        findings.extend(DETECTORS[k](view, th))
    if device_id:
        findings = [f for f in findings if f.device_id == device_id]
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.kind, f.device_id))
    return findings
