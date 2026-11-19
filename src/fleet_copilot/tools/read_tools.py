"""Read-only tools. All are tenant-scoped through ``ctx.view``.

Every row returned to the model carries an ``evidence_id`` registered in the
ledger, plus the snapshot ref it came from. Negative results ("no device
matched") also produce evidence covering the devices that were checked, so a
"none" answer is just as traceable as a positive one.
"""

from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..analytics.insights import INSIGHT_KINDS, compliance_trend, run_detectors, serviceable, slope_per_day
from ..config import APPROVED_SOFTWARE, COMPANY_NAMES, RISKY_SOFTWARE
from ..data.models import Snapshot, os_version_key
from .registry import ArgsBase, ToolContext, ToolError, ToolSpec


class _Args(ArgsBase):
    pass


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_device(ctx: ToolContext, device_id: str) -> None:
    if not ctx.view.has_device(device_id):
        # Same message whether the id exists in another tenant or nowhere: no existence oracle.
        raise ToolError(f"Device '{device_id}' was not found in your organization's fleet.")


def _device_row(s: Snapshot) -> dict:
    return {
        "device_id": s.device_id,
        "employee_id": s.employee_id,
        "hostname": s.hostname,
        "model": s.model_name,
        "device_type": s.device_type,
        "os": f"{s.os_family} {s.os_version}",
        "ram_gb": s.ram_gb,
        "mem_used_pct": s.mem_used_pct,
        "disk_free_pct": s.disk_free_pct,
        "disk_free_gb": round(s.disk_free_bytes / 2**30, 1),
        "disk_size_gb": s.disk_size_gb,
        "failing_checks": [{"check_id": c, "severity": sev} for c, sev in s.failing_checks],
        "battery": (
            {"condition": s.battery.condition, "cycle_count": s.battery.cycle_count}
            if s.battery else ("not reported in this snapshot" if s.device_type == "laptop" else "no battery (desktop)")
        ),
        "unapproved_software": sorted(set(s.software) & set(RISKY_SOFTWARE)),
        "collected_at": _ts(s.collected_at),
        "snapshot_ref": s.ref,
    }


# ---------------------------------------------------------------------------
# fleet_overview
# ---------------------------------------------------------------------------

class FleetOverviewArgs(_Args):
    pass


def fleet_overview(ctx: ToolContext, args: FleetOverviewArgs) -> dict:
    latest = ctx.view.latest_all()

    def count(key):
        out: dict[str, int] = {}
        for s in latest:
            out[key(s)] = out.get(key(s), 0) + 1
        return dict(sorted(out.items()))

    failing: dict[str, dict] = {}
    for s in latest:
        for cid, (st, sev) in s.compliance.items():
            row = failing.setdefault(cid, {"check_id": cid, "severity": sev, "failing_devices": 0})
            row["failing_devices"] += st == "fail"
    data = {
        "tenant": COMPANY_NAMES.get(ctx.tenant_id, ctx.tenant_id),
        "as_of": _ts(ctx.view.as_of),
        "window": f"{_ts(ctx.view.window_start)} .. {_ts(ctx.view.as_of)}",
        "devices": len(latest),
        "employees": len(ctx.view.employee_ids()),
        "by_device_type": count(lambda s: s.device_type),
        "by_os": count(lambda s: f"{s.os_family} {s.os_version}"),
        "by_model": count(lambda s: s.model_name),
        "compliance_checks_latest": sorted(failing.values(), key=lambda r: r["check_id"]),
    }
    data["evidence_id"] = ctx.ledger.cite(
        tool="fleet_overview", tenant_id=ctx.tenant_id, args={},
        claim=f"Fleet composition and latest compliance across {len(latest)} devices",
        devices=[s.device_id for s in latest], snapshot_refs=[s.ref for s in latest],
        data={k: v for k, v in data.items() if k != "evidence_id"},
    )
    return data


# ---------------------------------------------------------------------------
# search_devices
# ---------------------------------------------------------------------------

class SearchDevicesArgs(_Args):
    device_type: Literal["laptop", "desktop"] | None = Field(None, description="Filter by form factor.")
    os_family: Literal["macOS", "Windows"] | None = Field(None, description="Filter by OS family.")
    os_version_below: str | None = Field(
        None, description="Only devices whose OS version is strictly older than this, compared within os_family "
                          "(e.g. '15' or '14.5' for macOS, '11 23H2' for Windows). Requires os_family.")
    model_name_contains: str | None = Field(None, description="Case-insensitive substring of the model name, e.g. 'MacBook'.")
    max_disk_free_pct: float | None = Field(None, description="Only devices with free disk % strictly below this value.")
    min_mem_used_pct: float | None = Field(None, description="Only devices with memory used % strictly above this value.")
    failing_check_id: str | None = Field(None, description="Only devices currently failing this compliance check id.")
    failing_severity: Literal["low", "medium", "high"] | None = Field(
        None, description="Only devices currently failing at least one check of this severity.")
    battery_condition: str | None = Field(None, description="Exact battery condition, e.g. 'Service Recommended'.")
    has_software: str | None = Field(None, description="Only devices with this application installed (exact name).")
    employee_id: str | None = Field(None, description="Only devices assigned to this employee.")


def search_devices(ctx: ToolContext, args: SearchDevicesArgs) -> dict:
    if args.os_version_below and not args.os_family:
        raise ToolError("os_version_below requires os_family (versions are only comparable within one OS family).")
    latest = ctx.view.latest_all()
    out = []
    for s in latest:
        if args.device_type and s.device_type != args.device_type:
            continue
        if args.os_family and s.os_family != args.os_family:
            continue
        if args.os_version_below:
            bound = os_version_key(args.os_family, args.os_version_below)
            n = max(len(bound), len(s.os_key))
            if not (s.os_key + (0,) * (n - len(s.os_key)) < bound + (0,) * (n - len(bound))):
                continue
        if args.model_name_contains and args.model_name_contains.lower() not in s.model_name.lower():
            continue
        if args.max_disk_free_pct is not None and not s.disk_free_pct < args.max_disk_free_pct:
            continue
        if args.min_mem_used_pct is not None and not s.mem_used_pct > args.min_mem_used_pct:
            continue
        if args.failing_check_id and s.compliance.get(args.failing_check_id, ("",))[0] != "fail":
            continue
        if args.failing_severity and not any(sev == args.failing_severity for _, sev in s.failing_checks):
            continue
        if args.battery_condition and not (s.battery and s.battery.condition == args.battery_condition):
            continue
        if args.has_software and args.has_software not in s.software:
            continue
        if args.employee_id and s.employee_id != args.employee_id:
            continue
        out.append(s)

    filters = args.model_dump(exclude_none=True)
    rows = []
    for s in sorted(out, key=lambda s: s.device_id):
        row = _device_row(s)
        row["evidence_id"] = ctx.ledger.cite(
            tool="search_devices", tenant_id=ctx.tenant_id, args=filters, device_id=s.device_id,
            employee_id=s.employee_id, snapshot_refs=[s.ref], data=row,
            claim=f"{s.device_id} matched {filters or 'no filters'} on its latest snapshot ({_ts(s.collected_at)})",
        )
        rows.append(row)
    result = {
        "as_of": _ts(ctx.view.as_of),
        "scope": "latest snapshot of each device in your organization",
        "filters": filters,
        "total_devices_in_scope": len(latest),
        "matched_count": len(rows),
        "devices": rows,
    }
    # The scope evidence makes counts and "none matched" answers citable.
    result["scope_evidence_id"] = ctx.ledger.cite(
        tool="search_devices", tenant_id=ctx.tenant_id, args=filters,
        devices=[s.device_id for s in latest], snapshot_refs=[s.ref for s in latest],
        data={"matched_count": len(rows), "total_devices_in_scope": len(latest),
              "matched": [r["device_id"] for r in rows]},
        claim=f"{len(rows)} of {len(latest)} devices matched {filters or 'no filters'} on latest snapshots",
    )
    return result


# ---------------------------------------------------------------------------
# get_device_profile
# ---------------------------------------------------------------------------

class DeviceArgs(_Args):
    device_id: str = Field(description="Device id (serial number), e.g. '8NM23J95R5I6'.")


def get_device_profile(ctx: ToolContext, args: DeviceArgs) -> dict:
    _require_device(ctx, args.device_id)
    h = ctx.view.history(args.device_id)
    s = h[-1]
    comp = {}
    for cid, (st, sev) in sorted(s.compliance.items()):
        streak = h[-1]
        for snap in reversed(h):
            if snap.compliance.get(cid, (None,))[0] != st:
                break
            streak = snap
        comp[cid] = {"status": st, "severity": sev, "since": _ts(streak.collected_at)}
    bats = [x for x in h if x.battery]
    profile = {
        **_device_row(s),
        "processor": s.processor,
        "disk_encrypted": s.disk_encrypted,
        "window": {"from": _ts(h[0].collected_at), "to": _ts(s.collected_at), "snapshots": len(h)},
        "disk_free_pct_stats": {"min": min(x.disk_free_pct for x in h), "mean": round(mean(x.disk_free_pct for x in h), 2),
                                "latest": s.disk_free_pct,
                                "trend_pct_points_per_day": round(slope_per_day([(x.collected_at, x.disk_free_pct) for x in h]), 3)},
        "mem_used_pct_stats": {"mean": round(mean(x.mem_used_pct for x in h), 2), "max": max(x.mem_used_pct for x in h),
                               "latest": s.mem_used_pct},
        "battery_trend": (
            {"cycle_count_start": bats[0].battery.cycle_count, "cycle_count_latest": bats[-1].battery.cycle_count,
             "full_charge_capacity_start": bats[0].battery.full_charge_capacity,
             "full_charge_capacity_latest": bats[-1].battery.full_charge_capacity,
             "condition_latest": bats[-1].battery.condition,
             "snapshots_with_battery_data": len(bats)} if bats else None
        ),
        "compliance": comp,
        "installed_software": [{"name": n, "approved": n in APPROVED_SOFTWARE,
                                **({"risk": RISKY_SOFTWARE[n]} if n in RISKY_SOFTWARE else {})} for n in s.software],
        "serviceability": {c: serviceable(s.model_name, c) for c in ("ram", "storage", "battery")},
        "devices_assigned_to_same_employee": ctx.view.devices_of(s.employee_id),
    }
    profile["evidence_id"] = ctx.ledger.cite(
        tool="get_device_profile", tenant_id=ctx.tenant_id, args=args.model_dump(), device_id=s.device_id,
        employee_id=s.employee_id, snapshot_refs=[h[0].ref, s.ref],
        data=profile, claim=f"Profile of {s.device_id} over {len(h)} snapshots ({_ts(h[0].collected_at)} .. {_ts(s.collected_at)})",
    )
    return profile


# ---------------------------------------------------------------------------
# get_metric_history
# ---------------------------------------------------------------------------

class MetricHistoryArgs(_Args):
    device_id: str = Field(description="Device id.")
    metric: Literal["disk_free_pct", "mem_used_pct", "battery_cycle_count", "battery_full_charge_capacity",
                    "battery_condition", "compliance_status", "installed_software", "os_version"] = Field(
        description="Which time series to return (one point per snapshot).")
    check_id: str | None = Field(None, description="Required when metric is 'compliance_status'.")


def get_metric_history(ctx: ToolContext, args: MetricHistoryArgs) -> dict:
    _require_device(ctx, args.device_id)
    if args.metric == "compliance_status" and not args.check_id:
        raise ToolError("check_id is required for metric 'compliance_status'.")
    h = ctx.view.history(args.device_id)

    def value(s: Snapshot):
        m = args.metric
        if m == "disk_free_pct":
            return s.disk_free_pct
        if m == "mem_used_pct":
            return s.mem_used_pct
        if m == "os_version":
            return f"{s.os_family} {s.os_version}"
        if m == "installed_software":
            return sorted(s.software)
        if m == "compliance_status":
            return s.compliance.get(args.check_id, ("not reported",))[0]
        if s.battery is None:
            return None
        return {"battery_cycle_count": s.battery.cycle_count,
                "battery_full_charge_capacity": s.battery.full_charge_capacity,
                "battery_condition": s.battery.condition}[m]

    series = [{"t": _ts(s.collected_at), "value": value(s), "ref": s.ref} for s in h]
    numeric = [(s.collected_at, p["value"]) for s, p in zip(h, series) if isinstance(p["value"], (int, float))]
    summary: dict = {"points": len(series), "missing_points": sum(p["value"] is None for p in series)}
    if numeric:
        vals = [v for _, v in numeric]
        summary.update(min=min(vals), max=max(vals), mean=round(mean(vals), 2), first=vals[0], last=vals[-1],
                       trend_per_day=round(slope_per_day(numeric), 4))
    result = {"device_id": args.device_id, "metric": args.metric, "check_id": args.check_id,
              "summary": summary, "series": series}
    result["evidence_id"] = ctx.ledger.cite(
        tool="get_metric_history", tenant_id=ctx.tenant_id, args=args.model_dump(exclude_none=True),
        device_id=args.device_id, employee_id=h[-1].employee_id, snapshot_refs=[p["ref"] for p in series],
        data={"summary": summary}, claim=f"{args.metric} history for {args.device_id} ({len(series)} snapshots)",
    )
    return result


# ---------------------------------------------------------------------------
# compliance_report
# ---------------------------------------------------------------------------

class ComplianceReportArgs(_Args):
    check_id: str | None = Field(None, description="Restrict to one check id, e.g. 'screen_lock'.")
    severity: Literal["low", "medium", "high"] | None = Field(None, description="Restrict to checks of this severity.")
    device_type: Literal["laptop", "desktop"] | None = Field(None, description="Restrict to a form factor.")


def compliance_report(ctx: ToolContext, args: ComplianceReportArgs) -> dict:
    latest = [s for s in ctx.view.latest_all() if not args.device_type or s.device_type == args.device_type]
    catalog: dict[str, dict] = {}
    for s in ctx.view.latest_all():
        for cid, (st, sev) in s.compliance.items():
            catalog.setdefault(cid, {"check_id": cid, "severity": sev, "failing_devices": 0, "passing_devices": 0})
            catalog[cid]["failing_devices" if st == "fail" else "passing_devices"] += 1
    if args.check_id and args.check_id not in catalog:
        raise ToolError(f"Unknown check_id '{args.check_id}'. Known checks: {sorted(catalog)}")
    rows = []
    for s in sorted(latest, key=lambda s: s.device_id):
        h = ctx.view.history(s.device_id)
        for cid, sev in s.failing_checks:
            if args.check_id and cid != args.check_id:
                continue
            if args.severity and sev != args.severity:
                continue
            start = h[-1]
            for snap in reversed(h):
                if snap.compliance.get(cid, (None,))[0] != "fail":
                    break
                start = snap
            row = {"device_id": s.device_id, "employee_id": s.employee_id, "model": s.model_name,
                   "device_type": s.device_type, "os": f"{s.os_family} {s.os_version}", "check_id": cid,
                   "severity": sev, "failing_since": _ts(start.collected_at),
                   "failing_entire_window": start is h[0], "snapshot_ref": s.ref}
            row["evidence_id"] = ctx.ledger.cite(
                tool="compliance_report", tenant_id=ctx.tenant_id, args=args.model_dump(exclude_none=True),
                device_id=s.device_id, employee_id=s.employee_id, snapshot_refs=[start.ref, s.ref], data=row,
                claim=f"{s.device_id} fails '{cid}' ({sev}) on latest snapshot; failing since {_ts(start.collected_at)}",
            )
            rows.append(row)
    filters = args.model_dump(exclude_none=True)
    result = {
        "as_of": _ts(ctx.view.as_of),
        "filters": filters,
        "devices_in_scope": len(latest),
        "checks_catalog": sorted(catalog.values(), key=lambda r: r["check_id"]),
        "failing_count": len(rows),
        "failing": rows,
    }
    result["scope_evidence_id"] = ctx.ledger.cite(
        tool="compliance_report", tenant_id=ctx.tenant_id, args=filters,
        devices=[s.device_id for s in latest], snapshot_refs=[s.ref for s in latest],
        data={"failing_count": len(rows), "devices_in_scope": len(latest), "checks_catalog": result["checks_catalog"],
              "matched": sorted({r["device_id"] for r in rows})},
        claim=f"{len(rows)} failing check results among {len(latest)} devices for filters {filters or 'none'} (latest snapshots)",
    )
    return result


# ---------------------------------------------------------------------------
# detect_insights
# ---------------------------------------------------------------------------

class DetectInsightsArgs(_Args):
    kinds: list[Literal[INSIGHT_KINDS]] = Field(  # type: ignore[valid-type]
        min_length=1, description="Which detectors to run. Use all of them for a general health review.")
    device_id: str | None = Field(None, description="Restrict findings to one device.")
    include_info: bool = Field(False, description="Include informational findings (e.g. remediated checks).")


def detect_insights(ctx: ToolContext, args: DetectInsightsArgs) -> dict:
    if args.device_id:
        _require_device(ctx, args.device_id)
    findings = run_detectors(ctx.view, ctx.thresholds, list(dict.fromkeys(args.kinds)), args.device_id)
    if not args.include_info:
        findings = [f for f in findings if f.severity != "info"]
    rows = []
    for f in findings:
        row = f.to_dict()
        row["evidence_id"] = ctx.ledger.cite(
            tool="detect_insights", tenant_id=ctx.tenant_id, args={"kind": f.kind}, device_id=f.device_id,
            employee_id=f.employee_id, snapshot_refs=f.snapshot_refs, data=row, claim=f"{f.device_id}: {f.title}",
        )
        rows.append(row)
    devices = [args.device_id] if args.device_id else ctx.view.device_ids()
    result: dict = {
        "as_of": _ts(ctx.view.as_of),
        "window": f"{_ts(ctx.view.window_start)} .. {_ts(ctx.view.as_of)}",
        "kinds": list(dict.fromkeys(args.kinds)),
        "rules": {
            "storage_constrained": f"free disk < {ctx.thresholds.disk_low_free_pct}% in >= {ctx.thresholds.sustained_fraction:.0%} of snapshots (or latest below threshold = emerging)",
            "ram_constrained": f"memory used > {ctx.thresholds.ram_high_used_pct}% in >= {ctx.thresholds.sustained_fraction:.0%} of snapshots",
            "battery_end_of_life": f"condition != Normal, or cycles >= {ctx.thresholds.battery_cycle_limit}, or projected to reach it within {ctx.thresholds.battery_eol_horizon_days} days, or capacity drop >= {ctx.thresholds.battery_capacity_drop_pct}%",
            "compliance_drift": "check transitioned pass -> fail and is still failing",
            "unapproved_software": f"risky software outside the approved baseline: {sorted(RISKY_SOFTWARE)}",
            "telemetry_gaps": "snapshots missing battery/network blocks (data quality, not device condition)",
        },
        "devices_analyzed": len(devices),
        "finding_count": len(rows),
        "findings": rows,
    }
    if "compliance_drift" in args.kinds and not args.device_id:
        trend = compliance_trend(ctx.view)
        result["tenant_compliance_trend"] = {
            cid: {k: v for k, v in c.items() if k != "daily_failing_devices"} for cid, c in trend["checks"].items()
        }
        result["tenant_compliance_trend_evidence_id"] = ctx.ledger.cite(
            tool="detect_insights", tenant_id=ctx.tenant_id, args={"kind": "compliance_trend"},
            devices=ctx.view.device_ids(), data=trend,
            snapshot_refs=[s.ref for s in ctx.view.latest_all()],
            claim=f"Daily failing-device counts per check, {trend['first_day']} .. {trend['last_day']}",
        )
    result["scope_evidence_id"] = ctx.ledger.cite(
        tool="detect_insights", tenant_id=ctx.tenant_id, args=args.model_dump(exclude_none=True), devices=devices,
        data={"finding_count": len(rows), "kinds": result["kinds"], "devices_analyzed": len(devices),
              "matched": sorted({r["device_id"] for r in rows if r.get("device_id")}),
              "findings_by_kind": {k: sum(1 for r in rows if r.get("kind") == k) for k in dict.fromkeys(args.kinds)}},
        claim=f"Ran {result['kinds']} over {len(devices)} device(s): {len(rows)} finding(s)",
    )
    return result


READ_TOOLS = [
    ToolSpec("fleet_overview",
             "Summarise the organization's fleet: device counts by type/OS/model, employees, and how many devices "
             "fail each compliance check on their latest snapshot. Good first call for broad questions.",
             FleetOverviewArgs, "read", fleet_overview),
    ToolSpec("search_devices",
             "Filter devices by their LATEST snapshot (OS family/version, form factor, model, free disk %, memory %, "
             "failing compliance check or severity, battery condition, installed app, employee). Returns matching "
             "devices with evidence ids plus a scope_evidence_id that supports counts and 'none matched' answers. "
             "Policy thresholds: low disk = free < 15% (critical < 10%); high memory = used > 85%.",
             SearchDevicesArgs, "read", search_devices),
    ToolSpec("get_device_profile",
             "Full profile of one device: latest state, 30-day stats for disk/memory/battery, compliance status with "
             "since-when, installed software (approved or not), owner, and which components are field-serviceable.",
             DeviceArgs, "read", get_device_profile),
    ToolSpec("get_metric_history",
             "Per-snapshot time series for one device and one metric (disk, memory, battery cycles/capacity/condition, "
             "a compliance check's status, installed software, OS version) with min/max/mean/trend.",
             MetricHistoryArgs, "read", get_metric_history),
    ToolSpec("compliance_report",
             "Current compliance failures from latest snapshots, optionally filtered by check, severity and form "
             "factor. Always includes the catalog of checks with their severities, and since-when each failure has "
             "persisted.",
             ComplianceReportArgs, "read", compliance_report),
    ToolSpec("detect_insights",
             "Run deterministic trend detectors over the 30-day time series: battery_end_of_life, ram_constrained, "
             "storage_constrained (consistently low), compliance_drift (pass->fail regressions + tenant trend), "
             "unapproved_software, telemetry_gaps (data quality). Each finding has metrics, an explanation, the rule "
             "that fired, and snapshot refs.",
             DetectInsightsArgs, "read", detect_insights),
]
