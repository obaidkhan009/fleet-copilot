"""Deterministic offline backend (no API key required).

This is a rule-based stand-in for the LLM that speaks the same protocol as
the Claude backend: it returns a structured plan, emits Anthropic-format
``tool_use`` blocks, reads the ``tool_result`` blocks, and composes a cited
answer. It exists so that the *entire* LangGraph pipeline - planning, tool
execution, evidence ledger, action policy, approval interrupt, execution,
grounding verification and audit - runs end-to-end, reproducibly, in CI and
in the evaluation suite without network access.

It is intentionally simple (keyword intent routing + templates). It is a
harness for the architecture and guardrails, not a claim about language
understanding; open-ended phrasing is what the Claude backend is for.
"""

from __future__ import annotations

import json
import re

from ..config import COMPANY_NAMES, RISKY_SOFTWARE, THRESHOLDS, UPGRADE_CATALOG
from ..guardrails.action_policy import _DEVICE_TOKEN_CI, DEVICE_TOKEN, EMPLOYEE_TOKEN
from .base import LLMTurn

ALL_KINDS = ["battery_end_of_life", "ram_constrained", "storage_constrained", "compliance_drift",
             "unapproved_software", "telemetry_gaps"]
CHECK_SYNONYMS = [
    ("screen_lock", r"screen[\s_-]?lock|auto[\s-]?lock|lock screen"),
    ("os_up_to_date", r"os[\s_-]?up[\s_-]?to[\s_-]?date|out[\s-]of[\s-]date os|os updates?|patch(ing|es)?"),
    ("disk_encryption", r"disk[\s_-]?encryption|encrypt(ed|ion)?|filevault|bitlocker"),
]
UNSUPPORTED_OPS = r"\b(remote(ly)?[\s-]?wipe|wipe|erase|factory[\s-]?reset|reboot|restart|shut[\s-]?down|lock (the|this|that|their) (device|laptop|machine)|delete (the |this )?(device|user|employee))\b"


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def _step(tool: str, purpose: str, **args) -> dict:
    return {"tool": tool, "purpose": purpose, "args": {k: v for k, v in args.items() if v is not None}}


def build_plan(question: str) -> dict:
    q, ql = question, question.lower()
    devices = list(dict.fromkeys(m.upper() for m in _DEVICE_TOKEN_CI.findall(q)))   # ids are case-insensitive
    employees = list(dict.fromkeys(EMPLOYEE_TOKEN.findall(ql)))
    check = next((cid for cid, pat in CHECK_SYNONYMS if re.search(pat, ql)), None)
    if not check:  # a named check we don't recognise is passed through, so the tool can say it doesn't exist
        m = re.search(r"\b(?:failing|fail|failed|passing)\s+(?:the\s+)?([a-z][a-z_-]+)(?:\s+compliance)?\s+checks?\b", ql)
        if m and "severity" not in m.group(1) and m.group(1) not in {"compliance", "any", "high", "medium", "low", "severity", "security", "their",
                                    "our", "these", "all", "a", "an"}:
            check = m.group(1)
    severity = "high" if re.search(r"high[\s-]?severity|critical", ql) else (
        "medium" if re.search(r"medium[\s-]?severity", ql) else None)
    device_type = "laptop" if re.search(r"laptops?|notebooks?", ql) else (
        "desktop" if re.search(r"desktops?|mac ?minis?", ql) else None)
    os_m = re.search(r"(older than|below|before|earlier than|prior to|<)\s*(macos|mac os|windows)\s*([\d.]+(?:\s*\d{2}h\d)?)", ql)
    trend = bool(re.search(r"consistent|trend|over time|approach|drift|regress|getting worse|last (30|thirty) days|"
                           r"pattern|insight|sustain|end[\s-]of[\s-]life|\beol\b|degrad|chronic|persistent", ql))
    wants = set()
    if re.search(r"ticket|remediat|\bfix\b|uninstall|remove", ql):
        wants.add("ticket")
    if re.search(r"\border\b|upgrade|add (more )?(ram|memory|storage)", ql):
        wants.add("upgrade")
    if re.search(r"replac|\bflag\b|refresh", ql):
        wants.add("replace")
    if re.search(r"notify|message|e-?mail|tell (the )?(employee|user|owner)|let (the )?(employee|user)s? know", ql):
        wants.add("notify")
    if re.search(r"take action|propose|recommend|what should (we|i) do|next steps|act on|deal with|address", ql):
        wants.add("auto")

    topics = set()
    for name, pat in [("storage", r"disk|storage|space|\bssd\b"), ("ram", r"\bram\b|memory"),
                      ("battery", r"batter"), ("software", r"software|\bapps?\b|applications?|utorrent|teamviewer|cleanmymac|unapproved|risky"),
                      ("gaps", r"\bgaps?\b|missing data|data quality|telemetry quality"),
                      ("overview", r"overview|summary|summari[sz]e|health|status|how (is|are|does)|state of|report on|top (issues|risks|priorities)")]:
        if re.search(pat, ql):
            topics.add(name)
    if check or re.search(r"complian|failing|checks?\b|security posture", ql):
        topics.add("compliance")
    if os_m or re.search(r"\bos\b|operating system|macos|windows", ql):
        topics.add("os")

    steps: list[dict] = []
    assumptions: list[str] = []
    if re.search(UNSUPPORTED_OPS, ql):
        return {"intent": "out_of_scope", "goal": question, "steps": [], "needs_actions": False,
                "assumptions": ["requested operation is not supported by any tool"],
                "_offline": {"reason": "unsupported_operation"}}
    if not (topics or devices or employees or wants):
        return {"intent": "out_of_scope", "goal": question, "steps": [], "needs_actions": False,
                "assumptions": [], "_offline": {"reason": "no_fleet_topic"}}

    needs_actions = bool(wants)
    for d in devices[:3]:
        steps.append(_step("get_device_profile", f"Load current state and 30-day stats for {d}", device_id=d))
        if trend or needs_actions or "overview" in topics:
            steps.append(_step("detect_insights", f"Run all detectors for {d}", kinds=ALL_KINDS, device_id=d))
    for e in employees[:3]:
        steps.append(_step("search_devices", f"Find devices assigned to {e}", employee_id=e))

    if not devices:
        if "storage" in topics:
            if trend or needs_actions:
                steps.append(_step("detect_insights", "Find devices consistently low on storage", kinds=["storage_constrained"]))
            else:
                steps.append(_step("search_devices", "Find devices currently low on disk", device_type=device_type,
                                   max_disk_free_pct=THRESHOLDS.disk_low_free_pct))
            assumptions.append(f"low disk = free < {THRESHOLDS.disk_low_free_pct:.0f}% (critical < {THRESHOLDS.disk_critical_free_pct:.0f}%)")
        if "ram" in topics:
            if trend or needs_actions:
                steps.append(_step("detect_insights", "Find devices consistently RAM constrained", kinds=["ram_constrained"]))
            else:
                steps.append(_step("search_devices", "Find devices with high memory use now", device_type=device_type,
                                   min_mem_used_pct=THRESHOLDS.ram_high_used_pct))
            assumptions.append(f"high memory = used > {THRESHOLDS.ram_high_used_pct:.0f}%")
        if "battery" in topics:
            steps.append(_step("detect_insights", "Find batteries at or approaching end of life", kinds=["battery_end_of_life"]))
        if "compliance" in topics:
            if trend:
                steps.append(_step("detect_insights", "Detect compliance regressions over the window", kinds=["compliance_drift"]))
            else:
                steps.append(_step("compliance_report", "List current compliance failures", check_id=check,
                                   severity=severity, device_type=device_type))
        if "software" in topics:
            steps.append(_step("detect_insights", "Find unapproved software installations", kinds=["unapproved_software"]))
        if "gaps" in topics:
            steps.append(_step("detect_insights", "Find telemetry gaps", kinds=["telemetry_gaps"]))
        if os_m:
            fam = "Windows" if "windows" in os_m.group(2) else "macOS"
            steps.append(_step("search_devices", f"Find devices on {fam} older than {os_m.group(3)}", os_family=fam,
                               os_version_below=os_m.group(3).strip(), device_type=device_type))
            assumptions.append(f"'older than {fam} {os_m.group(3)}' compares versions within {fam} only")
        elif "os" in topics and not steps:
            steps.append(_step("fleet_overview", "Summarise OS versions across the fleet"))
        if not steps or ("overview" in topics and not employees) or (wants and not steps):
            if not any(s["tool"] == "fleet_overview" for s in steps):
                steps.append(_step("fleet_overview", "Summarise fleet composition and compliance"))
            steps.append(_step("detect_insights", "Run all trend detectors", kinds=ALL_KINDS))

    intent = "action_request" if needs_actions else ("insight_analysis" if trend or any(
        s["tool"] == "detect_insights" for s in steps) else "fleet_question")
    plan_steps = steps + ([{"tool": "<action tools>", "purpose": "Propose evidence-backed actions for approval", "args": {}}]
                          if needs_actions else [])
    return {"intent": intent, "goal": question, "steps": plan_steps, "needs_actions": needs_actions,
            "assumptions": assumptions,
            "_offline": {"devices": devices, "wants": sorted(wants), "topics": sorted(topics), "check": check,
                         "spec": (re.search(r"(\d+)\s*(gb|tb)", ql).group(0).upper().replace("GB", " GB").replace("TB", " TB").replace("  ", " ")
                                  if re.search(r"(\d+)\s*(gb|tb)", ql) else None)}}


# ---------------------------------------------------------------------------
# helpers to read the transcript
# ---------------------------------------------------------------------------

def _turn_slice(messages: list[dict]) -> list[dict]:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "user" and isinstance(m["content"], list) and any(
                b.get("type") == "text" and b.get("text", "").startswith("<plan>") for b in m["content"]):
            return messages[i + 1:]
    return []


def _results(turn: list[dict]) -> list[dict]:
    calls = {}
    out = []
    for m in turn:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                calls[b["id"]] = b
            elif b.get("type") == "tool_result":
                call = calls.get(b["tool_use_id"], {})
                content = b.get("content")
                text = content if isinstance(content, str) else "".join(c.get("text", "") for c in content or [])
                try:
                    data = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    data = {"error": text}
                out.append({"tool": call.get("name"), "input": call.get("input", {}), "data": data,
                            "is_error": bool(b.get("is_error"))})
    return out


def _next_size(options: list[str], current_gb: float) -> str | None:
    def gb(s):
        n, unit = s.split()
        return float(n) * (1024 if unit == "TB" else 1)
    bigger = [o for o in options if gb(o) > current_gb]
    return bigger[0] if bigger else None


# ---------------------------------------------------------------------------
# action derivation
# ---------------------------------------------------------------------------

def derive_actions(plan: dict, results: list[dict]) -> list[tuple[str, dict]]:
    meta = plan.get("_offline", {})
    wants = set(meta.get("wants", []))
    auto = "auto" in wants or not (wants - {"auto"})
    allow = lambda kind: auto or kind in wants  # noqa: E731
    actions: list[tuple[str, dict]] = []
    seen = set()

    def add(tool, args, key):
        if key not in seen:
            seen.add(key)
            actions.append((tool, args))

    profiles = {r["data"]["device_id"]: r["data"] for r in results if r["tool"] == "get_device_profile" and not r["is_error"]}
    findings = [f for r in results if r["tool"] == "detect_insights" and not r["is_error"] for f in r["data"].get("findings", [])]
    comp_rows = [row for r in results if r["tool"] == "compliance_report" and not r["is_error"] for row in r["data"].get("failing", [])]

    # Explicit single-device requests ("order 32 GB RAM for X", "replace X", "notify the owner of X")
    for dev, prof in profiles.items():
        dev_f = [f for f in findings if f["device_id"] == dev]
        ev = [prof["evidence_id"]] + [f["evidence_id"] for f in dev_f]
        if "upgrade" in wants:
            comp = "storage" if "storage" in meta.get("topics", []) else ("battery" if "battery" in meta.get("topics", []) else "ram")
            current = prof["ram_gb"] if comp == "ram" else prof["disk_size_gb"]
            spec = meta.get("spec") if comp != "battery" else "OEM replacement battery"
            spec = spec or (_next_size(UPGRADE_CATALOG[comp], current) if comp != "battery" else spec) or UPGRADE_CATALOG[comp][-1]
            add("create_upgrade_order", {"device_id": dev, "component": comp, "spec": spec,
                                         "justification": f"Requested {comp} upgrade for {dev} based on its telemetry profile.",
                                         "evidence_ids": ev}, ("upg", dev, comp))
        if "replace" in wants:
            reason = "; ".join(f["title"] for f in dev_f) or "Requested replacement"
            add("flag_device_for_replacement", {"device_id": dev, "reason": reason, "evidence_ids": ev}, ("rpl", dev))
        if "ticket" in wants:
            for cid, c in prof.get("compliance", {}).items():
                if c["status"] == "fail":
                    add("open_remediation_ticket", {"device_id": dev, "check_id": cid,
                                                    "note": f"Remediate failing '{cid}' ({c['severity']}) on {dev}; failing since {c['since']}.",
                                                    "evidence_ids": ev}, ("tkt", dev, cid))
            for app in prof.get("unapproved_software", []):
                add("open_remediation_ticket", {"device_id": dev, "check_id": f"unapproved_software:{app}",
                                                "note": f"Remove unapproved software '{app}' from {dev}.",
                                                "evidence_ids": ev}, ("tkt", dev, app))
        if "notify" in wants:
            issues = [f["title"] for f in dev_f] or [f"failing check '{c}'" for c, v in prof.get("compliance", {}).items() if v["status"] == "fail"]
            add("notify_employee", {"employee_id": prof["employee_id"], "device_id": dev, "evidence_ids": ev,
                                    "message": f"Hi, IT noticed the following on your device {prof['hostname']}: "
                                               + "; ".join(issues[:3]) + ". We'll follow up with next steps."},
                ("msg", dev))
    if profiles:
        return actions[:10]

    # Fleet-wide requests: map findings / failures to the standard remediation for each.
    for row in comp_rows:
        if allow("ticket"):
            add("open_remediation_ticket", {"device_id": row["device_id"], "check_id": row["check_id"],
                                            "note": f"Remediate failing '{row['check_id']}' ({row['severity']}); failing since {row['failing_since']}.",
                                            "evidence_ids": [row["evidence_id"]]}, ("tkt", row["device_id"], row["check_id"]))
    for f in findings:
        m, dev, ev = f["metrics"], f["device_id"], [f["evidence_id"]]
        if f["severity"] in ("low", "info"):
            continue
        if f["kind"] == "battery_end_of_life" and m.get("battery_serviceable") and allow("upgrade"):
            add("create_upgrade_order", {"device_id": dev, "component": "battery", "spec": "OEM replacement battery",
                                         "justification": f["title"], "evidence_ids": ev}, ("upg", dev, "battery"))
        elif f["kind"] == "ram_constrained":
            if m.get("ram_upgradeable") and allow("upgrade"):
                spec = _next_size(UPGRADE_CATALOG["ram"], m["ram_gb"])
                add("create_upgrade_order", {"device_id": dev, "component": "ram", "spec": spec,
                                             "justification": f["title"], "evidence_ids": ev}, ("upg", dev, "ram"))
            elif m.get("ram_upgradeable") is False and allow("replace"):
                add("flag_device_for_replacement", {"device_id": dev, "reason": f["title"] + " (RAM is soldered)",
                                                    "evidence_ids": ev}, ("rpl", dev))
        elif f["kind"] == "storage_constrained" and m.get("sustained"):
            if m.get("storage_upgradeable") and allow("upgrade"):
                spec = _next_size(UPGRADE_CATALOG["storage"], m["disk_size_gb"])
                if spec:
                    add("create_upgrade_order", {"device_id": dev, "component": "storage", "spec": spec,
                                                 "justification": f["title"], "evidence_ids": ev}, ("upg", dev, "storage"))
            elif m.get("storage_upgradeable") is False and allow("replace"):
                add("flag_device_for_replacement", {"device_id": dev, "reason": f["title"] + " (storage is soldered)",
                                                    "evidence_ids": ev}, ("rpl", dev))
        elif f["kind"] == "compliance_drift" and allow("ticket"):
            cid = m["check_id"]
            add("open_remediation_ticket", {"device_id": dev, "check_id": cid,
                                            "note": f"'{cid}' regressed to failing on {m['failing_since']}; restore the setting.",
                                            "evidence_ids": ev}, ("tkt", dev, cid))
        elif f["kind"] == "unapproved_software" and m.get("present_in_latest") and allow("ticket"):
            app = m["software"]
            add("open_remediation_ticket", {"device_id": dev, "check_id": f"unapproved_software:{app}",
                                            "note": f"Remove '{app}' ({m['risk']}).", "evidence_ids": ev}, ("tkt", dev, app))
        if "notify" in wants:
            add("notify_employee", {"employee_id": f["employee_id"], "device_id": dev, "evidence_ids": ev,
                                    "message": f"Hi, IT noticed an issue on your device: {f['title']}. We'll be in touch with next steps."},
                ("msg", dev))
    # Don't order RAM/storage for a device we're also proposing to replace.
    replacing = {a["device_id"] for t, a in actions if t == "flag_device_for_replacement"}
    actions = [(t, a) for t, a in actions
               if not (t == "create_upgrade_order" and a["device_id"] in replacing and a["component"] != "battery")]
    return actions[:10]


# ---------------------------------------------------------------------------
# answer composition
# ---------------------------------------------------------------------------

def _describe_filters(f: dict) -> str:
    parts = []
    if f.get("device_type"):
        parts.append(f"{f['device_type']}s")
    if f.get("os_family") and f.get("os_version_below"):
        parts.append(f"running {f['os_family']} older than {f['os_version_below']}")
    elif f.get("os_family"):
        parts.append(f"running {f['os_family']}")
    if f.get("max_disk_free_pct") is not None:
        parts.append(f"with less than {f['max_disk_free_pct']:g}% free disk")
    if f.get("min_mem_used_pct") is not None:
        parts.append(f"using more than {f['min_mem_used_pct']:g}% of RAM")
    if f.get("failing_check_id"):
        parts.append(f"failing '{f['failing_check_id']}'")
    if f.get("failing_severity"):
        parts.append(f"failing a {f['failing_severity']}-severity check")
    if f.get("employee_id"):
        parts.append(f"assigned to {f['employee_id']}")
    if f.get("has_software"):
        parts.append(f"with {f['has_software']} installed")
    return " ".join(parts) or "devices"


def _compose_search(r: dict) -> list[str]:
    d, f = r["data"], r["input"]
    lines = [f"**{d['matched_count']} of {d['total_devices_in_scope']}** devices {_describe_filters(f)} "
             f"(latest snapshot per device, as of {d['as_of']}) [{d['scope_evidence_id']}]."]
    for row in d["devices"]:
        bits = [f"{row['model']}, {row['employee_id']}"]
        if "max_disk_free_pct" in f or not f:
            crit = " - critical" if row["disk_free_pct"] < THRESHOLDS.disk_critical_free_pct else ""
            bits.append(f"{row['disk_free_pct']}% free ({row['disk_free_gb']} of {row['disk_size_gb']} GB){crit}")
        if "min_mem_used_pct" in f:
            bits.append(f"{row['mem_used_pct']}% of {row['ram_gb']} GB RAM in use")
        if "os_family" in f or "employee_id" in f:
            bits.append(row["os"])
        if "failing_check_id" in f or "failing_severity" in f or "employee_id" in f:
            bits.append("failing: " + (", ".join(c["check_id"] for c in row["failing_checks"]) or "none"))
        lines.append(f"- **{row['device_id']}** ({'; '.join(bits)}) [{row['evidence_id']}]")
    return lines


def _compose_compliance(r: dict) -> list[str]:
    d, f = r["data"], r["input"]
    scope = f"{f['device_type']}s" if f.get("device_type") else "devices"
    what = " ".join(x for x in [f"{f['severity']}-severity" if f.get("severity") else "",
                                f"'{f['check_id']}'" if f.get("check_id") else ""] if x) or "any"
    lines = []
    if d["failing_count"] == 0:
        lines.append(f"**No {scope}** are currently failing {what} compliance checks "
                     f"({d['devices_in_scope']} {scope} checked, latest snapshots as of {d['as_of']}) [{d['scope_evidence_id']}].")
        if f.get("severity"):
            same = [c for c in d["checks_catalog"] if c["severity"] == f["severity"]]
            other = [c for c in d["checks_catalog"] if c["severity"] != f["severity"] and c["failing_devices"]]
            if same:
                lines.append("- " + "; ".join(f"`{c['check_id']}` is the {f['severity']}-severity check and passes on "
                                              f"{c['passing_devices']} of {c['passing_devices'] + c['failing_devices']} devices" for c in same)
                             + f" [{d['scope_evidence_id']}].")
            if other:
                lines.append("- For context, lower-severity failures do exist across the fleet: "
                             + ", ".join(f"`{c['check_id']}` ({c['severity']}) on {c['failing_devices']} devices" for c in other)
                             + f" [{d['scope_evidence_id']}].")
        return lines
    lines.append(f"**{d['failing_count']}** failing {what} check results across {d['devices_in_scope']} {scope} "
                 f"(latest snapshots as of {d['as_of']}) [{d['scope_evidence_id']}]:")
    for row in d["failing"]:
        since = "the whole 30-day window" if row["failing_entire_window"] else f"since {row['failing_since']}"
        lines.append(f"- **{row['device_id']}** ({row['model']}, {row['employee_id']}): `{row['check_id']}` "
                     f"({row['severity']}) failing {since} [{row['evidence_id']}]")
    return lines


KIND_HEADINGS = {
    "battery_end_of_life": "Batteries at or approaching end of life",
    "ram_constrained": "Devices consistently RAM constrained",
    "storage_constrained": "Devices low on storage",
    "compliance_drift": "Compliance drift",
    "unapproved_software": "Unapproved software",
    "telemetry_gaps": "Telemetry gaps (data quality)",
}


def _compose_insights(r: dict) -> list[str]:
    d = r["data"]
    lines = []
    by_kind: dict[str, list] = {}
    for f in d["findings"]:
        by_kind.setdefault(f["kind"], []).append(f)
    for kind in d["kinds"]:
        fs = by_kind.get(kind, [])
        lines.append(f"**{KIND_HEADINGS[kind]}** - {len(fs)} finding(s) over {d['devices_analyzed']} device(s) "
                     f"[{d['scope_evidence_id']}]. Rule: {d['rules'][kind]}.")
        for f in fs:
            lines.append(f"- **{f['device_id']}** ({f['severity']}, {f['employee_id']}): {f['title']}. "
                         f"{f['explanation']} [{f['evidence_id']}]")
        if kind == "compliance_drift" and "tenant_compliance_trend" in d:
            for cid, t in d["tenant_compliance_trend"].items():
                if t["devices_failing_first_day"] != t["devices_failing_last_day"]:
                    lines.append(f"- Fleet trend: `{cid}` failures went from {t['devices_failing_first_day']} to "
                                 f"{t['devices_failing_last_day']} devices ({t['failure_rate_first_day_pct']}% -> "
                                 f"{t['failure_rate_last_day_pct']}%) [{d['tenant_compliance_trend_evidence_id']}]")
    return lines


def _compose_profile(r: dict) -> list[str]:
    p = r["data"]
    e = p["evidence_id"]
    failing = [f"`{c}` ({v['severity']}) since {v['since']}" for c, v in p["compliance"].items() if v["status"] == "fail"]
    bat = p["battery_trend"]
    lines = [f"**{p['device_id']}** - {p['model']} ({p['os']}), assigned to {p['employee_id']}, last seen {p['collected_at']} [{e}]."]
    lines.append(f"- Disk: {p['disk_free_pct']}% free now (min {p['disk_free_pct_stats']['min']}%, mean "
                 f"{p['disk_free_pct_stats']['mean']}% over the window) [{e}]")
    lines.append(f"- Memory: {p['mem_used_pct']}% of {p['ram_gb']} GB used now (mean {p['mem_used_pct_stats']['mean']}%) [{e}]")
    if bat:
        lines.append(f"- Battery: {bat['cycle_count_latest']} cycles, condition '{bat['condition_latest']}', capacity "
                     f"{bat['full_charge_capacity_start']} -> {bat['full_charge_capacity_latest']} mAh [{e}]")
    lines.append("- Compliance: " + ("failing " + "; ".join(failing) if failing else "all checks passing") + f" [{e}]")
    if p["unapproved_software"]:
        lines.append(f"- Unapproved software: {', '.join(p['unapproved_software'])} [{e}]")
    return lines


def _compose_overview(r: dict) -> list[str]:
    d = r["data"]
    e = d["evidence_id"]
    comp = ", ".join(f"`{c['check_id']}` ({c['severity']}) failing on {c['failing_devices']}" for c in d["compliance_checks_latest"])
    return [f"**{d['tenant']} fleet:** {d['devices']} devices for {d['employees']} employees as of {d['as_of']} [{e}].",
            f"- By OS: {', '.join(f'{k}: {v}' for k, v in d['by_os'].items())} [{e}]",
            f"- By type: {', '.join(f'{k}: {v}' for k, v in d['by_device_type'].items())} [{e}]",
            f"- Compliance (latest): {comp} [{e}]"]


def compose_answer(plan: dict, results: list[dict], action_results: list[dict], tenant_name: str) -> str:
    meta = plan.get("_offline", {})
    if plan["intent"] == "out_of_scope":
        if meta.get("reason") == "unsupported_operation":
            return ("I can't do that. Remote operations such as wiping, locking, rebooting or deleting devices are not "
                    "available to me. I can propose remediation tickets, hardware upgrade orders, replacement flags and "
                    "employee notifications, each of which requires your explicit approval.")
        return (f"That's outside what I can help with. I answer questions about {tenant_name}'s device telemetry "
                "(health, compliance, battery, storage, memory, OS versions, installed software) and can propose "
                "operational actions for your approval.")
    lines: list[str] = []
    for r in results:
        if r["is_error"]:
            err = r["data"].get("error", "error")
            m = re.match(r"Unknown check_id '([^']+)'\. Known checks: (.*)", err)
            if m:
                lines.append(f"The telemetry does not include a '{m.group(1)}' compliance check, so I can't say which "
                             f"devices fail it. The checks reported by the device agent are: {m.group(2)}.")
            else:
                lines.append(f"I couldn't complete `{r['tool']}`: {err}")
            continue
        composer = {"search_devices": _compose_search, "compliance_report": _compose_compliance,
                    "detect_insights": _compose_insights, "get_device_profile": _compose_profile,
                    "fleet_overview": _compose_overview}.get(r["tool"])
        if composer:
            lines.extend(composer(r))
            lines.append("")
    if action_results:
        ok = [a for a in action_results if a["data"].get("status") == "pending_approval"]
        bad = [a for a in action_results if a["data"].get("status") == "rejected_by_policy"]
        if ok:
            lines.append("**Proposed actions (awaiting your approval - nothing has been executed):**")
            for a in ok:
                i = a["input"]
                target = i.get("check_id") or i.get("component") or i.get("employee_id") or ""
                lines.append(f"- {a['data']['proposal_id']}: `{a['tool']}` for {i['device_id']}"
                             + (f" ({target})" if target else "") + f" [{', '.join(i['evidence_ids'])}]")
        if bad:
            lines.append("**Not proposed - blocked by the evidence policy:**")
            for a in bad:
                i = a["input"]
                lines.append(f"- `{a['tool']}` for {i.get('device_id')}: " + " ".join(a["data"].get("reasons", [])))
    elif plan.get("needs_actions"):
        lines.append("No actions are warranted: the telemetry above does not show a condition that an available "
                     "action would address.")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------

class OfflineBackend:
    name = "offline"

    def plan(self, *, system: str, question: str, history: list[str]) -> tuple[dict, dict]:
        return build_plan(question), {}

    def step(self, *, system: str, messages: list[dict], tools: list[dict], plan: dict,
             allow_tools: bool) -> LLMTurn:
        turn = _turn_slice(messages)
        results = _results(turn)
        tool_names = {t["name"] for t in tools}
        reads = [r for r in results if r["tool"] in tool_names and not r["tool"].startswith(
            ("create_", "open_", "flag_", "notify_"))]
        acts = [r for r in results if r not in reads]
        n_calls = sum(1 for m in turn if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_use")
        uid = lambda i: f"toolu_offline_{len(messages)}_{i}"  # noqa: E731

        if allow_tools and n_calls == 0 and plan.get("steps") and plan["intent"] != "out_of_scope":
            blocks = [{"type": "tool_use", "id": uid(i), "name": s["tool"], "input": s["args"]}
                      for i, s in enumerate(s for s in plan["steps"] if s["tool"] in tool_names)]
            if blocks:
                return LLMTurn([{"type": "text", "text": "Gathering evidence per plan."}] + blocks, "tool_use")
        if allow_tools and plan.get("needs_actions") and not acts and reads:
            proposals = derive_actions(plan, reads)
            if proposals:
                blocks = [{"type": "tool_use", "id": uid(i), "name": t, "input": a} for i, (t, a) in enumerate(proposals)]
                return LLMTurn([{"type": "text", "text": "Proposing evidence-backed actions."}] + blocks, "tool_use")
        tenant = next((n for t, n in COMPANY_NAMES.items() if t in system), "your organization")
        text = compose_answer(plan, reads, acts, tenant)
        return LLMTurn([{"type": "text", "text": text}], "end_turn")
