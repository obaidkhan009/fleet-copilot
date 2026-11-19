"""Evidence policy for state-changing actions.

The model's claims are never trusted. For every proposed action the policy
re-derives the justification from telemetry with the same deterministic
detectors and refuses when the data does not support it. Rules:

* Common: target device/employee must be in the caller's tenant; at least one
  cited evidence id must exist in the ledger, belong to the tenant and concern
  the target device; free text must not reference other devices or employees;
  no duplicate of an already submitted or pending action.
* create_upgrade_order: the component must be field-serviceable on that model
  (no RAM orders for soldered-memory laptops), the spec must come from the
  catalogue and be an actual upgrade, and a matching sustained constraint
  (RAM / storage / battery EOL) must be detected.
* open_remediation_ticket: the check must be failing on the latest snapshot
  (or the unapproved app must be installed now).
* flag_device_for_replacement: needs a sustained constraint on a component that
  cannot be serviced on that model (soldered RAM/storage). Problems that an
  upgrade or battery service fixes never justify a replacement, however many.
* notify_employee: employee must own the referenced device; the message must not
  ask for credentials or contain links.
* Free text (all actions): no destructive instructions (wipe / erase / factory
  reset), and it must not describe a compliance check as an issue when that
  check is passing on the device's latest snapshot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..actions_backend import ActionBackend
from ..analytics.insights import run_detectors, serviceable
from ..config import RISKY_SOFTWARE, UPGRADE_CATALOG, Thresholds
from ..data.store import TenantView
from ..evidence import EvidenceLedger

DEVICE_TOKEN = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{12}\b")
EMPLOYEE_TOKEN = re.compile(r"\bemp-[a-z]+-\d+\b")
# Case-insensitive variants used on untrusted text (user input, model output, free-text args), so that
# "i3jlkilkqtac" is treated exactly like "I3JLKILKQTAC". Results are normalised to canonical case.
_DEVICE_TOKEN_CI = re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{12}\b")
_EMPLOYEE_TOKEN_CI = re.compile(r"\bemp-[a-z]+-\d+\b", re.I)


def find_device_ids(text: str) -> set[str]:
    return {m.upper() for m in _DEVICE_TOKEN_CI.findall(text or "")}


def find_employee_ids(text: str) -> set[str]:
    return {m.lower() for m in _EMPLOYEE_TOKEN_CI.findall(text or "")}


CREDENTIAL_REQUEST = re.compile(r"\b(passwords?|passcodes?|pass ?phrases?|mfa|2fa|one[- ]time (codes?|passwords?)|otp|"
                                r"verification codes?|security codes?|credentials?|log ?in details)\b", re.I)
LINK = re.compile(r"(https?://|\bwww\.)", re.I)
DESTRUCTIVE = re.compile(r"\b(wipe|erase|factory[- ]reset|re-?format|format (the|your) (disk|drive)|delete (all|every))", re.I)
CHECK_PHRASES = {
    "disk_encryption": r"disk[ _]encryption|filevault|bitlocker|\bunencrypted\b|\bencryption\b",
    "screen_lock": r"screen[ _]lock",
    "os_up_to_date": r"os[ _]up[ _]to[ _]date|out[- ]of[- ]date os|outdated os|os updates?",
}
RISK = {"notify_employee": "low", "open_remediation_ticket": "low",
        "create_upgrade_order": "medium", "flag_device_for_replacement": "high"}


@dataclass
class PolicyDecision:
    allowed: bool
    risk: str
    reasons: list[str] = field(default_factory=list)       # why it was refused
    verified_facts: list[str] = field(default_factory=list)  # data-derived support

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "risk": self.risk, "reasons": self.reasons,
                "verified_facts": self.verified_facts}


def _size_gb(spec: str) -> float:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(GB|TB)", spec, re.I)
    if not m:
        return 0.0
    return float(m.group(1)) * (1024 if m.group(2).upper() == "TB" else 1)


def evaluate(tool: str, args: dict, *, view: TenantView, ledger: EvidenceLedger, th: Thresholds,
             backend: ActionBackend, pending: list[dict] | None = None) -> PolicyDecision:
    d = PolicyDecision(allowed=False, risk=RISK.get(tool, "high"))
    device_id = args.get("device_id")

    # --- tenant scope ------------------------------------------------------
    if tool == "notify_employee":
        emp = args.get("employee_id", "")
        if not view.has_employee(emp):
            d.reasons.append(f"Employee '{emp}' is not in your organization.")
            return d
    if not device_id or not view.has_device(device_id):
        d.reasons.append(f"Device '{device_id}' was not found in your organization's fleet.")
        return d
    latest = view.latest(device_id)
    if tool == "notify_employee" and latest.employee_id != args.get("employee_id"):
        d.reasons.append(f"Employee '{args.get('employee_id')}' is not assigned device {device_id} "
                         f"(assigned to {latest.employee_id}).")
        return d

    # --- evidence citations -----------------------------------------------
    cited = args.get("evidence_ids") or []
    if not cited:
        d.reasons.append("No evidence cited. Run a read tool first and cite the evidence ids that justify this action.")
    else:
        unknown = [e for e in cited if ledger.get(e) is None]
        foreign = [e for e in cited if ledger.get(e) and ledger.get(e)["tenant_id"] != view.tenant_id]
        # Only device-specific evidence counts: aggregate "scope" records cover every device and
        # would otherwise let any action cite them.
        relevant = [e for e in cited if ledger.get(e) and ledger.get(e)["device_id"] == device_id]
        if unknown:
            d.reasons.append(f"Cited evidence ids do not exist: {unknown}.")
        if foreign:
            d.reasons.append(f"Cited evidence from another tenant: {foreign}.")
        if not relevant:
            d.reasons.append(f"None of the cited evidence is specific to device {device_id}.")

    # --- free-text hygiene --------------------------------------------------
    text = " ".join(str(args.get(k, "")) for k in ("justification", "note", "reason", "message"))
    if not text.strip():
        d.reasons.append("A justification / note / reason / message is required.")
    other_devices = sorted(find_device_ids(text) - {device_id})
    other_emps = sorted(find_employee_ids(text) - {latest.employee_id})
    if other_devices or other_emps:
        # Do not echo the identifiers: they may belong to another tenant.
        d.reasons.append(f"Free text references {len(other_devices) + len(other_emps)} other device/employee id(s); "
                         "each action must concern a single device.")
    if len(text) > 2000:
        d.reasons.append("Free text exceeds 2000 characters.")
    if DESTRUCTIVE.search(text):
        d.reasons.append("Free text contains a destructive instruction (wipe / erase / reset). Destructive operations "
                         "are not supported by Fleet Copilot.")
    if tool == "notify_employee":
        msg = str(args.get("message", ""))
        if CREDENTIAL_REQUEST.search(msg):
            d.reasons.append("Employee messages must never mention or request passwords, codes or credentials "
                             "(phishing risk).")
        if LINK.search(msg):
            d.reasons.append("Employee messages may not contain links; the notification service adds the official "
                             "IT portal link.")
    for cid, pat in CHECK_PHRASES.items():
        st = latest.compliance.get(cid)
        if st and st[0] != "fail" and re.search(pat, text, re.I):
            d.reasons.append(f"The text refers to '{cid}', but that check is passing on {device_id}'s latest snapshot "
                             f"({latest.ref}). Only describe issues the telemetry supports.")

    # --- duplicates -----------------------------------------------------------
    key_fields = {k: args[k] for k in ("device_id", "check_id", "component", "employee_id") if k in args}
    # (Notifications may legitimately repeat; identical messages are still deduped by the backend.)
    for r in (backend.find(view.tenant_id, tool, **key_fields) if tool != "notify_employee" else []):
        d.reasons.append(f"Duplicate: an equivalent action was already submitted ({r['external_id']}).")
        break
    for p in pending or []:
        if p["tool"] == tool and p["status"] == "pending_approval" and all(
                p["args"].get(k) == v for k, v in key_fields.items()):
            d.reasons.append(f"Duplicate of pending proposal {p['id']}.")
            break

    # --- tool-specific, data-derived justification --------------------------
    findings = {f.kind: f for f in run_detectors(view, th, None, device_id) if f.severity != "info"}
    if tool == "create_upgrade_order":
        comp, spec = args.get("component"), args.get("spec", "")
        ok = serviceable(latest.model_name, comp)
        if ok is False:
            d.reasons.append(f"{comp} is not field-upgradeable on a {latest.model_name}; "
                             "consider flag_device_for_replacement instead.")
        if spec not in UPGRADE_CATALOG.get(comp, []):
            d.reasons.append(f"Spec '{spec}' is not in the {comp} catalogue {UPGRADE_CATALOG.get(comp)}.")
        need = {"ram": "ram_constrained", "storage": "storage_constrained", "battery": "battery_end_of_life"}.get(comp)
        f = findings.get(need)
        if comp == "storage" and f and not f.metrics.get("sustained") and latest.disk_free_pct >= th.disk_critical_free_pct:
            f = None
        if not f:
            d.reasons.append(f"Telemetry does not show a sustained {need} condition on {device_id}.")
        else:
            d.verified_facts.append(f.explanation)
        if comp == "ram" and _size_gb(spec) <= latest.ram_gb:
            d.reasons.append(f"Spec '{spec}' is not larger than installed RAM ({latest.ram_gb} GB).")
        if comp == "storage" and _size_gb(spec) <= latest.disk_size_gb:
            d.reasons.append(f"Spec '{spec}' is not larger than the installed disk ({latest.disk_size_gb} GB).")

    elif tool == "open_remediation_ticket":
        cid = args.get("check_id", "")
        if cid.startswith("unapproved_software:"):
            app = cid.split(":", 1)[1]
            if app not in RISKY_SOFTWARE:
                d.reasons.append(f"'{app}' is not on the unapproved-software list.")
            elif app not in latest.software:
                d.reasons.append(f"'{app}' is not installed on {device_id} in the latest snapshot.")
            else:
                d.verified_facts.append(f"'{app}' is installed on {device_id} as of {latest.ref}.")
        else:
            st = latest.compliance.get(cid)
            if st is None:
                d.reasons.append(f"Check '{cid}' is not reported for {device_id}.")
            elif st[0] != "fail":
                d.reasons.append(f"Check '{cid}' is passing on {device_id}'s latest snapshot ({latest.ref}); nothing to remediate.")
            else:
                d.verified_facts.append(f"'{cid}' ({st[1]}) is failing on {latest.ref}.")

    elif tool == "flag_device_for_replacement":
        hw = [findings[k] for k in ("battery_end_of_life", "ram_constrained", "storage_constrained") if k in findings]
        non_serviceable = [
            f for f in hw
            if (f.kind == "ram_constrained" and serviceable(latest.model_name, "ram") is False)
            or (f.kind == "storage_constrained" and f.metrics.get("sustained") and serviceable(latest.model_name, "storage") is False)
        ]
        if non_serviceable:
            d.verified_facts.extend(f.explanation for f in hw)
        else:
            d.reasons.append(
                "Replacement requires a sustained constraint on a component that cannot be serviced on a "
                f"{latest.model_name}; found {[f.kind for f in hw] or 'none'}, all fixable in place. "
                "Propose an upgrade order or battery service instead.")

    elif tool == "notify_employee":
        if findings:
            d.verified_facts.extend(f"{f.kind}: {f.title}" for f in findings.values())
        failing = latest.failing_checks
        if failing:
            d.verified_facts.append(f"Failing checks on {latest.ref}: {failing}")
        if not findings and not failing:
            d.reasons.append(f"No current finding or failing check on {device_id} justifies contacting the employee.")

    d.allowed = not d.reasons
    return d
