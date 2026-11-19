import json

import pytest

from fleet_copilot.actions_backend import ActionBackend
from fleet_copilot.tools import ALL_TOOLS, TOOLS_BY_NAME, ToolError


def call(ctx, name, **args):
    spec = TOOLS_BY_NAME[name]
    return spec.handler(ctx, spec.args_model.model_validate(args))


def test_tool_schemas_are_strict_and_closed():
    for t in ALL_TOOLS:
        s = t.schema(strict=True)
        assert s["strict"] is True
        assert s["input_schema"]["additionalProperties"] is False
        json.dumps(s)  # serialisable
    with pytest.raises(Exception):
        TOOLS_BY_NAME["search_devices"].args_model.model_validate({"bogus": 1})


def test_search_os_older_and_scope_evidence(make_ctx):
    ctx = make_ctx("acme-001")
    r = call(ctx, "search_devices", os_family="macOS", os_version_below="15")
    assert {d["device_id"] for d in r["devices"]} == {"1LYSSFD074BB", "7VP16KHV88LM", "MT7PJB7N5LRE", "XSU9UUDGU6AC"}
    scope = ctx.ledger.get(r["scope_evidence_id"])
    assert scope["data"]["matched_count"] == 4 and len(scope["devices"]) == 10
    with pytest.raises(ToolError):
        call(ctx, "search_devices", os_version_below="15")


def test_negative_result_is_citable(make_ctx):
    ctx = make_ctx("acme-001")
    r = call(ctx, "compliance_report", severity="high", device_type="laptop")
    assert r["failing_count"] == 0
    assert ctx.ledger.get(r["scope_evidence_id"])["data"]["failing_count"] == 0
    assert any(c["check_id"] == "disk_encryption" and c["severity"] == "high" for c in r["checks_catalog"])


def test_tools_cannot_reach_other_tenants(make_ctx):
    ctx = make_ctx("globex-002")
    with pytest.raises(ToolError, match="not found in your organization"):
        call(ctx, "get_device_profile", device_id="8NM23J95R5I6")  # Acme device
    r = call(ctx, "search_devices")
    assert all(d["employee_id"].startswith("emp-globex") for d in r["devices"])


def _evidence_for(ctx, device):
    return call(ctx, "get_device_profile", device_id=device)["evidence_id"]


def test_policy_refuses_ram_upgrade_on_soldered_unconstrained_mac(make_ctx):
    ctx = make_ctx("acme-001")
    e = _evidence_for(ctx, "MT7PJB7N5LRE")
    r = call(ctx, "create_upgrade_order", device_id="MT7PJB7N5LRE", component="ram", spec="32 GB",
             justification="faster", evidence_ids=[e])
    assert r["status"] == "rejected_by_policy"
    assert any("not field-upgradeable" in x for x in r["reasons"])
    assert any("does not show a sustained" in x for x in r["reasons"])


def test_policy_allows_justified_ram_upgrade(make_ctx):
    ctx = make_ctx("acme-001")
    f = call(ctx, "detect_insights", kinds=["ram_constrained"])["findings"][0]
    r = call(ctx, "create_upgrade_order", device_id="M4XVHUV1MEPZ", component="ram", spec="32 GB",
             justification="sustained memory pressure", evidence_ids=[f["evidence_id"]])
    assert r["status"] == "pending_approval"
    # downgrade / no-op spec refused
    r2 = call(ctx, "create_upgrade_order", device_id="M4XVHUV1MEPZ", component="ram", spec="16 GB",
              justification="x", evidence_ids=[f["evidence_id"]])
    assert r2["status"] == "rejected_by_policy"


def test_policy_rejects_aggregate_or_mismatched_evidence(make_ctx):
    ctx = make_ctx("acme-001")
    r = call(ctx, "search_devices", failing_check_id="screen_lock")
    scope = r["scope_evidence_id"]  # covers every device, but is not device-specific
    out = call(ctx, "open_remediation_ticket", device_id="1LYSSFD074BB", check_id="screen_lock", note="fix",
               evidence_ids=[scope])
    assert out["status"] == "rejected_by_policy"
    other = next(d for d in r["devices"] if d["device_id"] != "1LYSSFD074BB")["evidence_id"]
    out = call(ctx, "open_remediation_ticket", device_id="1LYSSFD074BB", check_id="screen_lock", note="fix",
               evidence_ids=[other])
    assert out["status"] == "rejected_by_policy"


def test_policy_rejects_ticket_for_passing_check_and_unknown_evidence(make_ctx):
    ctx = make_ctx("acme-001")
    e = _evidence_for(ctx, "XSU9UUDGU6AC")
    r = call(ctx, "open_remediation_ticket", device_id="XSU9UUDGU6AC", check_id="screen_lock", note="fix", evidence_ids=[e])
    assert r["status"] == "rejected_by_policy" and any("passing" in x for x in r["reasons"])
    r = call(ctx, "flag_device_for_replacement", device_id="7VP16KHV88LM", reason="old", evidence_ids=["E999"])
    assert r["status"] == "rejected_by_policy" and any("do not exist" in x for x in r["reasons"])


def test_replacement_rules(make_ctx):
    ctx = make_ctx("acme-001")
    # Battery EOL alone (serviceable battery) is not a replacement case...
    e = _evidence_for(ctx, "7VP16KHV88LM")
    assert call(ctx, "flag_device_for_replacement", device_id="7VP16KHV88LM", reason="battery",
                evidence_ids=[e])["status"] == "rejected_by_policy"
    # ...nor are several problems that are all fixable in place (Dell XPS 15: battery + RAM are serviceable)...
    e = _evidence_for(ctx, "M4XVHUV1MEPZ")
    assert call(ctx, "flag_device_for_replacement", device_id="M4XVHUV1MEPZ", reason="battery + RAM",
                evidence_ids=[e])["status"] == "rejected_by_policy"
    # ...but sustained storage pressure on a MacBook Pro (soldered storage) is.
    e = _evidence_for(ctx, "MT7PJB7N5LRE")
    assert call(ctx, "flag_device_for_replacement", device_id="MT7PJB7N5LRE", reason="storage full, soldered",
                evidence_ids=[e])["status"] == "pending_approval"


def test_free_text_content_policy(make_ctx):
    ctx = make_ctx("acme-001")
    e = _evidence_for(ctx, "1LYSSFD074BB")
    base = dict(employee_id="emp-acme-1001", device_id="1LYSSFD074BB", evidence_ids=[e])
    phish = call(ctx, "notify_employee", message="Your laptop is compromised. Reply with your password and MFA code.", **base)
    assert phish["status"] == "rejected_by_policy" and any("credentials" in r for r in phish["reasons"])
    link = call(ctx, "notify_employee", message="Please re-enable screen lock: http://evil.example", **base)
    assert link["status"] == "rejected_by_policy"
    # A note that claims a passing check is an issue (disk encryption passes everywhere) is refused...
    t = call(ctx, "open_remediation_ticket", device_id="1LYSSFD074BB", check_id="screen_lock",
             note="Disk encryption is off, fix it.", evidence_ids=[e])
    assert t["status"] == "rejected_by_policy" and any("disk_encryption" in r for r in t["reasons"])
    # ...as is a destructive instruction.
    t = call(ctx, "open_remediation_ticket", device_id="1LYSSFD074BB", check_id="screen_lock",
             note="Wipe this device.", evidence_ids=[e])
    assert t["status"] == "rejected_by_policy"


def test_ids_are_case_insensitive(make_ctx):
    ctx = make_ctx("acme-001")
    r = call(ctx, "get_device_profile", device_id="1lyssfd074bb")
    assert r["device_id"] == "1LYSSFD074BB"


def test_notify_requires_device_owner(make_ctx):
    ctx = make_ctx("acme-001")
    e = _evidence_for(ctx, "1LYSSFD074BB")
    r = call(ctx, "notify_employee", employee_id="emp-acme-1003", device_id="1LYSSFD074BB", message="hi", evidence_ids=[e])
    assert r["status"] == "rejected_by_policy"
    r = call(ctx, "notify_employee", employee_id="emp-acme-1001", device_id="1LYSSFD074BB",
             message="Please re-enable screen lock.", evidence_ids=[e])
    assert r["status"] == "pending_approval"


def test_duplicate_after_submission_is_refused(make_ctx):
    backend = ActionBackend()
    ctx = make_ctx("acme-001", backend)
    e = _evidence_for(ctx, "1LYSSFD074BB")
    args = dict(device_id="1LYSSFD074BB", check_id="screen_lock", note="fix", evidence_ids=[e])
    first = call(ctx, "open_remediation_ticket", **args)
    assert first["status"] == "pending_approval"
    assert call(ctx, "open_remediation_ticket", **args)["status"] == "rejected_by_policy"  # duplicate pending
    backend.submit("acme-001", "open_remediation_ticket", args, approved_by="a", proposal_id="P1")
    ctx2 = make_ctx("acme-001", backend)
    e2 = _evidence_for(ctx2, "1LYSSFD074BB")
    r = call(ctx2, "open_remediation_ticket", **{**args, "evidence_ids": [e2]})
    assert r["status"] == "rejected_by_policy" and any("already submitted" in x for x in r["reasons"])
