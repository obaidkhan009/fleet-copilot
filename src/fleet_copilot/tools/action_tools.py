"""State-changing tools. Calling one only creates a *proposal*.

Nothing here touches a downstream system. A proposal that passes the evidence
policy is queued as ``pending_approval``; the graph then pauses at the human
approval gate, and only the executor node (after an explicit approval, and
after re-running the policy) submits it to the backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..guardrails.action_policy import evaluate
from .registry import ArgsBase, ToolContext, ToolSpec

EVIDENCE_FIELD = Field(min_length=1, description="Evidence ids (e.g. ['E4']) from read tools that justify this action.")


class _Args(ArgsBase):
    pass


class CreateUpgradeOrderArgs(_Args):
    device_id: str
    component: Literal["ram", "storage", "battery"]
    spec: str = Field(description="Catalogue spec: ram '16 GB'|'32 GB'|'64 GB'; storage '512 GB'|'1 TB'|'2 TB'; "
                                  "battery 'OEM replacement battery'.")
    justification: str = Field(description="Why, in one or two sentences, referencing the finding.")
    evidence_ids: list[str] = EVIDENCE_FIELD


class OpenRemediationTicketArgs(_Args):
    device_id: str
    check_id: str = Field(description="Failing compliance check id (e.g. 'screen_lock'), or "
                                      "'unapproved_software:<App Name>' for unapproved software removal.")
    note: str = Field(description="Remediation instructions for the IT technician.")
    evidence_ids: list[str] = EVIDENCE_FIELD


class FlagDeviceForReplacementArgs(_Args):
    device_id: str
    reason: str
    evidence_ids: list[str] = EVIDENCE_FIELD


class NotifyEmployeeArgs(_Args):
    employee_id: str
    device_id: str = Field(description="The device the message is about (must be assigned to the employee).")
    message: str = Field(description="Plain-language message to the employee (max 1000 chars).")
    evidence_ids: list[str] = EVIDENCE_FIELD


def _propose(tool: str):
    def handler(ctx: ToolContext, args: BaseModel) -> dict:
        a = args.model_dump()
        decision = evaluate(tool, a, view=ctx.view, ledger=ctx.ledger, th=ctx.thresholds,
                            backend=ctx.backend, pending=ctx.proposals)
        pid = f"P{len(ctx.proposals) + 1}"
        proposal = {
            "id": pid,
            "tool": tool,
            "args": a,
            "tenant_id": ctx.tenant_id,
            "status": "pending_approval" if decision.allowed else "rejected_by_policy",
            "policy": decision.to_dict(),
        }
        ctx.proposals.append(proposal)
        if decision.allowed:
            return {"status": "pending_approval", "proposal_id": pid, "risk": decision.risk,
                    "verified_facts": decision.verified_facts,
                    "note": "NOT executed. Queued for explicit approval by the IT administrator."}
        return {"status": "rejected_by_policy", "proposal_id": pid, "reasons": decision.reasons,
                "note": "This action will not be offered for approval. Explain the refusal to the user."}
    return handler


ACTION_TOOLS = [
    ToolSpec("create_upgrade_order",
             "PROPOSE a hardware upgrade order (RAM, storage or battery). Requires evidence of a sustained constraint "
             "and a field-serviceable component. Never executes directly; queues for admin approval.",
             CreateUpgradeOrderArgs, "action", _propose("create_upgrade_order")),
    ToolSpec("open_remediation_ticket",
             "PROPOSE an IT remediation ticket for a currently failing compliance check or installed unapproved "
             "software. Never executes directly; queues for admin approval.",
             OpenRemediationTicketArgs, "action", _propose("open_remediation_ticket")),
    ToolSpec("flag_device_for_replacement",
             "PROPOSE flagging a device for replacement. Requires a sustained constraint on a non-upgradeable "
             "component or >= 2 hardware findings. Never executes directly; queues for admin approval.",
             FlagDeviceForReplacementArgs, "action", _propose("flag_device_for_replacement")),
    ToolSpec("notify_employee",
             "PROPOSE a notification to the employee assigned to a device about a finding on that device. Never "
             "executes directly; queues for admin approval.",
             NotifyEmployeeArgs, "action", _propose("notify_employee")),
]
