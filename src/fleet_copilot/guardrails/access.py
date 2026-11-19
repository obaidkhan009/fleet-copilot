"""Who may approve what: roles per tenant and the two-person rule.

Approval is checked against an access directory (tenant -> identity -> roles):

* ``approver``  may approve low- and medium-risk actions on their own;
* ``viewer``    may ask questions and see proposals, but cannot approve;
* high-risk actions (``flag_device_for_replacement``) need **two distinct
  approvers** (four-eyes principle).

Identities are only meaningful inside their tenant: an Acme identity has no
roles in Globex, so cross-tenant approval is impossible by construction.

The directory stands in for the platform's IdP / RBAC service. By default it
holds demo identities per tenant (``admin@<tenant>`` and ``secops@<tenant>`` are
approvers, ``analyst@<tenant>`` is a viewer); ``FLEET_ACCESS_FILE`` can point
to a YAML file with the same shape::

    acme-001:
      alice@acme-001: [approver]
      bob@acme-001: [viewer]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

REQUIRED_APPROVERS = {"low": 1, "medium": 1, "high": 2}
APPROVER_ROLE = "approver"


def _norm(identity: str | None) -> str:
    return (identity or "").strip().lower()


@dataclass(frozen=True)
class ApprovalCheck:
    ok: bool
    approvers: list[str] = field(default_factory=list)   # valid, distinct approvers that count
    reason: str | None = None


@dataclass(frozen=True)
class AccessDirectory:
    roles: Mapping[str, Mapping[str, frozenset[str]]]

    @classmethod
    def demo(cls, tenants: Iterable[str]) -> "AccessDirectory":
        return cls({t: {f"admin@{t}": frozenset({APPROVER_ROLE}), f"secops@{t}": frozenset({APPROVER_ROLE}),
                        f"analyst@{t}": frozenset({"viewer"})} for t in tenants})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AccessDirectory":
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls({t: {_norm(i): frozenset(r or []) for i, r in (ids or {}).items()} for t, ids in raw.items()})

    @classmethod
    def load(cls, tenants: Iterable[str]) -> "AccessDirectory":
        path = os.environ.get("FLEET_ACCESS_FILE")
        return cls.from_yaml(path) if path else cls.demo(tenants)

    def roles_of(self, tenant_id: str, identity: str) -> frozenset[str]:
        return frozenset(self.roles.get(tenant_id, {}).get(_norm(identity), frozenset()))

    def can_approve(self, tenant_id: str, identity: str) -> bool:
        return APPROVER_ROLE in self.roles_of(tenant_id, identity)

    def check(self, tenant_id: str, risk: str, identities: Iterable[str]) -> ApprovalCheck:
        """Do these identities together satisfy the approval rule for an action of this risk?"""
        given = list(dict.fromkeys(_norm(i) for i in identities if _norm(i)))
        valid = [i for i in given if self.can_approve(tenant_id, i)]
        need = REQUIRED_APPROVERS.get(risk, 2)
        if not given:
            return ApprovalCheck(False, [], "no approver identity")
        if len(valid) < len(given):
            return ApprovalCheck(False, valid, f"{len(given) - len(valid)} identity(ies) lack the approver role "
                                               f"in this organization")
        if len(valid) < need:
            return ApprovalCheck(False, valid, f"{risk}-risk actions need {need} distinct approvers "
                                               f"(got {len(valid)})")
        return ApprovalCheck(True, valid)
