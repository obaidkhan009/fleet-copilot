"""Deterministic pre-LLM input screening (defence in depth).

This is *not* the primary tenant-isolation control - that is structural (the
agent only ever holds a ``TenantView``). The guard exists to (1) refuse
obvious cross-tenant requests early with a clear message, (2) flag them as
security events in the audit log, and (3) stop blatant prompt-injection
attempts before they reach the model.

The guard uses a directory of *identifiers* of other tenants (names, device
ids, employee ids) purely for detection. Its refusal for a foreign device id
is the same "not found in your organization" message a non-existent id gets,
so the guard is not an existence oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 4000

INJECTION_PATTERNS = [
    r"\b(ignore|forget|disregard|override|bypass)\s+(all\s+|any\s+)?(of\s+)?(your|the|my|these|those|previous|prior|"
    r"above|earlier|existing|system)(\s+\w+)?\s+(instructions|guidance|rules|prompts?|guardrails|policies|constraints)\b",
    r"(reveal|print|show|repeat|output) (me )?(your|the) (system prompt|instructions|hidden prompt|initial prompt)",
    r"\byou are now (an?|the|my|acting|in|no longer)\b",
    r"\bfrom now on,? you (are|will|must)\b",
    r"\b(developer|god|jailbreak|dan) mode\b",
    r"pretend (you are|to be) (an? )?(admin|administrator|root|another tenant)",
    r"act as (an? )?(admin of|administrator of|super ?admin)",
    r"switch (to )?(tenant|company|organization)",
]
CROSS_TENANT_PATTERNS = [
    r"\b(all|every|other|another|each)\s+(tenants?|compan(y|ies)|organi[sz]ations?|customers?)('s|s')?\b",
    r"\b(across|between)\s+(tenants|companies|organi[sz]ations|customers)\b",
]
APPROVAL_BYPASS_PATTERNS = [
    r"without (my |human |admin |any |an? )?(approval|confirmation|review|asking)",
    r"(skip|bypass|disable|turn off) (the )?(approval|human review|confirmation|guardrails?)",
    r"auto[- ]?approve",
    r"(execute|run|submit) (it|them|everything|all)? ?(right away|immediately|directly) without",
    r"don'?t ask (me )?(for )?(approval|confirmation)",
]


@dataclass
class GuardVerdict:
    allowed: bool
    category: str | None = None
    message: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "category": self.category, "message": self.message, "flags": self.flags}


@dataclass(frozen=True)
class TenantDirectory:
    """Identifiers per tenant, used only for cross-tenant *detection*."""
    names: dict[str, str]                 # tenant_id -> display name
    device_ids: dict[str, frozenset[str]]  # tenant_id -> device ids
    employee_ids: dict[str, frozenset[str]]


def check_input(text: str, tenant_id: str, directory: TenantDirectory) -> GuardVerdict:
    lowered = text.lower()
    own_name = directory.names.get(tenant_id, tenant_id)
    if not text.strip():
        return GuardVerdict(False, "empty", f"Please ask a question about {own_name}'s fleet.", ["empty"])
    if len(text) > MAX_QUESTION_CHARS:
        return GuardVerdict(False, "too_long", f"That request is too long (limit {MAX_QUESTION_CHARS} characters). "
                                               "Please shorten it.", ["too_long"])

    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return GuardVerdict(False, "prompt_injection",
                                "I can't follow instructions that try to change my rules or scope. I can help you "
                                f"with questions and actions about {own_name}'s fleet.", ["prompt_injection"])

    for other, name in directory.names.items():
        if other == tenant_id:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered) or other.lower() in lowered:
            return GuardVerdict(False, "cross_tenant",
                                f"Your session is scoped to {own_name}. I can't access or act on data belonging to "
                                f"other organizations, including {name}.", ["cross_tenant_name"])

    for pat in CROSS_TENANT_PATTERNS:
        if re.search(pat, lowered):
            return GuardVerdict(False, "cross_tenant",
                                f"Your session is scoped to {own_name}. I can't access, list or compare data from "
                                "other organizations.", ["cross_tenant_scope"])

    raw = set(re.findall(r"[A-Za-z0-9-]+", text))
    tokens = {t.upper() for t in raw} | {t.lower() for t in raw}   # ids are matched case-insensitively
    for other in directory.names:
        if other == tenant_id:
            continue
        hits = sorted((directory.device_ids.get(other, frozenset()) | directory.employee_ids.get(other, frozenset())) & tokens)
        if hits:
            return GuardVerdict(False, "cross_tenant",
                                f"{', '.join(hits)} was not found in {own_name}'s fleet. I can only access "
                                f"{own_name} devices and employees.", ["cross_tenant_identifier"])

    flags = []
    for pat in APPROVAL_BYPASS_PATTERNS:
        if re.search(pat, lowered):
            flags.append("approval_bypass_requested")
            break
    return GuardVerdict(True, None, None, flags)
