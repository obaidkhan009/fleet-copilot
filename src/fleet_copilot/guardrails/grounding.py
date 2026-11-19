"""Post-hoc grounding verifier for final answers.

Deterministic checks run on every final answer before it is shown:

1. every ``[E#]`` citation resolves to a ledger record;
2. no identifier from outside the tenant appears (counted, never echoed);
3. **claim binding, per sentence**: a device id named in a sentence must be the
   subject of one of the records *that sentence* cites (a device-specific
   record, or an aggregate record whose ``matched`` list contains it). A real
   citation attached to a claim about a different device is a violation;
4. **numbers, per sentence**: every number in a cited sentence must appear in
   the records that sentence cites (or be a configured threshold, or appear in
   the user's question). Uncited sentences are checked against the whole ledger;
5. an answer produced after tool use must cite evidence.

A sentence without its own citation inherits the citations of its line (a
bullet or paragraph), so "- X: 9% free. Owner emp-1. [E3]" is checked as one
claim. On failure the graph sends the violations back to the agent for one
revision; if it still fails, the answer is shown with a visible warning (or
withheld entirely when it names foreign identifiers).

Limits (documented in the README): the checks are lexical. They catch invented,
misattributed and mis-copied facts, not a wrong *interpretation* of a correctly
cited number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..config import THRESHOLDS
from ..data.store import TenantView
from ..evidence import EvidenceLedger
from .action_policy import find_device_ids, find_employee_ids

CITATION = re.compile(r"\[(E\d+(?:\s*[,;]\s*E\d+)*)\]")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?")
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]|\.\d)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*_\-(\[`])")


@dataclass
class GroundingReport:
    ok: bool
    violations: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    mentioned_devices: list[str] = field(default_factory=list)
    mentioned_employees: list[str] = field(default_factory=list)
    foreign_entities: int = 0   # count only - foreign identifiers are never stored or echoed
    unsupported_numbers: int = 0
    misattributed_entities: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for m in CITATION.finditer(text):
        ids.extend(x.strip() for x in re.split(r"[,;]", m.group(1)))
    return list(dict.fromkeys(ids))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Remove things that look numeric but are not claims: citations, ids, dates, proposal ids."""
    text = CITATION.sub(" ", text)
    text = _DATE.sub(" ", text)
    text = re.sub(r"\b[A-Za-z0-9]{12}@\S+", " ", text)          # snapshot refs
    text = re.sub(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{12}\b", " ", text)  # device ids
    text = re.sub(r"\bemp-[a-z]+-\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\b[EP]\d+\b", " ", text)                     # evidence / proposal ids
    return text


def _numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(_strip_noise(text))


def _collect_numbers(obj: Any, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        if 0 < obj < 1:
            out.add(round(float(obj) * 100, 6))  # fractions are often stated as percentages
        return
    if isinstance(obj, str):
        for n in _NUMBER.findall(_DATE.sub(" ", obj)):
            out.add(float(n))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():   # keys carry facts too, e.g. {"macOS 14.6": 3}
            _collect_numbers(k, out)
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_numbers(v, out)


def _record_numbers(records: Iterable[dict]) -> set[float]:
    out: set[float] = set()
    for r in records:
        _collect_numbers(r.get("data"), out)
        _collect_numbers(r.get("claim"), out)
        _collect_numbers(r.get("args"), out)
    return out


def _always_allowed(question: str) -> set[float]:
    th = THRESHOLDS
    vals = {th.disk_low_free_pct, th.disk_critical_free_pct, th.ram_high_used_pct, th.sustained_fraction * 100,
            th.battery_cycle_limit, th.battery_eol_horizon_days, th.battery_capacity_drop_pct,
            th.min_snapshots_for_trend, 30.0, 0.0}
    vals |= {float(n) for n in _numbers_in(question)}
    return {float(v) for v in vals}


def _supported(num: str, pool: set[float]) -> bool:
    x = float(num)
    decimals = len(num.split(".")[1]) if "." in num else 0
    return any(round(y, decimals) == x or abs(y - x) < 1e-9 for y in pool)


def _units(text: str) -> list[tuple[str, list[str]]]:
    """Split into (sentence, citations-in-scope) units. Sentences without citations inherit their line's."""
    units = []
    for line in text.splitlines():
        if not line.strip():
            continue
        line_cites = extract_citations(line)
        for sent in _SENTENCE_SPLIT.split(line):
            if sent.strip():
                units.append((sent, extract_citations(sent) or line_cites))
    return units


def _redact(snippet: str, view: TenantView) -> str:
    for d in find_device_ids(snippet):
        if not view.has_device(d):
            snippet = re.sub(re.escape(d), "[id]", snippet, flags=re.I)
    for e in find_employee_ids(snippet):
        if not view.has_employee(e):
            snippet = re.sub(re.escape(e), "[id]", snippet, flags=re.I)
    snippet = " ".join(snippet.split())
    return snippet if len(snippet) <= 90 else snippet[:87] + "..."


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------

def verify_answer(text: str, ledger: EvidenceLedger, view: TenantView, used_tools: bool,
                  question: str = "") -> GroundingReport:
    rep = GroundingReport(ok=True)
    rep.cited_ids = extract_citations(text)
    devices, emps = find_device_ids(text), find_employee_ids(text)
    rep.mentioned_devices = sorted(d for d in devices if view.has_device(d))
    rep.mentioned_employees = sorted(e for e in emps if view.has_employee(e))
    # Identifiers outside the tenant are counted but never stored or echoed (not even in violations).
    for _ in devices - set(rep.mentioned_devices):
        rep.foreign_entities += 1
        rep.violations.append("The answer names a device id that is not in this organization; remove it.")
    for _ in emps - set(rep.mentioned_employees):
        rep.foreign_entities += 1
        rep.violations.append("The answer names an employee id that is not in this organization; remove it.")

    unknown = [c for c in rep.cited_ids if ledger.get(c) is None]
    if unknown:
        rep.violations.append(f"Citations {unknown} do not correspond to any evidence record.")

    covered, emps_covered = ledger.devices_covered(), ledger.employees_covered()
    rep.violations += [f"Device {d} is mentioned but no tool result is about it."
                       for d in rep.mentioned_devices if d not in covered]
    rep.violations += [f"Employee {e} is mentioned but no tool result is about them."
                       for e in rep.mentioned_employees if e not in emps_covered]

    allowed = _always_allowed(question)
    global_numbers = _record_numbers(ledger.records.values()) | allowed
    seen: set[str] = set()
    for sent, cites in _units(text):
        recs = [ledger.get(c) for c in cites if ledger.get(c)]
        # (3) claim binding
        if recs:
            subjects = {r["device_id"] for r in recs if r.get("device_id")}
            subjects |= {d for r in recs if not r.get("device_id") for d in (r.get("data") or {}).get("matched", [])}
            subject_emps = {r["employee_id"] for r in recs if r.get("employee_id")}
            for d in find_device_ids(sent):
                if view.has_device(d) and d not in subjects and d in covered:
                    rep.misattributed_entities += 1
                    rep.violations.append(f"\"{_redact(sent, view)}\": device {d} is not what {cites} is about; "
                                          "cite the evidence for that device.")
            for e in find_employee_ids(sent):
                if view.has_employee(e) and e not in subject_emps and e in emps_covered and not subjects & set(
                        view.devices_of(e)):
                    rep.misattributed_entities += 1
                    rep.violations.append(f"\"{_redact(sent, view)}\": employee {e} is not what {cites} is about.")
        # (4) numbers
        pool = (_record_numbers(recs) | allowed) if recs else global_numbers
        for n in _numbers_in(sent):
            if not _supported(n, pool):
                key = f"{n}|{sent}"
                if key in seen:
                    continue
                seen.add(key)
                rep.unsupported_numbers += 1
                where = f"the cited evidence {cites}" if recs else "any tool result"
                rep.violations.append(f"\"{_redact(sent, view)}\": the number {n} does not appear in {where}.")

    if used_tools and not rep.cited_ids and (rep.mentioned_devices or re.search(r"\d", _strip_noise(text))):
        rep.violations.append("The answer states facts from tool results but cites no evidence ids.")

    rep.ok = not rep.violations
    return rep


def sentence_units(text: str) -> list[tuple[str, list[str]]]:
    """Public view of how the verifier splits an answer into (sentence, citations-in-scope) claims."""
    return _units(text)
