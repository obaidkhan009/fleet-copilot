"""Traceability report for an answer: sentence -> evidence id -> tool call -> raw NDJSON line.

``fleet-copilot explain`` uses this to show, for the last answer on a thread,
exactly which telemetry lines back each sentence. Snapshot refs
(``<device_id>@<timestamp>``) are resolved to line numbers in the dataset file,
so a reviewer can open the raw record (``sed -n '<line>p' <dataset>``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .data.models import Snapshot
from .guardrails.grounding import sentence_units

MAX_REFS_SHOWN = 10   # an aggregate's matched devices are all shown; other refs are summarised


@lru_cache(maxsize=4)
def _line_index(path: str) -> dict[str, tuple[int, dict]]:
    index: dict[str, tuple[int, dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                snap = Snapshot.from_record(json.loads(line))
                index[snap.ref] = (lineno, _excerpt(snap))
    return index


def _excerpt(s: Snapshot) -> dict:
    out = {"collected_at": s.ref.split("@", 1)[1], "model": s.model_name, "os": f"{s.os_family} {s.os_version}",
           "mem_used_pct": s.mem_used_pct, "disk_free_pct": s.disk_free_pct,
           "failing_checks": [c for c, _ in s.failing_checks]}
    if s.battery:
        out["battery"] = {"condition": s.battery.condition, "cycle_count": s.battery.cycle_count}
    return out


def explain_answer(final: dict, evidence: dict[str, dict], data_path: str | Path) -> dict:
    """Build the trace for one completed turn (``final`` = the turn's final payload)."""
    index = _line_index(str(data_path))
    units = []
    for sentence, cites in sentence_units(final.get("answer", "")):
        links = []
        for eid in cites:
            rec = evidence.get(eid)
            if rec is None:
                links.append({"id": eid, "missing": True})
                continue
            refs = rec.get("snapshot_refs", [])
            matched = set((rec.get("data") or {}).get("matched", []))
            if not rec.get("device_id") and "matched" in (rec.get("data") or {}):
                # aggregate record: show the devices that matched; the rest were checked and did not match
                shown = [r for r in refs if r.split("@", 1)[0] in matched]
                note = f"{len(refs) - len(shown)} other device(s) checked, did not match"
            else:
                shown, note = refs, None
            lines = []
            for ref in shown[:MAX_REFS_SHOWN]:
                lineno, excerpt = index.get(ref, (None, None))
                lines.append({"ref": ref, "line": lineno, "excerpt": excerpt})
            # For aggregate records the numbers a sentence quotes (counts, per-check totals) live in the record's
            # data, not in its snapshot lines - show that data so every number can be checked.
            data = None
            if not rec.get("device_id"):
                data = {k: v for k, v in (rec.get("data") or {}).items() if k != "matched"}
            links.append({"id": eid, "tool": rec["tool"], "args": rec.get("args", {}), "claim": rec["claim"],
                          "device_id": rec.get("device_id"), "aggregate": not rec.get("device_id"),
                          "lines": lines, "more_refs": max(0, len(shown) - MAX_REFS_SHOWN), "note": note,
                          "data": data})
        units.append({"sentence": sentence.strip(), "citations": links})
    g = final.get("grounding") or {}
    return {"question": (final.get("plan") or {}).get("goal", ""), "answer": final.get("answer", ""),
            "grounding_ok": g.get("ok", True), "violations": g.get("violations", []),
            "data_file": str(data_path), "units": units}
