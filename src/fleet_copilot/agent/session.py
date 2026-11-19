"""Session wrapper around the compiled graph (one LangGraph thread per session).

With a durable checkpointer (the CLI uses SQLite) a thread outlives the process:
an administrator can ask today, and a colleague can review and approve the
pending proposals tomorrow from a new process (``fleet-copilot approvals``).
Re-opening a thread is tenant-checked: a thread created for one organization
can never be listed, resumed or explained from another.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langgraph.types import Command

from .graph import Deps, build_graph


class ThreadAccessError(PermissionError):
    """The thread does not exist or belongs to another organization (same message for both)."""


@dataclass
class TurnResult:
    final: dict | None = None            # set when the turn completed
    approval_request: dict | None = None  # set when the graph is paused for approval

    @property
    def awaiting_approval(self) -> bool:
        return self.approval_request is not None


class CopilotSession:
    def __init__(self, deps: Deps, tenant_id: str, actor: str, thread_id: str | None = None,
                 must_exist: bool = False):
        deps.store.scoped(tenant_id)  # validate tenant up front (raises UnknownTenantError)
        self.deps = deps
        self.tenant_id = tenant_id
        self.actor = actor
        self.thread_id = thread_id or f"t-{uuid.uuid4().hex[:10]}"
        self.graph = build_graph(deps)
        self.config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 60}
        existing = self.graph.get_state(self.config).values
        if existing and existing.get("tenant_id") != tenant_id:
            deps.audit.record("thread_access_denied", tenant_id=tenant_id, actor=actor, thread_id=self.thread_id)
            raise ThreadAccessError(f"Thread {self.thread_id} was not found in your organization.")
        if must_exist and not existing:
            raise ThreadAccessError(f"Thread {self.thread_id} was not found in your organization.")
        deps.audit.record("session_resume" if existing else "session_start", tenant_id=tenant_id, actor=actor,
                          thread_id=self.thread_id, llm=getattr(deps.llm, "name", "?"))

    def _result(self, out: dict) -> TurnResult:
        intr = out.get("__interrupt__")
        if intr:
            return TurnResult(approval_request=intr[0].value)
        return TurnResult(final=out.get("final"))

    def ask(self, question: str) -> TurnResult:
        if self.pending():
            raise RuntimeError("A previous turn is awaiting approval; call resume() first.")
        out = self.graph.invoke({"question": question, "tenant_id": self.tenant_id, "actor": self.actor,
                                 "thread_id": self.thread_id}, self.config)
        return self._result(out)

    def resume(self, decisions: dict[str, dict]) -> TurnResult:
        """decisions: {proposal_id: {"approved": bool, "approver": str, "approvers": [str], "comment": str}}"""
        out = self.graph.invoke(Command(resume=decisions), self.config)
        return self._result(out)

    def pending(self) -> bool:
        return bool(self.graph.get_state(self.config).next)

    def pending_request(self) -> dict | None:
        """The approval request a paused thread is waiting on (None if the thread is not paused)."""
        for task in self.graph.get_state(self.config).tasks:
            for intr in task.interrupts:
                return intr.value
        return None

    @property
    def state(self) -> dict:
        return self.graph.get_state(self.config).values


def list_threads(deps: Deps, tenant_id: str) -> list[dict]:
    """Threads of one tenant, most recent first: id, last question, whether it awaits approval."""
    deps.store.scoped(tenant_id)
    graph = build_graph(deps)
    seen: dict[str, str] = {}
    for tup in graph.checkpointer.list(None):
        tid = tup.config["configurable"]["thread_id"]
        ts = tup.checkpoint.get("ts", "")
        if ts > seen.get(tid, ""):
            seen[tid] = ts
    out = []
    for tid, ts in sorted(seen.items(), key=lambda kv: kv[1], reverse=True):
        snap = graph.get_state({"configurable": {"thread_id": tid}})
        values = snap.values or {}
        if values.get("tenant_id") != tenant_id:
            continue   # other tenants' threads are invisible
        request = next((i.value for t in snap.tasks for i in t.interrupts), None)
        out.append({"thread_id": tid, "updated": ts, "question": values.get("question", ""),
                    "turns": len(values.get("questions", [])), "awaiting_approval": request is not None,
                    "proposals": request["proposals"] if request else []})
    return out
