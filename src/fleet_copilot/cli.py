"""Command-line interface.

    fleet-copilot chat      --tenant acme-001                  interactive session with approval prompts
    fleet-copilot ask       --tenant acme-001 "question"       one-shot (--approve prompt|none|all|later|P1,P2)
    fleet-copilot approvals list   --tenant acme-001           conversations waiting for approval
    fleet-copilot approvals resume --tenant acme-001 --thread t-...   review and decide later, from any process
    fleet-copilot explain   --tenant acme-001 [--thread t-...] sentence -> evidence -> tool call -> NDJSON line
    fleet-copilot insights  --tenant acme-001                  run the trend detectors directly (no LLM)
    fleet-copilot audit     verify                             verify the hash chain and head anchor
    fleet-copilot audit     tail --tenant acme-001             recent events for one tenant (--all: operator view)

Conversations are checkpointed in var/checkpoints.sqlite, so a thread survives the process: pass --thread to
continue it, or approve its pending proposals later with `approvals resume`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .actions_backend import ActionBackend
from .audit import AuditIntegrityError, AuditLog
from .config import COMPANY_NAMES, DATA_PATH, THRESHOLDS, VAR_DIR
from .data import TelemetryStore, UnknownTenantError

console = Console()


def _default_llm() -> str:
    return "claude" if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) else "offline"


def _deps(llm_name: str):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from .agent.graph import Deps
    from .llm import make_backend
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(VAR_DIR / "checkpoints.sqlite", check_same_thread=False))
    return Deps(store=TelemetryStore.load(DATA_PATH), llm=make_backend(llm_name),
                audit=AuditLog(VAR_DIR / "audit.jsonl"), backend=ActionBackend(VAR_DIR / "outbox.jsonl"),
                checkpointer=saver)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_final(final: dict, show_trace: bool = False, thread_id: str | None = None) -> None:
    console.print(Panel(Markdown(final["answer"] or "_(empty)_"), title="Fleet Copilot", border_style="cyan"))
    if final["evidence"]:
        t = Table(title="Evidence", show_lines=False, header_style="bold")
        for col in ("id", "tool", "device", "claim", "snapshot refs"):
            t.add_column(col, overflow="fold")
        for e in final["evidence"]:
            refs = e["snapshot_refs"]
            ref_txt = ", ".join(r.split("@", 1)[1] if "@" in r else r for r in refs[:2]) + (f" (+{len(refs) - 2})" if len(refs) > 2 else "")
            t.add_row(e["id"], e["tool"], e.get("device_id") or f"{len(e['devices'])} devices", e["claim"], ref_txt)
        console.print(t)
    if final["proposals"]:
        t = Table(title="Proposals", header_style="bold")
        for col in ("id", "action", "target", "status", "detail"):
            t.add_column(col, overflow="fold")
        for p in final["proposals"]:
            detail = (p.get("execution", {}).get("external_id") or (p.get("decision") or {}).get("reason")
                      or "; ".join(p["policy"]["reasons"]) or ", ".join(p["args"].get("evidence_ids", [])))
            t.add_row(p["id"], p["tool"], _target(p), p["status"], detail)
        console.print(t)
    g = final.get("grounding") or {}
    u = final.get("usage") or {}
    meta = [f"intent={final.get('plan', {}).get('intent', '-')}", f"steps={final.get('steps', 0)}",
            f"grounded={'yes' if g.get('ok', True) else 'NO'}"]
    if thread_id:
        meta.append(f"thread={thread_id}")
    if u:
        meta.append(f"tokens in/out={u.get('input_tokens', 0)}/{u.get('output_tokens', 0)} (cache read {u.get('cache_read_input_tokens', 0)})")
    console.print("[dim]" + "  ".join(meta) + "[/dim]")
    if show_trace:
        console.print_json(json.dumps({"plan": final.get("plan"), "trace": final.get("trace")}, default=str))


def _target(p: dict) -> str:
    a = p["args"]
    extra = a.get("check_id") or (f"{a.get('component')} {a.get('spec')}" if a.get("component") else "") or a.get("employee_id", "")
    return f"{a.get('device_id')} {extra}".strip()


def render_approval(req: dict) -> None:
    console.print(Panel(Markdown(req.get("answer") or ""), title="Draft answer", border_style="cyan"))
    for p in req["proposals"]:
        facts = "\n".join(f"  • {f}" for f in p["policy"]["verified_facts"]) or "  • (none)"
        risk = p["policy"]["risk"]
        body = (f"[bold]{p['tool']}[/bold]  risk={risk}" + ("  [yellow](needs two approvers)[/yellow]" if risk == "high" else "")
                + f"\nargs: {json.dumps({k: v for k, v in p['args'].items() if k != 'evidence_ids'})}\n"
                f"evidence: {', '.join(p['args'].get('evidence_ids', []))}\n"
                f"verified by policy from telemetry:\n{facts}")
        console.print(Panel(body, title=f"Approval required: {p['id']}", border_style="yellow"))


def collect_decisions(req: dict, actor: str, mode: str | None, second_approver: str | None = None) -> dict:
    """Turn a CLI choice into resume decisions. High-risk proposals also carry a second approver."""
    ids = [p["id"] for p in req["proposals"]]
    interactive = mode is None
    ans = Prompt.ask("Approve which proposals? ([bold]all[/bold] / [bold]none[/bold] / comma-separated ids)",
                     default="none") if interactive else mode
    ans = ans.strip().lower()
    chosen = set(ids) if ans == "all" else set() if ans in ("none", "") else {x.strip().upper() for x in ans.split(",")}
    high = [p["id"] for p in req["proposals"] if p["id"] in chosen and p["policy"]["risk"] == "high"]
    if high and not second_approver and interactive:
        second_approver = Prompt.ask(f"{', '.join(high)} high risk: second approver identity (blank = skip)",
                                     default="") or None
    out = {}
    for p in req["proposals"]:
        dec = {"approved": p["id"] in chosen, "approver": actor, "comment": "via CLI"}
        if p["id"] in high and second_approver:
            dec["approvers"] = [second_approver]
        out[p["id"]] = dec
    return out


def _drive(session, res, args, mode):
    """Run approval rounds until the turn completes, or stop if approval is deferred."""
    while res.awaiting_approval:
        if mode == "later":
            if not getattr(args, "json", False):
                render_approval(res.approval_request)
            console.print(f"[yellow]Awaiting approval.[/yellow] Thread [bold]{session.thread_id}[/bold]. Decide later with:\n"
                          f"  fleet-copilot approvals resume --tenant {session.tenant_id} --thread {session.thread_id}")
            return None
        if not getattr(args, "json", False):
            render_approval(res.approval_request)
        res = session.resume(collect_decisions(res.approval_request, args.actor, mode,
                                               getattr(args, "second_approver", None)))
    return res


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_chat(args) -> int:
    from .agent.session import CopilotSession, TurnResult
    deps = _deps(args.llm)
    session = CopilotSession(deps, args.tenant, args.actor, thread_id=args.thread, must_exist=bool(args.thread))
    name = COMPANY_NAMES.get(args.tenant, args.tenant)
    console.print(f"[bold]Fleet Copilot[/bold] - {name} ({args.tenant}) - backend: {deps.llm.name} - "
                  f"thread {session.thread_id}\nType a question, or 'exit'.")
    if session.pending():
        res = _drive(session, TurnResult(approval_request=session.pending_request()),
                     args, None)
        render_final(res.final, args.trace, session.thread_id)
    while True:
        try:
            q = Prompt.ask("[bold green]you[/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("exit", "quit", ":q"):
            break
        if not q.strip():
            continue
        res = _drive(session, session.ask(q), args, None)
        render_final(res.final, args.trace, session.thread_id)
    return 0


def cmd_ask(args) -> int:
    from .agent.session import CopilotSession
    deps = _deps(args.llm)
    session = CopilotSession(deps, args.tenant, args.actor, thread_id=args.thread, must_exist=bool(args.thread))
    if session.pending():
        console.print(f"[red]Thread {session.thread_id} is awaiting approval; resolve it first with "
                      f"`fleet-copilot approvals resume`.[/red]")
        return 2
    mode = args.approve if args.approve != "prompt" else (None if sys.stdin.isatty() else "none")
    res = _drive(session, session.ask(args.question), args, mode)
    if res is None:
        return 0
    if args.json:
        print(json.dumps({"thread_id": session.thread_id, **res.final}, indent=2, default=str))
    else:
        render_final(res.final, args.trace, session.thread_id)
    return 0


def cmd_approvals(args) -> int:
    from .agent.session import CopilotSession, TurnResult, list_threads
    deps = _deps(args.llm)
    if args.action == "list":
        threads = [t for t in list_threads(deps, args.tenant) if t["awaiting_approval"] or args.all_threads]
        if not threads:
            console.print("No conversations are waiting for approval.")
            return 0
        t = Table(title=f"{'Conversations' if args.all_threads else 'Awaiting approval'} - {args.tenant}", header_style="bold")
        for col in ("thread", "updated", "question", "pending proposals"):
            t.add_column(col, overflow="fold")
        for th in threads:
            props = "\n".join(f"{p['id']} {p['tool']} {_target(p)} (risk {p['policy']['risk']})" for p in th["proposals"]) or "-"
            t.add_row(th["thread_id"], th["updated"][:19], th["question"], props)
        console.print(t)
        return 0
    # resume
    if not args.thread:
        console.print("[red]--thread is required (see `fleet-copilot approvals list`).[/red]")
        return 2
    session = CopilotSession(deps, args.tenant, args.actor, thread_id=args.thread, must_exist=True)
    req = session.pending_request()
    if req is None:
        console.print(f"Thread {args.thread} is not waiting for approval.")
        return 0
    mode = args.approve if args.approve != "prompt" else (None if sys.stdin.isatty() else "none")
    res = _drive(session, TurnResult(approval_request=req), args, mode)
    render_final(res.final, args.trace, session.thread_id)
    return 0


def cmd_explain(args) -> int:
    from .agent.session import CopilotSession, list_threads
    from .explain import explain_answer
    deps = _deps("offline")   # explaining reads checkpointed state only; no model calls
    thread = args.thread
    if not thread:
        done = [t for t in list_threads(deps, args.tenant) if not t["awaiting_approval"]]
        if not done:
            console.print("No completed conversations for this organization yet.")
            return 0
        thread = done[0]["thread_id"]
    session = CopilotSession(deps, args.tenant, args.actor, thread_id=thread, must_exist=True)
    state = session.state
    if session.pending() or not state.get("final"):
        console.print(f"Thread {thread} has no completed answer yet (it may be awaiting approval).")
        return 0
    report = explain_answer(state["final"], state.get("evidence", {}), DATA_PATH)
    report["question"] = state.get("question", "")
    if args.json:
        print(json.dumps({"thread_id": thread, **report}, indent=2, default=str))
        return 0
    console.print(f"[bold]Thread[/bold] {thread}   [bold]Question[/bold] {report['question']}")
    console.print(f"[bold]Grounding verifier[/bold]: {'[green]passed[/green]' if report['grounding_ok'] else '[red]failed[/red]'}"
                  f"   data file: {report['data_file']}")
    for i, u in enumerate(report["units"], 1):
        lines = [f"[bold cyan]{i}. {u['sentence']}[/bold cyan]"]
        if not u["citations"]:
            lines.append("   [dim](no citation - statement without telemetry facts)[/dim]")
        for c in u["citations"]:
            if c.get("missing"):
                lines.append(f"   [red]{c['id']}: not in the evidence ledger[/red]")
                continue
            kind = "aggregate" if c["aggregate"] else c["device_id"]
            lines.append(f"   [bold]{c['id']}[/bold] <- {c['tool']}({json.dumps(c['args'])}) [{kind}]: {c['claim']}")
            if c.get("data"):
                lines.append(f"      data: {escape(json.dumps(c['data'], default=str))}")
            for ln in c["lines"]:
                ex = json.dumps(ln["excerpt"]) if ln["excerpt"] else ""
                lines.append(f"      line {ln['line']}  {ln['ref']}  {ex}")
            if c["more_refs"]:
                lines.append(f"      ... +{c['more_refs']} more snapshot refs")
            if c.get("note"):
                lines.append(f"      ({c['note']})")
        console.print("\n".join(lines))
    console.print(f"[dim]Open a raw record: sed -n '<line>p' {report['data_file']}[/dim]")
    return 0


def cmd_insights(args) -> int:
    from .analytics import run_detectors
    view = TelemetryStore.load(DATA_PATH).scoped(args.tenant)
    t = Table(title=f"Insights - {COMPANY_NAMES.get(args.tenant, args.tenant)} (as of {view.as_of:%Y-%m-%d})")
    for col in ("severity", "kind", "device", "finding"):
        t.add_column(col, overflow="fold")
    for f in run_detectors(view, THRESHOLDS):
        if f.severity != "info":
            t.add_row(f.severity, f.kind, f.device_id, f.title)
    console.print(t)
    return 0


def cmd_audit(args) -> int:
    log = AuditLog(VAR_DIR / "audit.jsonl", strict=False)
    if args.action == "verify":
        ok, bad = log.verify()
        console.print(f"{len(log.entries)} entries - chain " + ("[green]OK[/green]" if ok else f"[red]BROKEN at seq {bad}[/red]"))
        return 0 if ok else 1
    if not args.tenant and not args.all:
        console.print("[red]audit tail needs --tenant <id> (tenant admins see only their tenant) or --all "
                      "(platform operator view).[/red]")
        return 2
    entries = [e for e in log.entries if args.all or e["tenant_id"] == args.tenant]
    for e in entries[-args.n:]:
        console.print(f"[dim]{e['seq']:>5} {e['ts']}[/dim] [bold]{e['event']}[/bold] {e['tenant_id']} {e['actor']} "
                      + json.dumps(e["data"], default=str)[:160])
    return 0


def main(argv: list[str] | None = None) -> int:
    from .agent.session import ThreadAccessError

    p = argparse.ArgumentParser(prog="fleet-copilot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, llm=True):
        sp.add_argument("--tenant", required=True, help="tenant (company) id, e.g. acme-001")
        sp.add_argument("--actor", default=None, help="your identity (default admin@<tenant>); approvals are checked "
                                                      "against the access directory")
        if llm:
            sp.add_argument("--llm", choices=["claude", "offline"], default=_default_llm(),
                            help="LLM backend (default: claude if ANTHROPIC_API_KEY is set, else offline)")
        sp.add_argument("--trace", action="store_true", help="print plan and node trace")

    def approval_opts(sp):
        sp.add_argument("--approve", default="prompt",
                        help="prompt | none | all | later | comma-separated proposal ids")
        sp.add_argument("--second-approver", help="second identity for high-risk proposals (two-person rule)")

    sp = sub.add_parser("chat", help="interactive session")
    common(sp)
    sp.add_argument("--thread", help="continue an existing conversation")
    sp = sub.add_parser("ask", help="one-shot question")
    common(sp)
    sp.add_argument("question")
    sp.add_argument("--thread", help="continue an existing conversation")
    approval_opts(sp)
    sp.add_argument("--json", action="store_true", help="print the full structured result as JSON")
    sp = sub.add_parser("approvals", help="list or resolve conversations waiting for approval")
    sp.add_argument("action", choices=["list", "resume"])
    common(sp)
    sp.add_argument("--thread", help="thread to resume")
    sp.add_argument("--all-threads", action="store_true", help="list: include conversations not awaiting approval")
    approval_opts(sp)
    sp = sub.add_parser("explain", help="trace the last answer of a conversation back to raw telemetry lines")
    common(sp, llm=False)
    sp.add_argument("--thread", help="thread to explain (default: your most recent completed conversation)")
    sp.add_argument("--json", action="store_true")
    sp = sub.add_parser("insights", help="run trend detectors directly (no LLM)")
    sp.add_argument("--tenant", required=True)
    sp = sub.add_parser("audit", help="inspect the audit log")
    sp.add_argument("action", choices=["verify", "tail"])
    sp.add_argument("-n", type=int, default=30)
    sp.add_argument("--tenant", help="only show this tenant's events (tail)")
    sp.add_argument("--all", action="store_true", help="operator view across all tenants (tail)")

    args = p.parse_args(argv)
    if getattr(args, "actor", "x") is None:
        args.actor = f"admin@{args.tenant}"
    try:
        return {"chat": cmd_chat, "ask": cmd_ask, "approvals": cmd_approvals, "explain": cmd_explain,
                "insights": cmd_insights, "audit": cmd_audit}[args.cmd](args)
    except UnknownTenantError as e:
        console.print(f"[red]Unknown tenant {e}. Known tenants: {', '.join(COMPANY_NAMES)}[/red]")
        return 2
    except AuditIntegrityError as e:
        console.print(f"[red]{e}[/red]")
        return 3
    except ThreadAccessError as e:
        console.print(f"[red]{e}[/red]")
        return 4


if __name__ == "__main__":
    sys.exit(main())
