"""Fleet Copilot web UI (Streamlit).

    uv run --extra ui fleet-copilot ui

A thin layer over the same CopilotSession the CLI uses: same graph, tools, guardrails, approval interrupt, SQLite
checkpoints and audit log. Nothing here bypasses them; the UI only renders state and collects decisions.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from fleet_copilot.agent.session import CopilotSession, ThreadAccessError, list_threads
from fleet_copilot.analytics import run_detectors
from fleet_copilot.audit import AuditIntegrityError
from fleet_copilot.config import COMPANY_NAMES, DATA_PATH, THRESHOLDS
from fleet_copilot.explain import explain_answer
from fleet_copilot.guardrails.access import AccessDirectory

EXAMPLES = [
    "Which devices are low on disk space?",
    "Show me laptops failing high-severity compliance checks.",
    "How many devices are running an OS older than macOS 15?",
    "Which batteries are approaching end of life?",
    "Is our compliance drifting over time?",
    "Open remediation tickets for devices failing screen lock.",
    "Propose fixes for devices that are consistently constrained by RAM.",
    "Compare us with another company.",
]

st.set_page_config(page_title="Fleet Copilot", layout="wide")


# ---------------------------------------------------------------------------
# dependencies (one set per backend, shared across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_deps(llm_name: str):
    from fleet_copilot.cli import _deps
    return _deps(llm_name)


def has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def target(p: dict) -> str:
    a = p["args"]
    extra = a.get("check_id") or (f"{a.get('component')} {a.get('spec')}" if a.get("component") else "") or a.get("employee_id", "")
    return f"{a.get('device_id')} {extra}".strip()


# ---------------------------------------------------------------------------
# sidebar: who am I, which company, which model
# ---------------------------------------------------------------------------

st.sidebar.title("Fleet Copilot")
tenant = st.sidebar.selectbox("Organization", list(COMPANY_NAMES), format_func=lambda t: f"{COMPANY_NAMES[t]} ({t})")
directory = AccessDirectory.load(COMPANY_NAMES)
identities = sorted(directory.roles.get(tenant, {}))
default_actor = identities.index(f"admin@{tenant}") if f"admin@{tenant}" in identities else 0
actor = st.sidebar.selectbox("Signed in as", identities, index=default_actor,
                             format_func=lambda i: f"{i}  ({', '.join(sorted(directory.roles_of(tenant, i))) or 'no role'})")
llm_name = st.sidebar.radio("Model", ["offline", "claude"], index=1 if has_key() else 0, horizontal=True,
                            help="offline = deterministic stand-in (no API key). claude = the real model "
                                 "(needs ANTHROPIC_API_KEY).")
if llm_name == "claude" and not has_key():
    st.sidebar.error("Set ANTHROPIC_API_KEY before starting the UI to use Claude.")
    st.stop()

try:
    deps = get_deps(llm_name)
except AuditIntegrityError as e:
    st.error(f"Audit log integrity check failed - refusing to run. {e}")
    st.stop()

# A conversation is (tenant, model, thread). Switching organization or model starts a new one.
ctx = (tenant, llm_name)
if st.session_state.get("ctx") != ctx:
    st.session_state.update(ctx=ctx, thread_id=None, history=[], session=None, session_key=None)


def get_session() -> CopilotSession:
    key = (tenant, actor, llm_name, st.session_state.thread_id)
    if st.session_state.session_key != key:
        s = CopilotSession(deps, tenant, actor, thread_id=st.session_state.thread_id)
        st.session_state.update(session=s, thread_id=s.thread_id,
                                session_key=(tenant, actor, llm_name, s.thread_id))
    return st.session_state.session


if st.sidebar.button("New conversation", width="stretch"):
    st.session_state.update(thread_id=None, history=[], session=None, session_key=None)
    st.rerun()

session = get_session()
st.sidebar.caption(f"Thread `{session.thread_id}`")

st.sidebar.subheader("Waiting for approval")
pending = [t for t in list_threads(deps, tenant) if t["awaiting_approval"]]
if not pending:
    st.sidebar.caption("Nothing pending.")
for t in pending[:8]:
    label = f"{t['question'][:40]}… ({len(t['proposals'])} proposal(s))"
    if st.sidebar.button(label, key=f"open-{t['thread_id']}", width="stretch"):
        st.session_state.update(thread_id=t["thread_id"], session=None, session_key=None)
        s = get_session()
        st.session_state.history = [{"question": t["question"], "final": None, "request": s.pending_request()}]
        st.rerun()


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

GUARD_LABELS = {
    "cross_tenant": "cross-organization request",
    "prompt_injection": "prompt-injection attempt",
    "empty": "empty question",
    "too_long": "question too long",
}


def render_final(final: dict) -> None:
    if final.get("blocked"):
        guard = final.get("guard") or {}
        st.markdown(final["answer"])
        st.error(f"Blocked by the input guardrail before reaching the model: "
                 f"**{GUARD_LABELS.get(guard.get('category'), guard.get('category') or 'blocked')}**. "
                 "No tools were called and no data was read; the attempt is recorded in the audit log as a security event.")
        return
    st.markdown(final["answer"] or "_(empty)_")
    g = final.get("grounding") or {}
    plan = final.get("plan") or {}
    cols = st.columns(4)
    cols[0].caption(f"intent: **{plan.get('intent', '-')}**")
    cols[1].caption(f"agent steps: **{final.get('steps', 0)}**")
    cols[2].caption("grounding: " + ("**verified**" if g.get("ok", True) else "**FAILED**"))
    cols[3].caption(f"evidence cited: **{len(final.get('evidence', []))}**")
    if final.get("evidence"):
        with st.expander(f"Evidence ({len(final['evidence'])})"):
            st.dataframe([{"id": e["id"], "tool": e["tool"],
                           "device": e.get("device_id") or f"{len(e['devices'])} devices (aggregate)",
                           "claim": e["claim"], "snapshot refs": len(e["snapshot_refs"])} for e in final["evidence"]],
                         hide_index=True, width="stretch")
    if final.get("proposals"):
        with st.expander(f"Proposals ({len(final['proposals'])})", expanded=True):
            st.dataframe([{"id": p["id"], "action": p["tool"], "target": target(p), "risk": p["policy"]["risk"],
                           "status": p["status"],
                           "detail": (p.get("execution") or {}).get("external_id") or (p.get("decision") or {}).get("reason")
                           or "; ".join(p["policy"]["reasons"])} for p in final["proposals"]],
                         hide_index=True, width="stretch")
    with st.expander("Explain: sentence → evidence → raw telemetry line"):
        try:
            rep = explain_answer(final, session.state.get("evidence", {}), DATA_PATH)
        except Exception as e:  # noqa: BLE001 - explaining must never break the chat
            st.caption(f"Could not build the trace: {e}")
            rep = {"units": []}
        for i, u in enumerate(rep["units"], 1):
            st.markdown(f"**{i}.** {u['sentence']}")
            if not u["citations"]:
                st.caption("no citation (no telemetry facts in this sentence)")
            for c in u["citations"]:
                if c.get("missing"):
                    st.error(f"{c['id']} is not in the evidence ledger")
                    continue
                kind = "aggregate" if c["aggregate"] else c["device_id"]
                body = [f"{c['id']} <- {c['tool']}({json.dumps(c['args'])}) [{kind}]", f"   {c['claim']}"]
                if c.get("data"):
                    body.append(f"   data: {json.dumps(c['data'], default=str)}")
                body += [f"   line {ln['line']}  {ln['ref']}  {json.dumps(ln['excerpt'])}" for ln in c["lines"]]
                if c.get("more_refs"):
                    body.append(f"   ... +{c['more_refs']} more snapshot refs")
                if c.get("note"):
                    body.append(f"   ({c['note']})")
                st.code("\n".join(body), language=None)
        st.caption(f"Line numbers refer to {DATA_PATH.name}.")
    with st.expander("Plan and agent trace"):
        st.json({"plan": plan, "trace": final.get("trace")}, expanded=False)


def render_approval(turn: dict) -> None:
    req = turn["request"]
    st.markdown(req.get("answer") or "")
    roles = directory.roles_of(tenant, actor)
    st.warning("The agent proposed the actions below. **Nothing has been executed.** Tick what you approve.")
    if "approver" not in roles:
        st.info(f"You are signed in as **{actor}** ({', '.join(sorted(roles)) or 'no role'}), who cannot approve: "
                "an approval from you will be refused. Switch identity in the sidebar.")
    with st.form(f"approve-{session.thread_id}-{len(st.session_state.history)}"):
        chosen = {}
        for p in req["proposals"]:
            risk = p["policy"]["risk"]
            with st.container(border=True):
                st.markdown(f"**{p['id']} · {p['tool']}** · target `{target(p)}` · risk **{risk}**"
                            + (" · needs two approvers" if risk == "high" else ""))
                args = {k: v for k, v in p["args"].items() if k != "evidence_ids"}
                st.caption("args: " + json.dumps(args) + f" · evidence: {', '.join(p['args'].get('evidence_ids', []))}")
                for f in p["policy"]["verified_facts"]:
                    st.caption(f"✓ verified from telemetry: {f}")
                chosen[p["id"]] = st.checkbox(f"Approve {p['id']}", key=f"chk-{session.thread_id}-{p['id']}")
        second = None
        if any(p["policy"]["risk"] == "high" for p in req["proposals"]):
            others = [i for i in identities if i != actor]
            second = st.selectbox("Second approver for high-risk actions (two-person rule)", ["(none)"] + others)
            second = None if second == "(none)" else second
        submitted = st.form_submit_button("Submit decisions", type="primary")
    if submitted:
        decisions = {}
        for p in req["proposals"]:
            dec = {"approved": chosen[p["id"]], "approver": actor, "comment": "via web UI"}
            if p["policy"]["risk"] == "high" and second:
                dec["approvers"] = [second]
            decisions[p["id"]] = dec
        with st.spinner("Executing approved actions…"):
            res = session.resume(decisions)
        turn.update(final=res.final, request=res.approval_request)
        st.rerun()


# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------

st.title(f"Fleet Copilot · {COMPANY_NAMES[tenant]}")
st.caption("Answers are grounded in device telemetry and cite evidence ids [E#]. Actions are proposals until an "
           "administrator approves them. Every step is written to the hash-chained audit log.")
tab_chat, tab_insights, tab_audit = st.tabs(["Chat", "Insights", "Audit log"])

with tab_chat:
    history = st.session_state.history
    awaiting = bool(history) and history[-1].get("request") is not None
    if not history:
        st.markdown("**Try an example:**")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex-{i}", width="stretch"):
                st.session_state.queued = q
                st.rerun()
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("request") is not None:
                render_approval(turn)
            elif turn.get("final"):
                render_final(turn["final"])
    typed = st.chat_input("Ask about your fleet…" if not awaiting else "Decide on the pending proposals first",
                          disabled=awaiting)
    question = typed or st.session_state.pop("queued", None)
    if question and not awaiting:
        with st.spinner("Planning, calling tools, verifying…"):
            try:
                res = session.ask(question)
            except (ThreadAccessError, AuditIntegrityError) as e:
                st.error(str(e))
                st.stop()
        history.append({"question": question, "final": res.final, "request": res.approval_request})
        st.rerun()

with tab_insights:
    st.caption("Deterministic trend detectors run directly over the telemetry (no model involved). These are the same "
               "detectors the agent calls through the detect_insights tool.")
    view = deps.store.scoped(tenant)
    rows = [{"severity": f.severity, "kind": f.kind, "device": f.device_id, "finding": f.title,
             "explanation": f.explanation} for f in run_detectors(view, THRESHOLDS) if f.severity != "info"]
    st.caption(f"{len(rows)} findings · data as of {view.as_of:%Y-%m-%d %H:%M} UTC")
    st.dataframe(rows, hide_index=True, width="stretch")

with tab_audit:
    try:
        ok, bad = deps.audit.verify()
        entries = [e for e in deps.audit.entries if e["tenant_id"] == tenant]
    except AuditIntegrityError as e:
        ok, bad, entries = False, None, []
        st.error(str(e))
    (st.success if ok else st.error)("Hash chain verified" if ok else f"Hash chain BROKEN at seq {bad}")
    st.caption(f"{len(entries)} events for {COMPANY_NAMES[tenant]} (other organizations' events are not shown).")
    st.dataframe([{"seq": e["seq"], "time": e["ts"][:19], "event": e["event"], "actor": e["actor"],
                   "thread": e["thread_id"], "data": json.dumps(e["data"], default=str)[:300]}
                  for e in reversed(entries[-300:])], hide_index=True, width="stretch")
