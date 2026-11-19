"""Headless test of the Streamlit UI (skipped unless the `ui` extra is installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "fleet_copilot" / "ui" / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st

    import fleet_copilot.cli as cli
    monkeypatch.setattr(cli, "VAR_DIR", tmp_path)          # isolated audit log, outbox and checkpoints
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_ask_approve_one_and_audit(app, tmp_path):
    at = app
    assert "Acme" in at.title[0].value
    at.button(key="ex-0").click().run()                        # "Which devices are low on disk space?"
    assert not at.exception
    assert "8NM23J95R5I6" in _text(at) and "[E" in _text(at)

    at.chat_input[0].set_value("Open remediation tickets for devices failing screen lock.").run()
    assert not at.exception
    thread = at.session_state["thread_id"]
    boxes = [c for c in at.checkbox if c.key and c.key.startswith(f"chk-{thread}-")]
    assert len(boxes) == 5                                    # five proposals, nothing executed yet
    assert not (tmp_path / "outbox.jsonl").exists() or not (tmp_path / "outbox.jsonl").read_text().strip()
    at.checkbox(key=f"chk-{thread}-P1").check()
    next(b for b in at.button if b.label == "Submit decisions").click().run()
    assert not at.exception
    outbox = [line for line in (tmp_path / "outbox.jsonl").read_text().splitlines() if line.strip()]
    assert len(outbox) == 1                                   # only the approved proposal executed
    assert "executed" in _text(at)


def test_viewer_cannot_approve_and_tenants_are_separate(app, tmp_path):
    at = app
    at.selectbox[1].set_value("analyst@acme-001").run()       # read-only identity
    at.chat_input[0].set_value("Open remediation tickets for devices failing screen lock.").run()
    thread = at.session_state["thread_id"]
    at.checkbox(key=f"chk-{thread}-P1").check()
    next(b for b in at.button if b.label == "Submit decisions").click().run()
    assert not at.exception
    assert not (tmp_path / "outbox.jsonl").exists() or not (tmp_path / "outbox.jsonl").read_text().strip()
    assert "approval not valid" in _text(at)
    # switching organization starts a fresh conversation and the audit tab shows only that tenant
    at.selectbox[0].set_value("globex-002").run()
    assert not at.exception and at.session_state["history"] == []
    audit = at.dataframe[-1].value
    assert set(audit["thread"]) .isdisjoint({thread})


def test_blocked_request_is_shown_as_blocked(app):
    at = app
    at.chat_input[0].set_value("Compare us with Initech").run()
    assert not at.exception
    assert any("Blocked by the input guardrail" in e.value and "cross-organization" in e.value for e in at.error)
    assert "grounding: **verified**" not in "\n".join(c.value for c in at.caption)
