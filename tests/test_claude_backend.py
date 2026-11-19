"""Request-shape tests for the Claude backend, with the SDK client replaced by a fake (no network)."""

from types import SimpleNamespace

from fleet_copilot.llm.claude import FALLBACK_BETA, ClaudeBackend
from fleet_copilot.tools import ALL_TOOLS


class Block(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.response


def _backend(response, fallbacks=True):
    b = ClaudeBackend.__new__(ClaudeBackend)
    fake = FakeMessages(response)
    b.client = SimpleNamespace(messages=fake, beta=SimpleNamespace(messages=fake))
    b.model, b.effort, b.use_fallbacks, b.name = "claude-opus-5-5", "medium", fallbacks, "claude:test"
    return b, fake


def _resp(content, stop="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                                 cache_creation_input_tokens=0))


TOOLS = [t.schema() for t in ALL_TOOLS]


def test_step_request_shape():
    b, fake = _backend(_resp([Block(type="text", text="hi")]))
    turn = b.step(system="SYS", messages=[{"role": "user", "content": "q"}], tools=TOOLS, plan={}, allow_tools=True)
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert all(t["strict"] for t in kw["tools"])
    assert "tool_choice" not in kw and "temperature" not in kw and "thinking" not in kw
    assert turn.text == "hi"


def test_step_budget_disables_tools():
    b, fake = _backend(_resp([Block(type="text", text="final")]), fallbacks=False)
    b.step(system="S", messages=[], tools=TOOLS, plan={}, allow_tools=False)
    assert fake.calls[0]["tool_choice"] == {"type": "none"}
    assert "betas" not in fake.calls[0]


def test_refusal_never_runs_tools():
    content = [Block(type="tool_use", id="t1", name="fleet_overview", input={})]
    b, _ = _backend(_resp(content, stop="refusal"))
    turn = b.step(system="S", messages=[], tools=TOOLS, plan={}, allow_tools=True)
    assert turn.tool_uses == [] and "can't help" in turn.text


def test_truncated_tool_call_becomes_explicit_non_answer():
    content = [Block(type="thinking", thinking="", signature="sig"), Block(type="tool_use", id="t1", name="fleet_overview", input={})]
    b, _ = _backend(_resp(content, stop="max_tokens"))
    turn = b.step(system="S", messages=[], tools=TOOLS, plan={}, allow_tools=True)
    assert turn.tool_uses == [] and "cut off" in turn.text
    assert turn.content[0]["type"] == "thinking"  # thinking still replayed unchanged


def test_thinking_blocks_are_preserved_verbatim():
    content = [Block(type="thinking", thinking="", signature="sig"), Block(type="tool_use", id="t1", name="fleet_overview", input={})]
    b, _ = _backend(_resp(content, stop="tool_use"))
    turn = b.step(system="S", messages=[], tools=TOOLS, plan={}, allow_tools=True)
    assert turn.content[0] == {"type": "thinking", "thinking": "", "signature": "sig"}


def test_planner_uses_structured_output():
    b, fake = _backend(_resp([Block(type="text", text='{"intent":"fleet_question","goal":"g","steps":[],'
                                                       '"needs_actions":false,"assumptions":[]}')]))
    plan, usage = b.plan(system="S", question="q", history=["earlier"])
    fmt = fake.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert plan["intent"] == "fleet_question" and usage["input_tokens"] == 10
