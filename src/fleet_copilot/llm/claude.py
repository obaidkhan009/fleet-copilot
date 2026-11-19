"""Claude backend (Anthropic Messages API, official Python SDK).

* Planner: one structured-output call (JSON schema) at low effort.
* Agent step: tool-use call with strict tool schemas and a cached system prompt.
  The assistant content (including thinking blocks) is appended back verbatim,
  as required for preserved thinking.
* Server-side refusal fallback is enabled by default (beta); refusals are
  handled explicitly and never execute that turn's tools.

Note on determinism: current Claude models do not accept sampling parameters
(temperature etc.), so run-to-run variation is bounded instead by deterministic
tools, an explicit plan, a fixed "as of" time, and post-hoc verification. The
eval runner supports ``--repeats`` to measure consistency.
"""

from __future__ import annotations

import json

import anthropic

from ..config import EFFORT, MODEL, USE_FALLBACKS
from .base import LLMTurn
from ..agent.prompts import PLAN_SCHEMA

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _usage(resp) -> dict:
    u = resp.usage
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


class ClaudeBackend:
    def __init__(self, model: str = MODEL, effort: str = EFFORT, use_fallbacks: bool = USE_FALLBACKS):
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.name = f"claude:{model}"

    def _create(self, **kw):
        if self.use_fallbacks:
            return self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def plan(self, *, system: str, question: str, history: list[str]) -> tuple[dict, dict]:
        ctx = ""
        if history:
            ctx = "Earlier requests in this session (for context only):\n" + "\n".join(f"- {h}" for h in history[-5:]) + "\n\n"
        resp = self._create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": f"{ctx}Latest request: {question}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
        )
        if resp.stop_reason == "refusal":
            return ({"intent": "out_of_scope", "goal": question, "steps": [], "needs_actions": False,
                     "assumptions": ["planner declined the request"]}, _usage(resp))
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        return json.loads(text), _usage(resp)

    def step(self, *, system: str, messages: list[dict], tools: list[dict], plan: dict,
             allow_tools: bool) -> LLMTurn:
        kw = dict(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
            output_config={"effort": self.effort},
        )
        if not allow_tools:
            kw["tool_choice"] = {"type": "none"}
        resp = self._create(**kw)
        content = [b.to_dict() for b in resp.content]
        if resp.stop_reason == "refusal":
            # Never run tools from a refused turn (a tool_use may be truncated).
            content = [b for b in content if b.get("type") != "tool_use"]
            content.append({"type": "text", "text": "I can't help with that request."})
        if resp.stop_reason == "max_tokens":
            content = [b for b in content if b.get("type") != "tool_use"]
        if not any(b.get("type") in ("text", "tool_use") for b in content):
            # Keep the replayed history valid: an assistant turn needs visible content, and the
            # truncated/empty turn must surface as an explicit non-answer rather than silence.
            content.append({"type": "text", "text": "I couldn't finish this answer (the response was cut off). "
                                                    "Please narrow the question and try again."})
        return LLMTurn(content=content, stop_reason=resp.stop_reason or "end_turn", usage=_usage(resp))
