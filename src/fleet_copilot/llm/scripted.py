"""Scripted backend for red-teaming: replays a fixed sequence of model turns.

Used by the eval suite and tests to simulate a *misbehaving* model (calling
tools on another tenant's devices, proposing actions without evidence,
inventing citations). Because the script ignores tool results, it shows that
the guardrails are enforced by code paths the model cannot influence.
"""

from __future__ import annotations

from .base import LLMTurn


class ScriptedBackend:
    name = "scripted"

    def __init__(self, script: list[dict]):
        """script items: {"tool_calls": [{"name", "input"}]} or {"text": "..."}"""
        self.script = script
        self._i = 0

    def plan(self, *, system: str, question: str, history: list[str]) -> tuple[dict, dict]:
        return {"intent": "scripted", "goal": question, "steps": [], "needs_actions": False, "assumptions": []}, {}

    def step(self, *, system: str, messages: list[dict], tools: list[dict], plan: dict, allow_tools: bool) -> LLMTurn:
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if "tool_calls" in item and allow_tools:
            blocks = [{"type": "tool_use", "id": f"toolu_scripted_{self._i}_{j}", "name": c["name"], "input": c["input"]}
                      for j, c in enumerate(item["tool_calls"])]
            return LLMTurn(blocks, "tool_use")
        return LLMTurn([{"type": "text", "text": item.get("text", "")}], "end_turn")
