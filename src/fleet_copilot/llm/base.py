"""Backend-neutral LLM interface used by the graph nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMTurn:
    content: list[dict]          # Anthropic-format content blocks (text / tool_use / thinking ...)
    stop_reason: str
    usage: dict = field(default_factory=dict)

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()


class LLMBackend(Protocol):
    name: str

    def plan(self, *, system: str, question: str, history: list[str]) -> tuple[dict, dict]:
        """Return (plan, usage)."""

    def step(self, *, system: str, messages: list[dict], tools: list[dict], plan: dict,
             allow_tools: bool) -> LLMTurn:
        ...
