"""Tool abstraction shared by the Claude and offline backends.

A ``ToolSpec`` binds a name + description + pydantic argument model + handler.
The same pydantic model (a) generates the JSON schema sent to Claude (strict,
``additionalProperties: false``) and (b) re-validates every call at execution
time, so malformed or extra arguments never reach a handler regardless of
which model produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from ..actions_backend import ActionBackend
from ..config import Thresholds
from ..data.store import TenantView
from ..evidence import EvidenceLedger


class ArgsBase(BaseModel):
    """Base for every tool's arguments: no extra fields; identifiers normalised to canonical case
    (device ids upper, employee ids lower) so lookups and policy checks are case-insensitive."""
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalise_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if isinstance(data.get("device_id"), str):
                data["device_id"] = data["device_id"].strip().upper()
            if isinstance(data.get("employee_id"), str):
                data["employee_id"] = data["employee_id"].strip().lower()
        return data


class ToolError(Exception):
    """Raised by handlers for user-facing tool errors (returned as is_error results)."""


@dataclass
class ToolContext:
    view: TenantView
    ledger: EvidenceLedger
    thresholds: Thresholds
    backend: ActionBackend
    proposals: list[dict] = field(default_factory=list)  # all proposals in this conversation

    @property
    def tenant_id(self) -> str:
        return self.view.tenant_id


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    kind: Literal["read", "action"]
    handler: Callable[[ToolContext, Any], dict]

    def schema(self, strict: bool = True) -> dict:
        js = _clean(self.args_model.model_json_schema())
        js.setdefault("properties", {})
        js["additionalProperties"] = False
        tool = {"name": self.name, "description": self.description, "input_schema": js}
        if strict:
            tool["strict"] = True
        return tool


def _clean(node: Any) -> Any:
    """Strip pydantic-only keywords (title/default) that add noise to the schema."""
    if isinstance(node, dict):
        return {k: _clean(v) for k, v in node.items() if k not in ("title", "default")}
    if isinstance(node, list):
        return [_clean(v) for v in node]
    return node
