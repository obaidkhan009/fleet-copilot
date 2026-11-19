"""Central configuration: paths, model settings, and the analytic policy thresholds.

Thresholds live in one place so that the *same* definition of "low disk" or
"RAM constrained" is used by the read tools, the insight detectors, the action
evidence policy and the evaluation oracle's documentation. Changing a threshold
here changes behaviour everywhere consistently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("FLEET_DATA_PATH", ROOT / "data" / "device-telemetry-dataset.ndjson"))
VAR_DIR = Path(os.environ.get("FLEET_VAR_DIR", ROOT / "var"))

# --- LLM settings -----------------------------------------------------------
MODEL = os.environ.get("FLEET_COPILOT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("FLEET_COPILOT_EFFORT", "medium")
# Server-side refusal fallback (beta). Set FLEET_COPILOT_FALLBACKS=off to disable.
USE_FALLBACKS = os.environ.get("FLEET_COPILOT_FALLBACKS", "on").lower() != "off"
# Strict tool schemas guarantee schema-valid tool arguments.
STRICT_TOOLS = os.environ.get("FLEET_COPILOT_STRICT_TOOLS", "on").lower() != "off"

# --- Agent loop budgets -----------------------------------------------------
MAX_AGENT_STEPS = int(os.environ.get("FLEET_COPILOT_MAX_STEPS", "10"))
MAX_GROUNDING_RETRIES = 1

# --- Tenancy ----------------------------------------------------------------
# The dataset only carries company ids; display names are derived from them.
COMPANY_NAMES: dict[str, str] = {
    "acme-001": "Acme",
    "globex-002": "Globex",
    "initech-003": "Initech",
}


@dataclass(frozen=True)
class Thresholds:
    """Operational definitions used across tools, insights and guardrails."""

    # Storage
    disk_low_free_pct: float = 15.0
    disk_critical_free_pct: float = 10.0
    # Memory
    ram_high_used_pct: float = 85.0
    # "Consistently" = condition holds in at least this share of snapshots in the window
    sustained_fraction: float = 0.7
    # Battery
    battery_cycle_limit: int = 1000          # typical rated cycle life for modern laptop cells
    battery_eol_horizon_days: int = 120       # "approaching" = projected to hit the limit within this horizon
    battery_capacity_drop_pct: float = 3.0    # full-charge-capacity loss within the window worth flagging
    # Trend analysis
    min_snapshots_for_trend: int = 7


THRESHOLDS = Thresholds()

# Software outside the approved baseline that carries a known risk.
RISKY_SOFTWARE: dict[str, str] = {
    "uTorrent": "peer-to-peer file sharing client (data exfiltration / licensing risk)",
    "TeamViewer": "unmanaged remote-access tool (remote takeover risk)",
    "CleanMyMac X": "unapproved system utility running with elevated privileges",
}
APPROVED_SOFTWARE: frozenset[str] = frozenset(
    {"Google Chrome", "Slack", "Zoom", "1Password", "Visual Studio Code"}
)

# Field-serviceability reference for the models present in the fleet. Telemetry
# cannot tell us whether RAM/SSD are soldered, so this is curated reference data.
# Apple-silicon Macs have on-package memory and soldered storage; the X1 Carbon
# Gen 11 has soldered LPDDR5 but an M.2 SSD; the XPS 15 9530 has SO-DIMM + M.2.
HARDWARE_SERVICEABILITY: dict[str, dict[str, bool]] = {
    "MacBook Pro": {"ram": False, "storage": False, "battery": True},
    "MacBook Air": {"ram": False, "storage": False, "battery": True},
    "Mac mini": {"ram": False, "storage": False, "battery": False},
    "ThinkPad X1 Carbon": {"ram": False, "storage": True, "battery": True},
    "Dell XPS 15": {"ram": True, "storage": True, "battery": True},
}

# Upgrade catalogue: the only specs the upgrade tool will order.
UPGRADE_CATALOG: dict[str, list[str]] = {
    "ram": ["16 GB", "32 GB", "64 GB"],
    "storage": ["512 GB", "1 TB", "2 TB"],
    "battery": ["OEM replacement battery"],
}
