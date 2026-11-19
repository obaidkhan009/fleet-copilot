import pytest

from fleet_copilot.actions_backend import ActionBackend
from fleet_copilot.audit import AuditLog
from fleet_copilot.config import DATA_PATH, THRESHOLDS
from fleet_copilot.data import TelemetryStore
from fleet_copilot.evidence import EvidenceLedger
from fleet_copilot.tools import ToolContext


@pytest.fixture(scope="session")
def store():
    return TelemetryStore.load(DATA_PATH)


@pytest.fixture
def make_ctx(store):
    def _make(tenant="acme-001", backend=None):
        return ToolContext(view=store.scoped(tenant), ledger=EvidenceLedger(), thresholds=THRESHOLDS,
                           backend=backend or ActionBackend())
    return _make


@pytest.fixture
def make_deps(store):
    from fleet_copilot.agent.graph import Deps

    def _make(llm):
        return Deps(store=store, llm=llm, audit=AuditLog(), backend=ActionBackend())
    return _make
