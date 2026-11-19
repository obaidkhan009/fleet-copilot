import pytest

from fleet_copilot.config import DATA_PATH
from fleet_copilot.data import TelemetryStore


@pytest.fixture(scope="session")
def store():
    return TelemetryStore.load(DATA_PATH)
