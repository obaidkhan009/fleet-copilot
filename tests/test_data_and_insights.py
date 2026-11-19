import pytest

from evals import oracle
from fleet_copilot.analytics import run_detectors
from fleet_copilot.config import THRESHOLDS
from fleet_copilot.data import UnknownTenantError
from fleet_copilot.data.models import os_version_key


def test_dataset_shape(store):
    assert store.tenant_ids() == ["acme-001", "globex-002", "initech-003"]
    assert sum(len(store.scoped(t).device_ids()) for t in store.tenant_ids()) == 25
    for t in store.tenant_ids():
        for d in store.scoped(t).device_ids():
            assert len(store.scoped(t).history(d)) == 30


def test_tenant_view_is_partitioned(store):
    acme, globex = store.scoped("acme-001"), store.scoped("globex-002")
    assert not set(acme.device_ids()) & set(globex.device_ids())
    assert all(s.company_id == "acme-001" for d in acme.device_ids() for s in acme.history(d))
    with pytest.raises(KeyError):
        acme.history(globex.device_ids()[0])
    with pytest.raises(UnknownTenantError):
        store.scoped("umbrella-999")


def test_os_version_key():
    assert os_version_key("macOS", "14.5") < os_version_key("macOS", "15")
    assert os_version_key("Windows", "10 22H2") < os_version_key("Windows", "11 22H2") < os_version_key("Windows", "11 23H2")


def test_missing_battery_block_is_none_not_zero(store):
    h = store.scoped("acme-001").history("MT7PJB7N5LRE")
    assert any(s.battery is None for s in h) and any(s.battery is not None for s in h)
    assert all(s.battery is None for s in store.scoped("initech-003").history("RLD7WV7J82SH"))  # Mac mini


DETECTOR_VS_ORACLE = [
    ("battery_end_of_life", oracle.battery_eol),
    ("ram_constrained", oracle.ram_constrained),
    ("storage_constrained", oracle.storage_constrained),
    ("unapproved_software", oracle.unapproved_software),
]


@pytest.mark.parametrize("tenant", ["acme-001", "globex-002", "initech-003"])
@pytest.mark.parametrize("kind,oracle_fn", DETECTOR_VS_ORACLE)
def test_detectors_agree_with_independent_oracle(store, tenant, kind, oracle_fn):
    found = {f.device_id for f in run_detectors(store.scoped(tenant), THRESHOLDS, [kind]) if f.severity != "info"}
    assert found == oracle_fn(tenant)


@pytest.mark.parametrize("tenant", ["acme-001", "globex-002", "initech-003"])
def test_compliance_drift_agrees_with_oracle(store, tenant):
    found = {f.device_id for f in run_detectors(store.scoped(tenant), THRESHOLDS, ["compliance_drift"])
             if f.metrics.get("direction") == "regression" and f.metrics["check_id"] == "screen_lock"}
    assert found == oracle.compliance_regressions(tenant)


def test_findings_carry_snapshot_evidence(store):
    view = store.scoped("acme-001")
    for f in run_detectors(view, THRESHOLDS):
        assert f.snapshot_refs, f
        for ref in f.snapshot_refs:
            dev, ts = ref.split("@")
            assert dev == f.device_id
            assert any(s.ref == ref for s in view.history(dev))
