import pytest
from storage.state_store import StateStore, CheckpointRecord


def test_checkpoint_record_fields():
    rec = CheckpointRecord(key="hmm", state={"a": 1}, updated_at=1000.0)
    assert rec.key == "hmm"
    assert rec.state == {"a": 1}
    assert rec.updated_at == 1000.0


def test_state_store_is_protocol():
    """StateStore is a Protocol — cannot be instantiated directly."""
    assert hasattr(StateStore, 'checkpoint')
    assert hasattr(StateStore, 'load')
    assert hasattr(StateStore, 'load_positions')
    assert hasattr(StateStore, 'upsert_config_audit')
    assert hasattr(StateStore, 'set_halt')
    assert hasattr(StateStore, 'get_halt')


import time
import tempfile
import os
from storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db_path)


def test_checkpoint_and_load(store):
    store.checkpoint("hmm", {"weights": [1, 2, 3]})
    rec = store.load("hmm")
    assert rec is not None
    assert rec.key == "hmm"
    assert rec.state == {"weights": [1, 2, 3]}
    assert rec.updated_at > 0


def test_load_missing_key(store):
    assert store.load("nonexistent") is None


def test_checkpoint_overwrites(store):
    store.checkpoint("hmm", {"v": 1})
    store.checkpoint("hmm", {"v": 2})
    rec = store.load("hmm")
    assert rec.state == {"v": 2}


def test_set_and_get_halt(store):
    assert store.get_halt() is None
    store.set_halt("MAX_DD_BREACH")
    halt = store.get_halt()
    assert halt is not None
    assert halt["reason"] == "MAX_DD_BREACH"


def test_clear_halt(store):
    store.set_halt("TEST")
    store.set_halt(None)
    assert store.get_halt() is None


def test_upsert_config_audit(store):
    store.upsert_config_audit("abc123", "hash456")
    # Should not raise — just verifies write succeeds


def test_load_positions_empty(store):
    result = store.load_positions("2026-03-28")
    assert result == []


def test_position_checkpoint_schema(store):
    """Verify the position checkpoint has the fields the risk process expects."""
    store.checkpoint("position", {
        "symbol": "AAPL",
        "qty": 100.0,
        "entry_price": 150.0,
        "unrealized_pnl": 50.0,
    })
    rec = store.load("position")
    assert rec is not None
    assert "qty" in rec.state
    assert "unrealized_pnl" in rec.state

def test_risk_metrics_checkpoint_schema(store):
    """Verify the risk_metrics checkpoint has the fields the risk process expects."""
    store.checkpoint("risk_metrics", {
        "drawdown": 0.05,
        "equity": 100000.0,
        "pnl_1h": -200.0,
        "kill_switch": False,
    })
    rec = store.load("risk_metrics")
    assert rec is not None
    assert "drawdown" in rec.state
    assert "pnl_1h" in rec.state
    assert "kill_switch" in rec.state
