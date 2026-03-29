import time
import threading
import pytest
import numpy as np
from unittest.mock import MagicMock
from storage.sqlite_store import SQLiteStore
from infrastructure.checkpoint_manager import CheckpointManager

class FakeComponent:
    """Mock component with get_state/restore_state."""
    def __init__(self, value=0):
        self.value = value

    def get_state(self):
        return {"value": self.value}

    def restore_state(self, state):
        self.value = state["value"]

@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "test.db"))

class TestCheckpointManager:
    def test_flush_writes_all_components(self, store):
        comp_a = FakeComponent(value=42)
        comp_b = FakeComponent(value=99)
        mgr = CheckpointManager(store, {"a": comp_a, "b": comp_b})
        mgr._flush()

        rec_a = store.load("a")
        rec_b = store.load("b")
        assert rec_a.state == {"value": 42}
        assert rec_b.state == {"value": 99}

    def test_restore_all_succeeds(self, store):
        store.checkpoint("a", {"value": 10})
        store.checkpoint("b", {"value": 20})

        comp_a = FakeComponent(value=0)
        comp_b = FakeComponent(value=0)
        mgr = CheckpointManager(store, {"a": comp_a, "b": comp_b})
        result = mgr.restore_all()

        assert result is True
        assert comp_a.value == 10
        assert comp_b.value == 20

    def test_restore_fails_on_missing_key(self, store):
        store.checkpoint("a", {"value": 10})
        # "b" is missing

        comp_a = FakeComponent()
        comp_b = FakeComponent()
        mgr = CheckpointManager(store, {"a": comp_a, "b": comp_b})
        result = mgr.restore_all()

        assert result is False

    def test_restore_fails_on_stale_checkpoint(self, store):
        import sqlite3, json
        conn = store._conn
        conn.execute(
            "INSERT OR REPLACE INTO checkpoints (key, state, updated_at) VALUES (?, ?, ?)",
            ("a", json.dumps({"value": 1}), time.time() - 600),
        )
        conn.commit()

        comp_a = FakeComponent()
        mgr = CheckpointManager(store, {"a": comp_a})
        result = mgr.restore_all()

        assert result is False

    def test_stop_flushes_before_exit(self, store):
        comp = FakeComponent(value=77)
        mgr = CheckpointManager(store, {"x": comp})
        mgr.start()
        mgr.stop()

        rec = store.load("x")
        assert rec is not None
        assert rec.state == {"value": 77}

    def test_background_thread_writes_periodically(self, store):
        comp = FakeComponent(value=1)
        mgr = CheckpointManager(store, {"x": comp})
        mgr.INTERVAL_S = 0.1  # Speed up for test
        mgr.start()
        time.sleep(0.3)
        comp.value = 999
        time.sleep(0.2)
        mgr.stop()

        rec = store.load("x")
        assert rec.state == {"value": 999}


class TestCheckpointIntegration:
    """End-to-end: real components -> checkpoint -> restore -> identical output."""

    def test_hmm_checkpoint_round_trip_via_manager(self, store):
        from regime.hmm import GaussianHMM
        rng = np.random.RandomState(42)

        hmm = GaussianHMM(n_states=3, n_features=6, seed=42)
        for _ in range(80):
            hmm.filter_step(rng.randn(6))

        # Checkpoint via manager
        mgr = CheckpointManager(store, {"hmm": hmm})
        mgr._flush()

        # Create fresh HMM and restore
        hmm2 = GaussianHMM(n_states=3, n_features=6, seed=99)
        mgr2 = CheckpointManager(store, {"hmm": hmm2})
        assert mgr2.restore_all() is True

        # Feed 100 identical observations, outputs must match
        for _ in range(100):
            obs = rng.randn(6)
            s1 = hmm.filter_step(obs)
            s2 = hmm2.filter_step(obs)
            assert s1[0] == s2[0]
            np.testing.assert_array_equal(s1[1], s2[1])

    def test_full_pipeline_checkpoint(self, store):
        """All 5 components checkpoint and restore."""
        from regime.hmm import GaussianHMM
        from regime.transition import TransitionDetector, _PyKalmanDerivativeTracker
        from microstructure.pdf.normalizing_flow import GMMDensityEstimator

        try:
            from alpha.return_predictor import _PyReturnPredictor as RP
        except ImportError:
            from alpha.return_predictor import ReturnPredictor as RP

        # Create components
        hmm = GaussianHMM(n_states=3, n_features=6, seed=42)
        td = TransitionDetector()
        rp = RP(n_features=5)
        gmm = GMMDensityEstimator()

        components = {"hmm": hmm, "transition": td, "predictor": rp, "gmm": gmm}
        mgr = CheckpointManager(store, components)
        mgr._flush()

        # Restore into fresh components
        hmm2 = GaussianHMM(n_states=3, n_features=6, seed=99)
        td2 = TransitionDetector()
        rp2 = RP(n_features=5)
        gmm2 = GMMDensityEstimator()

        components2 = {"hmm": hmm2, "transition": td2, "predictor": rp2, "gmm": gmm2}
        mgr2 = CheckpointManager(store, components2)
        assert mgr2.restore_all() is True
