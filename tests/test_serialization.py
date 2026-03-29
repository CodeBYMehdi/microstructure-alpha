import numpy as np
import pytest


class TestHMMSerialization:
    def test_get_state_returns_dict(self):
        from regime.hmm import GaussianHMM
        hmm = GaussianHMM(n_states=3, n_features=6)
        state = hmm.get_state()
        assert isinstance(state, dict)
        assert "transition_matrix" in state
        assert "means" in state
        assert "covariances" in state
        assert "initialized" in state

    def test_round_trip_pre_warmup(self):
        from regime.hmm import GaussianHMM
        hmm1 = GaussianHMM(n_states=3, n_features=6)
        state = hmm1.get_state()

        hmm2 = GaussianHMM(n_states=3, n_features=6)
        hmm2.restore_state(state)

        assert hmm2._initialized == hmm1._initialized
        np.testing.assert_array_equal(hmm2.transition_matrix, hmm1.transition_matrix)
        np.testing.assert_array_equal(hmm2.means, hmm1.means)

    def test_round_trip_post_warmup(self):
        from regime.hmm import GaussianHMM
        rng = np.random.RandomState(42)
        hmm1 = GaussianHMM(n_states=3, n_features=6, seed=42)

        # Feed enough data to trigger initialization
        for _ in range(80):
            obs = rng.randn(6)
            hmm1.filter_step(obs)

        state = hmm1.get_state()

        hmm2 = GaussianHMM(n_states=3, n_features=6, seed=99)
        hmm2.restore_state(state)

        # Feed 100 more observations — both must produce identical output
        for _ in range(100):
            obs = rng.randn(6)
            s1 = hmm1.filter_step(obs)
            s2 = hmm2.filter_step(obs)
            # filter_step returns (int, np.ndarray) — compare separately
            assert s1[0] == s2[0], f"State diverged: {s1[0]} != {s2[0]}"
            np.testing.assert_array_equal(s1[1], s2[1])

    def test_state_is_json_serializable(self):
        import json
        from regime.hmm import GaussianHMM
        hmm = GaussianHMM(n_states=3, n_features=6)
        state = hmm.get_state()
        serialized = json.dumps(state)
        deserialized = json.loads(serialized)
        assert deserialized["initialized"] == state["initialized"]


class TestKalmanSerialization:
    def test_get_state_returns_dict(self):
        from regime.transition import _PyKalmanDerivativeTracker
        kf = _PyKalmanDerivativeTracker()
        state = kf.get_state()
        assert "x" in state
        assert "P" in state
        assert "initialized" in state

    def test_round_trip(self):
        from regime.transition import _PyKalmanDerivativeTracker
        kf1 = _PyKalmanDerivativeTracker()
        # Feed some data
        for v in [1.0, 1.1, 1.05, 0.95, 1.02, 1.08, 0.99, 1.01]:
            kf1.update(v)

        state = kf1.get_state()

        kf2 = _PyKalmanDerivativeTracker()
        kf2.restore_state(state)

        # Next 50 updates must produce identical output
        rng = np.random.RandomState(42)
        for _ in range(50):
            v = rng.randn() * 0.01 + 1.0
            vel1, acc1 = kf1.update(v)
            vel2, acc2 = kf2.update(v)
            assert abs(vel1 - vel2) < 1e-12, f"Velocity diverged: {vel1} != {vel2}"
            assert abs(acc1 - acc2) < 1e-12, f"Acceleration diverged: {acc1} != {acc2}"

    def test_state_is_json_serializable(self):
        import json
        from regime.transition import _PyKalmanDerivativeTracker
        kf = _PyKalmanDerivativeTracker()
        kf.update(1.0)
        state = kf.get_state()
        serialized = json.dumps(state)
        assert len(serialized) > 0


from regime.state_vector import StateVector


class TestTransitionDetectorSerialization:
    def _make_detector(self):
        from regime.transition import TransitionDetector
        return TransitionDetector()

    def _make_state_vector(self, rng):
        return StateVector(
            mu=float(rng.randn()),
            sigma=float(abs(rng.randn()) + 0.01),
            skew=float(rng.randn() * 0.5),
            kurtosis=float(abs(rng.randn()) + 2.0),
            tail_slope=float(rng.randn() * 0.1),
            entropy=float(abs(rng.randn()) + 1.0),
        )

    def test_get_state_returns_dict(self):
        td = self._make_detector()
        state = td.get_state()
        assert "kf_mu" in state
        assert "kf_entropy" in state
        assert "prev_regime" in state
        assert "pca_fitted" in state

    def test_round_trip(self):
        rng = np.random.RandomState(42)
        td1 = self._make_detector()

        # Feed enough data to initialize PCA and Kalman
        for i in range(100):
            sv = self._make_state_vector(rng)
            td1.update(curr_regime=i % 3, curr_state=sv)

        state = td1.get_state()

        td2 = self._make_detector()
        td2.restore_state(state)

        # Both must have same Kalman state
        np.testing.assert_array_almost_equal(
            td1._kf_mu._x if hasattr(td1._kf_mu, '_x') else [0],
            td2._kf_mu._x if hasattr(td2._kf_mu, '_x') else [0],
        )
        assert td1._pca_fitted == td2._pca_fitted
        assert td1._accumulator_count == td2._accumulator_count

    def test_state_is_json_serializable(self):
        import json
        td = self._make_detector()
        rng = np.random.RandomState(42)
        for i in range(10):
            sv = self._make_state_vector(rng)
            td.update(curr_regime=0, curr_state=sv)
        state = td.get_state()
        serialized = json.dumps(state)
        assert len(serialized) > 0


class TestReturnPredictorSerialization:
    def test_get_state_returns_dict(self):
        from alpha.return_predictor import ReturnPredictor, _USE_RUST_CORE
        # Use Python backend for deterministic testing
        if _USE_RUST_CORE:
            from alpha.return_predictor import _PyReturnPredictor as RP
        else:
            RP = ReturnPredictor
        rp = RP(n_features=5)
        state = rp.get_state()
        assert "weights" in state
        assert "bias" in state
        assert "n_updates" in state

    def test_round_trip(self):
        from alpha.return_predictor import PredictionResult
        try:
            from alpha.return_predictor import _PyReturnPredictor as RP
        except ImportError:
            from alpha.return_predictor import ReturnPredictor as RP

        rng = np.random.RandomState(42)
        rp1 = RP(n_features=5)

        # Train for a while
        for _ in range(50):
            features = rng.randn(5)
            rp1.predict(features)
            rp1.update(rng.randn() * 0.01)

        state = rp1.get_state()

        rp2 = RP(n_features=5)
        rp2.restore_state(state)

        # Next 100 predictions must match
        for _ in range(100):
            features = rng.randn(5)
            p1 = rp1.predict(features)
            p2 = rp2.predict(features)
            assert abs(p1.expected_return - p2.expected_return) < 1e-12
            actual = rng.randn() * 0.01
            rp1.update(actual)
            rp2.update(actual)

    def test_state_is_json_serializable(self):
        import json
        try:
            from alpha.return_predictor import _PyReturnPredictor as RP
        except ImportError:
            from alpha.return_predictor import ReturnPredictor as RP
        rp = RP(n_features=5)
        rp.predict(np.ones(5))
        rp.update(0.001)
        state = rp.get_state()
        serialized = json.dumps(state)
        assert len(serialized) > 0


class TestGMMSerialization:
    def test_get_state_unfitted(self):
        from microstructure.pdf.normalizing_flow import GMMDensityEstimator
        gmm = GMMDensityEstimator()
        state = gmm.get_state()
        assert state["fitted"] is False

    def test_round_trip_fitted(self):
        from microstructure.pdf.normalizing_flow import GMMDensityEstimator
        rng = np.random.RandomState(42)
        gmm1 = GMMDensityEstimator()

        # Fit with enough data
        data = rng.randn(200) * 0.5 + 100.0
        gmm1.fit(data)
        assert gmm1.fitted

        state = gmm1.get_state()

        gmm2 = GMMDensityEstimator()
        gmm2.restore_state(state)

        assert gmm2.fitted
        assert abs(gmm2._data_mean - gmm1._data_mean) < 1e-10
        assert abs(gmm2._data_std - gmm1._data_std) < 1e-10

        # PDF evaluation must produce identical results
        test_points = np.linspace(98, 102, 50)
        pdf1 = gmm1.evaluate(test_points)
        pdf2 = gmm2.evaluate(test_points)
        np.testing.assert_array_almost_equal(pdf1, pdf2, decimal=10)

    def test_state_is_json_serializable(self):
        import json
        from microstructure.pdf.normalizing_flow import GMMDensityEstimator
        rng = np.random.RandomState(42)
        gmm = GMMDensityEstimator()
        gmm.fit(rng.randn(200))
        state = gmm.get_state()
        serialized = json.dumps(state)
        assert len(serialized) > 0
