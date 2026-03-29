import numpy as np
import pytest
from microstructure.garch import GarchVolatility


class TestGarchVolatility:
    def test_fallback_when_insufficient_data(self):
        """With < min_obs data, should return rolling std."""
        garch = GarchVolatility(min_obs=100)
        data = np.random.RandomState(42).randn(50) * 0.01
        vol = garch.conditional_vol(data)
        expected = float(np.std(data, ddof=1))
        assert abs(vol - expected) < 1e-12

    def test_fallback_before_first_fit(self):
        """Before refit_interval ticks, should still work."""
        garch = GarchVolatility(refit_interval=200, min_obs=50)
        rng = np.random.RandomState(42)
        data = rng.randn(60) * 0.01
        vol = garch.conditional_vol(data)
        assert vol > 0
        assert np.isfinite(vol)

    def test_refit_produces_positive_vol(self):
        """After refit, conditional vol should be positive and finite."""
        garch = GarchVolatility(refit_interval=1, min_obs=50)
        rng = np.random.RandomState(42)
        returns = []
        vol = 0.01
        for _ in range(200):
            vol = 0.0001 + 0.1 * returns[-1]**2 if returns else 0.01
            vol = max(vol, 0.001)
            returns.append(rng.randn() * vol)
        data = np.array(returns)
        result = garch.conditional_vol(data)
        assert result > 0
        assert np.isfinite(result)

    def test_conditional_vol_differs_from_rolling_std(self):
        """After fitting, GARCH vol should differ from rolling std."""
        garch = GarchVolatility(refit_interval=1, min_obs=100)
        rng = np.random.RandomState(42)
        data = rng.randn(200) * 0.01
        garch_vol = garch.conditional_vol(data)
        rolling_std = float(np.std(data, ddof=1))
        assert garch_vol > 0
        # GARCH vol should differ from naive rolling std after fitting
        assert abs(garch_vol - rolling_std) > 1e-10

    def test_state_is_json_serializable(self):
        """GarchVolatility state must be serializable for checkpointing."""
        import json
        garch = GarchVolatility(refit_interval=1, min_obs=50)
        rng = np.random.RandomState(42)
        data = rng.randn(100) * 0.01
        garch.conditional_vol(data)
        state = garch.get_state()
        serialized = json.dumps(state)
        assert len(serialized) > 0

    def test_restore_state_round_trip(self):
        """Restored GarchVolatility produces same output."""
        garch1 = GarchVolatility(refit_interval=1, min_obs=50)
        rng = np.random.RandomState(42)
        data = rng.randn(100) * 0.01
        garch1.conditional_vol(data)

        state = garch1.get_state()
        garch2 = GarchVolatility(refit_interval=1, min_obs=50)
        garch2.restore_state(state)

        new_data = rng.randn(100) * 0.01
        v1 = garch1.conditional_vol(new_data)
        v2 = garch2.conditional_vol(new_data)
        assert abs(v1 - v2) < 1e-12
