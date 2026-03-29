import numpy as np
import pytest
from microstructure.market_features import VPINEstimator, KyleLambdaEstimator, AmihudEstimator
from alpha.feature_engine import FeatureEngine, FeatureVector
from regime.state_vector import StateVector


class TestVPINEstimator:
    def test_empty_returns_zero(self):
        vpin = VPINEstimator(volume_bucket_size=100, n_buckets=10)
        assert vpin.estimate() == 0.0

    def test_fills_bucket_and_produces_value(self):
        vpin = VPINEstimator(volume_bucket_size=100, n_buckets=10)
        rng = np.random.RandomState(42)
        price = 100.0
        for _ in range(50):
            price += rng.randn() * 0.1
            vpin.update(price, volume=rng.uniform(5, 20))
        result = vpin.estimate()
        assert 0.0 <= result <= 1.0

    def test_high_imbalance_produces_high_vpin(self):
        """All buys (rising prices) should produce VPIN near 1.0."""
        vpin = VPINEstimator(volume_bucket_size=50, n_buckets=5)
        price = 100.0
        for _ in range(100):
            price += 0.01  # monotonically rising
            vpin.update(price, volume=10.0)
        assert vpin.estimate() > 0.5


class TestKyleLambdaEstimator:
    def test_empty_returns_zero(self):
        kyle = KyleLambdaEstimator(window=50)
        assert kyle.estimate() == 0.0

    def test_insufficient_data_returns_zero(self):
        kyle = KyleLambdaEstimator(window=50)
        for _ in range(10):
            kyle.update(mid_change=0.01, signed_volume=100.0)
        assert kyle.estimate() == 0.0  # < 20 observations

    def test_positive_impact_produces_positive_lambda(self):
        """When buying pushes price up, lambda should be positive."""
        kyle = KyleLambdaEstimator(window=100)
        rng = np.random.RandomState(42)
        for _ in range(50):
            sv = rng.uniform(50, 200)
            mid_change = sv * 0.001 + rng.randn() * 0.001  # correlated
            kyle.update(mid_change=mid_change, signed_volume=sv)
        assert kyle.estimate() > 0

    def test_lambda_can_be_negative(self):
        """Contrarian flow: buying pushes price down."""
        kyle = KyleLambdaEstimator(window=100)
        rng = np.random.RandomState(42)
        for _ in range(50):
            sv = rng.uniform(50, 200)
            mid_change = -sv * 0.001 + rng.randn() * 0.001  # anti-correlated
            kyle.update(mid_change=mid_change, signed_volume=sv)
        assert kyle.estimate() < 0


class TestAmihudEstimator:
    def test_empty_returns_zero(self):
        amihud = AmihudEstimator(window=100)
        assert amihud.estimate() == 0.0

    def test_positive_ratio(self):
        amihud = AmihudEstimator(window=100)
        amihud.update(abs_return=0.01, volume=1000.0)
        assert amihud.estimate() == pytest.approx(0.01 / 1000.0)

    def test_zero_volume_ignored(self):
        amihud = AmihudEstimator(window=100)
        amihud.update(abs_return=0.01, volume=0.0)
        assert amihud.estimate() == 0.0

    def test_tiny_volume_ignored(self):
        """Volume below 1e-8 should be ignored to prevent outliers."""
        amihud = AmihudEstimator(window=100)
        amihud.update(abs_return=0.01, volume=1e-12)
        assert amihud.estimate() == 0.0

    def test_multiple_updates(self):
        amihud = AmihudEstimator(window=100)
        amihud.update(abs_return=0.01, volume=1000.0)
        amihud.update(abs_return=0.02, volume=2000.0)
        expected = np.mean([0.01/1000.0, 0.02/2000.0])
        assert amihud.estimate() == pytest.approx(expected)


class TestFeatureVector:
    def test_feature_count_is_38(self):
        """FeatureVector must produce exactly 38 features."""
        fv = FeatureVector()
        arr = fv.to_array()
        assert len(arr) == 38
        assert len(FeatureVector.feature_names()) == 38

    def test_interaction_features_in_names(self):
        names = FeatureVector.feature_names()
        for interaction in ['vpin_x_spread', 'kyle_x_regime_conf',
                            'amihud_x_vol_of_vol', 'ofi_x_book_pressure',
                            'vpin_x_kyle', 'fat_tail_active']:
            assert interaction in names

    def test_to_array_and_names_aligned(self):
        fv = FeatureVector(mu=1.0, vpin=0.5, spread_pct=0.002)
        arr = fv.to_array()
        names = FeatureVector.feature_names()
        assert arr[names.index('mu')] == 1.0
        assert arr[names.index('vpin')] == 0.5
        assert arr[names.index('spread_pct')] == 0.002


class TestFeatureEngineInteractions:
    def _make_engine_with_data(self, n_ticks=100):
        engine = FeatureEngine(window=200)
        rng = np.random.RandomState(42)
        price = 100.0
        for i in range(n_ticks):
            price += rng.randn() * 0.05
            bid = price - 0.01
            ask = price + 0.01
            volume = rng.uniform(50, 200)
            engine.update(price, volume, bid, ask, ofi=rng.randn(), timestamp=float(i))
        return engine

    def test_interaction_features_computed(self):
        engine = self._make_engine_with_data(150)
        state = StateVector(mu=0.001, sigma=0.02, skew=0.1,
                            kurtosis=1.5, tail_slope=0.5, entropy=2.0)
        fv = engine.compute(state, regime_confidence=0.8,
                            cluster_distance=0.5, regime_age=50,
                            ticks_since_transition=30)
        # VPIN × spread should be nonzero when both inputs are nonzero
        if fv.vpin > 0 and fv.spread_pct > 0:
            assert fv.vpin_x_spread > 0

    def test_fat_tail_active_flag(self):
        engine = self._make_engine_with_data(50)
        # tail_slope=0.5 > 0.333 → fat_tail_active should be 1.0
        state = StateVector(mu=0.0, sigma=0.01, skew=0.0,
                            kurtosis=0.0, tail_slope=0.5, entropy=1.0)
        fv = engine.compute(state)
        assert fv.fat_tail_active == 1.0

        # tail_slope=0.2 < 0.333 → fat_tail_active should be 0.0
        state2 = StateVector(mu=0.0, sigma=0.01, skew=0.0,
                             kurtosis=0.0, tail_slope=0.2, entropy=1.0)
        fv2 = engine.compute(state2)
        assert fv2.fat_tail_active == 0.0

    def test_mid_price_autocorrelation(self):
        """Autocorrelation should use mid-price returns, not trade returns."""
        engine = FeatureEngine(window=200)
        rng = np.random.RandomState(42)
        price = 100.0
        for i in range(200):
            price += rng.randn() * 0.05
            bid = price - 0.01
            ask = price + 0.01
            engine.update(price, rng.uniform(50, 200), bid, ask,
                          ofi=rng.randn(), timestamp=float(i))
        # Mid returns buffer should be populated
        assert len(engine._mid_returns) > 0
        state = StateVector(mu=0.0, sigma=0.01, skew=0.0,
                            kurtosis=0.0, tail_slope=0.0, entropy=1.0)
        fv = engine.compute(state)
        # Autocorrelation should be computed (non-default)
        assert fv.autocorr_1 != 0.0 or fv.autocorr_5 != 0.0

    def test_reset_clears_mid_returns(self):
        engine = self._make_engine_with_data(50)
        assert len(engine._mid_returns) > 0
        engine.reset()
        assert len(engine._mid_returns) == 0
