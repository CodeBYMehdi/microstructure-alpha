import numpy as np
import copy


# --- Config ---

def test_hmm_config_loads():
    from config.loader import get_config
    config = get_config()
    hmm = config.thresholds.regime.hmm
    assert hmm.n_states == 4
    assert 0 < hmm.learning_rate < 1
    assert hmm.warmup_ticks > 0
    assert 0 < hmm.min_confidence < 1
    assert hmm.emission_reg > 0


# --- Core HMM ---

def test_hmm_init():
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=4, n_features=6)
    assert hmm.n_states == 4
    assert hmm.n_features == 6
    assert hmm.transition_matrix.shape == (4, 4)
    assert hmm.start_prob.shape == (4,)
    for row in hmm.transition_matrix:
        assert abs(np.sum(row) - 1.0) < 1e-6


def test_hmm_filter_step():
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=3, n_features=6)
    # Feed warmup data first
    for i in range(60):
        obs = np.random.randn(6) * 0.01
        hmm.filter_step(obs)
    # Now should be initialized
    assert hmm.is_initialized
    obs = np.array([0.001, 0.02, 0.1, 3.0, 2.0, -5.0])
    state, posterior = hmm.filter_step(obs)
    assert 0 <= state < 3
    assert len(posterior) == 3
    assert abs(np.sum(posterior) - 1.0) < 1e-6


def test_hmm_learns_two_regimes():
    from regime.hmm import GaussianHMM
    rng = np.random.RandomState(42)
    hmm = GaussianHMM(n_states=2, n_features=6, learning_rate=0.1, seed=42)

    # Two extremely well-separated regimes (10+ std apart)
    center_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    center_b = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    # Interleave during warmup so k-means++ sees BOTH regimes for initialization
    warmup_a = rng.randn(30, 6) * 0.1 + center_a
    warmup_b = rng.randn(30, 6) * 0.1 + center_b
    warmup = np.vstack([warmup_a, warmup_b])
    rng.shuffle(warmup)

    # Then present clean blocks for regime detection
    regime_a = rng.randn(300, 6) * 0.1 + center_a
    regime_b = rng.randn(300, 6) * 0.1 + center_b
    data = np.vstack([warmup, regime_a, regime_b])

    states = []
    for obs in data:
        s, _ = hmm.filter_step(obs)
        hmm.online_update(obs)
        states.append(s)

    # After learning, the two blocks should map to different states
    from collections import Counter
    # warmup=60, then regime_a=300 (indices 60-360), regime_b=300 (indices 360-660)
    a_counts = Counter(states[260:360])
    b_counts = Counter(states[560:660])
    dominant_a = a_counts.most_common(1)[0][0]
    dominant_b = b_counts.most_common(1)[0][0]
    assert dominant_a != dominant_b, "HMM should distinguish two well-separated regimes"


def test_hmm_transition_probability():
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=3, n_features=6)
    prob = hmm.get_transition_prob(0, 1)
    assert 0.0 <= prob <= 1.0
    # Self-transition should be higher (sticky prior)
    self_prob = hmm.get_transition_prob(0, 0)
    assert self_prob > prob


def test_hmm_nan_guard():
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=2, n_features=6)
    # Feed warmup
    for _ in range(60):
        hmm.filter_step(np.random.randn(6) * 0.01)
    assert hmm.is_initialized

    # Get clean state
    clean_obs = np.array([0.001, 0.02, 0.1, 3.0, 2.0, -5.0])
    state1, post1 = hmm.filter_step(clean_obs)

    # NaN observation should not corrupt state
    nan_obs = np.array([np.nan, 0.02, 0.1, 3.0, 2.0, -5.0])
    state2, post2 = hmm.filter_step(nan_obs)
    assert np.all(np.isfinite(post2))

    # Inf observation
    inf_obs = np.array([np.inf, 0.02, 0.1, 3.0, 2.0, -5.0])
    state3, post3 = hmm.filter_step(inf_obs)
    assert np.all(np.isfinite(post3))

    # Online update with NaN should be no-op
    means_before = hmm.means.copy()
    hmm.online_update(nan_obs)
    assert np.allclose(hmm.means, means_before)


def test_hmm_covariance_stays_bounded():
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=2, n_features=6, learning_rate=0.01)
    # Feed warmup
    for _ in range(60):
        hmm.filter_step(np.random.randn(6) * 0.01)

    # Run 5000 updates — covariances should not explode
    for i in range(5000):
        obs = np.random.randn(6) * 0.01
        hmm.filter_step(obs)
        hmm.online_update(obs)

    for k in range(hmm.n_states):
        cov = hmm.get_covariance(k)
        max_diag = np.max(np.diag(cov))
        assert max_diag < 10.0, f"Covariance exploded: max diagonal = {max_diag}"
        assert np.all(np.isfinite(cov)), "Covariance has non-finite values"


# --- Adapter ---

def test_hmm_adapter_interface():
    from regime.hmm_adapter import HMMRegimeAdapter
    adapter = HMMRegimeAdapter()
    assert hasattr(adapter, 'update')
    assert hasattr(adapter, 'fit')
    assert hasattr(adapter, 'predict_latest')
    assert hasattr(adapter, 'get_latest_regime_output')
    assert hasattr(adapter, 'calculate_confidence')
    assert hasattr(adapter, 'get_regime_stats')
    assert hasattr(adapter, 'get_cluster_quality')
    assert hasattr(adapter, 'n_regimes')


def test_hmm_adapter_update_and_predict():
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()

    for i in range(300):
        sv = StateVector(
            mu=0.001 * np.sin(i * 0.1),
            sigma=0.02 + 0.001 * np.cos(i * 0.05),
            skew=0.1 * np.sin(i * 0.02),
            kurtosis=3.0 + 0.5 * np.cos(i * 0.03),
            tail_slope=2.0,
            entropy=-5.0 + 0.1 * np.sin(i * 0.01),
        )
        adapter.update(sv)
        adapter.fit()

    regime_id = adapter.predict_latest()
    assert isinstance(regime_id, int)
    assert regime_id >= 0


def test_hmm_adapter_regime_output():
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector
    from core.types import RegimeOutput

    adapter = HMMRegimeAdapter()
    sv = StateVector(mu=0.001, sigma=0.02, skew=0.1, kurtosis=3.0, tail_slope=2.0, entropy=-5.0)

    # Before warmup
    output = adapter.get_latest_regime_output(sv)
    assert isinstance(output, RegimeOutput)
    assert output.is_noise

    # After warmup: use varied data so HMM converges to distinct states
    rng = np.random.RandomState(99)
    for i in range(300):
        sv_i = StateVector(
            mu=0.001 + rng.randn() * 0.005,
            sigma=0.02 + abs(rng.randn()) * 0.005,
            skew=0.1 + rng.randn() * 0.3,
            kurtosis=3.0 + abs(rng.randn()) * 0.5,
            tail_slope=2.0 + rng.randn() * 0.2,
            entropy=-5.0 + rng.randn() * 0.3,
        )
        adapter.update(sv_i)
        adapter.fit()

    output = adapter.get_latest_regime_output(sv)
    assert isinstance(output, RegimeOutput)
    assert 0.0 <= output.confidence <= 1.0


def test_hmm_adapter_regime_stats_format():
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    rng = np.random.RandomState(77)
    for i in range(300):
        sv = StateVector(
            mu=0.001 + rng.randn() * 0.005,
            sigma=0.02 + abs(rng.randn()) * 0.005,
            skew=0.1 + rng.randn() * 0.3,
            kurtosis=3.0 + abs(rng.randn()) * 0.5,
            tail_slope=2.0 + rng.randn() * 0.2,
            entropy=-5.0 + rng.randn() * 0.3,
        )
        adapter.update(sv)
        adapter.fit()

    regime_id = adapter.predict_latest()
    # After varied data, should have a valid regime
    assert regime_id >= 0, f"Expected valid regime, got {regime_id}"
    stats = adapter.get_regime_stats(regime_id)
    assert stats is not None
    assert 'centroid' in stats
    assert 'std' in stats
    assert isinstance(stats['centroid'], StateVector)
    assert isinstance(stats['std'], StateVector)


def test_hmm_adapter_transition_info():
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    for i in range(300):
        sv = StateVector(
            mu=0.001, sigma=0.02, skew=0.1,
            kurtosis=3.0, tail_slope=2.0, entropy=-5.0,
        )
        adapter.update(sv)
        adapter.fit()

    info = adapter.get_transition_info()
    assert 'transition_matrix' in info
    assert 'current_state' in info
    assert 'posterior' in info
    assert info['is_initialized']
    # Transition matrix rows should sum to 1
    for row in info['transition_matrix']:
        assert abs(np.sum(row) - 1.0) < 1e-6


def test_hmm_adapter_cluster_quality():
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    for i in range(300):
        sv = StateVector(
            mu=0.001, sigma=0.02, skew=0.1,
            kurtosis=3.0, tail_slope=2.0, entropy=-5.0,
        )
        adapter.update(sv)
        adapter.fit()

    quality = adapter.get_cluster_quality()
    assert 'n_clusters' in quality
    assert 'noise_ratio' in quality
    assert 'n_samples' in quality
    assert quality['model'] == 'HMM'


# --- Integration ---

def test_transition_detector_with_hmm_probability():
    from regime.transition import TransitionDetector
    from regime.state_vector import StateVector

    detector = TransitionDetector()

    s0 = StateVector(mu=0.001, sigma=0.02, skew=0.1, kurtosis=3.0, tail_slope=2.0, entropy=-5.0)
    s1 = StateVector(mu=-0.002, sigma=0.03, skew=-0.2, kurtosis=4.0, tail_slope=1.5, entropy=-4.0)

    event = detector.update(0, s0, hmm_transition_prob=0.8)
    assert event is None  # First call, no prev state

    event = detector.update(1, s1, hmm_transition_prob=0.85)
    assert event is not None
    assert event.from_regime == 0
    assert event.to_regime == 1


def test_strategy_with_hmm_does_not_crash():
    from config.loader import get_config
    from core.types import Tick
    from datetime import datetime, timedelta

    config = copy.deepcopy(get_config())
    from main import Strategy
    from regime.hmm_adapter import HMMRegimeAdapter

    strategy = Strategy(config=config, enable_viz=False)
    assert isinstance(strategy.clustering, HMMRegimeAdapter)

    base_time = datetime(2024, 1, 1)
    for i in range(500):
        tick = Tick(
            timestamp=base_time + timedelta(milliseconds=i * 100),
            symbol="TEST",
            price=100.0 + 0.01 * np.sin(i * 0.05),
            volume=100.0,
        )
        strategy.on_tick(tick)

    assert strategy.tick_count == 500


# --- Hardening fix tests ---

def test_feature_scaling_in_adapter():
    """Adapter scales observations before HMM (fix #1)."""
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    rng = np.random.RandomState(55)
    for i in range(300):
        sv = StateVector(
            mu=0.001 + rng.randn() * 0.005,
            sigma=0.02 + abs(rng.randn()) * 0.005,
            skew=0.1 + rng.randn() * 0.3,
            kurtosis=3.0 + abs(rng.randn()) * 0.5,
            tail_slope=2.0 + rng.randn() * 0.2,
            entropy=-5.0 + rng.randn() * 0.3,
        )
        adapter.update(sv)
        adapter.fit()

    # Scaler should be fitted
    assert adapter._scaler_fitted
    # HMM means should be in scaled space (near 0)
    for k in range(adapter._hmm.n_states):
        mean = adapter._hmm.get_mean(k)
        assert np.all(np.abs(mean) < 10), f"Scaled mean too large: {mean}"


def test_entropy_regularization_prevents_collapse():
    """Transition matrix doesn't collapse to identity (fix #2)."""
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=4, n_features=6, learning_rate=0.01, seed=42)

    # Feed identical data — would normally cause collapse to one state
    for _ in range(60):
        hmm.filter_step(np.zeros(6))
    for _ in range(2000):
        obs = np.zeros(6) + np.random.randn(6) * 0.001
        hmm.filter_step(obs)
        hmm.online_update(obs)

    # No row should be more than 98% self-transition (entropy reg prevents it)
    for i in range(hmm.n_states):
        self_prob = hmm.transition_matrix[i, i]
        assert self_prob < 0.98, f"State {i} self-transition {self_prob:.4f} — degenerate"


def test_adaptive_learning_rate():
    """Learning rate increases on innovation spike (fix #3)."""
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=2, n_features=6, learning_rate=0.01, seed=42)

    # Warmup with one regime
    for _ in range(60):
        hmm.filter_step(np.ones(6) * 0.5 + np.random.randn(6) * 0.01)
    base_lr = hmm.learning_rate

    # Big innovation — switch to very different data
    hmm.filter_step(np.ones(6) * 5.0)  # 10x away from learned mean
    spike_lr = hmm.learning_rate

    # lr should have increased
    assert spike_lr > base_lr, f"Expected adaptive lr increase, got {spike_lr:.6f} <= {base_lr:.6f}"


def test_post_init_mean_separation():
    """Means are separated after initialization (fix #5)."""
    from regime.hmm import GaussianHMM
    hmm = GaussianHMM(n_states=4, n_features=6, seed=42)

    # Feed near-identical data (would cause means to cluster)
    for _ in range(60):
        hmm.filter_step(np.ones(6) * 0.1 + np.random.randn(6) * 0.0001)

    assert hmm.is_initialized
    # Check pairwise distances
    for k in range(hmm.n_states):
        for j in range(k + 1, hmm.n_states):
            dist = np.linalg.norm(hmm.means[k] - hmm.means[j])
            assert dist > 0.01, f"States {k},{j} too close: dist={dist:.6f}"


def test_transition_info_has_volatility():
    """Adapter exposes transition_volatility (fix #8)."""
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    rng = np.random.RandomState(44)
    for i in range(300):
        sv = StateVector(
            mu=rng.randn() * 0.005, sigma=0.02 + abs(rng.randn()) * 0.005,
            skew=rng.randn() * 0.3, kurtosis=3.0 + abs(rng.randn()) * 0.5,
            tail_slope=2.0, entropy=-5.0 + rng.randn() * 0.3,
        )
        adapter.update(sv)
        adapter.fit()

    info = adapter.get_transition_info()
    assert 'transition_volatility' in info
    assert 0.0 <= info['transition_volatility'] <= 1.0


def test_posterior_entropy_in_confidence():
    """Confidence is discounted by posterior entropy (fix #7)."""
    from regime.hmm_adapter import HMMRegimeAdapter
    from regime.state_vector import StateVector

    adapter = HMMRegimeAdapter()
    rng = np.random.RandomState(66)
    for i in range(300):
        sv = StateVector(
            mu=rng.randn() * 0.005, sigma=0.02 + abs(rng.randn()) * 0.005,
            skew=rng.randn() * 0.3, kurtosis=3.0 + abs(rng.randn()) * 0.5,
            tail_slope=2.0, entropy=-5.0 + rng.randn() * 0.3,
        )
        adapter.update(sv)
        adapter.fit()

    regime_id = adapter.predict_latest()
    if regime_id >= 0:
        raw_posterior = float(adapter._current_posterior[regime_id])
        adjusted = adapter.calculate_confidence(sv, regime_id)
        # Adjusted should be <= raw (entropy discount)
        assert adjusted <= raw_posterior + 1e-6
