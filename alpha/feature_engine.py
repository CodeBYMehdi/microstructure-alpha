# on cherche les pepites
# la grosse machine

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging

from regime.state_vector import StateVector

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    # on cherche les pepites
    # la grosse machine
    # Microstructure moments
    mu: float = 0.0
    sigma: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    tail_slope: float = 0.0
    entropy: float = 0.0

    # Order flow features
    ofi: float = 0.0               # Order Flow Imbalance
    vpin: float = 0.0              # Volume-sync probability of informed trading
    depth_imbalance: float = 0.0   # (bid_vol - ask_vol) / total
    trade_flow_toxicity: float = 0.0
    kyle_lambda: float = 0.0
    amihud: float = 0.0

    # Autocorrelation features
    autocorr_1: float = 0.0        # Lag-1 autocorrelation
    autocorr_5: float = 0.0        # Lag-5 autocorrelation
    autocorr_10: float = 0.0       # Lag-10

    # Cross-window dynamics
    momentum_short: float = 0.0    # 50-tick return sum
    momentum_long: float = 0.0     # 200-tick return sum
    mean_reversion: float = 0.0    # Z-score of price relative to rolling mean
    vol_of_vol: float = 0.0        # Volatility of volatility

    # Regime context
    regime_age: int = 0
    regime_confidence: float = 0.0
    ticks_since_transition: int = 0
    cluster_distance: float = 0.0

    # Derived
    spread_pct: float = 0.0        # Spread as pct of mid price
    volume_intensity: float = 0.0  # Volume per unit time

    # L2 Order Book features (from L2OrderBook.get_features())
    liquidity_pull_score: float = 0.0   # Exponentially decayed liquidity pull severity
    book_pressure: float = 0.0          # Depth-weighted price asymmetry [-1, 1]
    spoofing_score: float = 0.0         # Flash-size spoofing detection score
    l2_spread_bps: float = 0.0          # L2 spread in basis points
    l2_depth_imbalance: float = 0.0     # L2 full-depth imbalance (more levels than tick-level)

    # 3D Surface features
    density_velocity: float = 0.0
    regime_trajectory_z: float = 0.0

    # Feature interactions (nonlinear cross-terms for microstructure edge)
    vpin_x_spread: float = 0.0            # VPIN × spread: informed trading under wide spreads
    kyle_x_regime_conf: float = 0.0       # Kyle λ × regime confidence: impact when regime is clear
    amihud_x_vol_of_vol: float = 0.0      # Amihud × vol-of-vol: illiquidity during vol regimes
    ofi_x_book_pressure: float = 0.0      # OFI × book pressure: flow aligned with depth
    vpin_x_kyle: float = 0.0              # VPIN × Kyle λ: informed flow with price impact

    # Transformed features
    fat_tail_active: float = 0.0          # Binary: Hill alpha < 3 (heavy tails present)

    def to_array(self) -> np.ndarray:
        # la grosse machine
        return np.array([
            self.mu, self.sigma, self.skew, self.kurtosis,
            self.tail_slope, self.entropy,
            self.ofi, self.vpin, self.depth_imbalance, self.trade_flow_toxicity,
            self.kyle_lambda, self.amihud,
            self.autocorr_1, self.autocorr_5, self.autocorr_10,
            self.momentum_short, self.momentum_long,
            self.mean_reversion, self.vol_of_vol,
            self.regime_age, self.regime_confidence,
            self.ticks_since_transition, self.cluster_distance,
            self.spread_pct, self.volume_intensity,
            self.liquidity_pull_score, self.book_pressure,
            self.spoofing_score, self.l2_spread_bps, self.l2_depth_imbalance,
            self.density_velocity, self.regime_trajectory_z,
            self.vpin_x_spread, self.kyle_x_regime_conf,
            self.amihud_x_vol_of_vol, self.ofi_x_book_pressure,
            self.vpin_x_kyle, self.fat_tail_active,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            'mu', 'sigma', 'skew', 'kurtosis', 'tail_slope', 'entropy',
            'ofi', 'vpin', 'depth_imbalance', 'trade_flow_toxicity',
            'kyle_lambda', 'amihud',
            'autocorr_1', 'autocorr_5', 'autocorr_10',
            'momentum_short', 'momentum_long',
            'mean_reversion', 'vol_of_vol',
            'regime_age', 'regime_confidence',
            'ticks_since_transition', 'cluster_distance',
            'spread_pct', 'volume_intensity',
            'liquidity_pull_score', 'book_pressure',
            'spoofing_score', 'l2_spread_bps', 'l2_depth_imbalance',
            'density_velocity', 'regime_trajectory_z',
            'vpin_x_spread', 'kyle_x_regime_conf',
            'amihud_x_vol_of_vol', 'ofi_x_book_pressure',
            'vpin_x_kyle', 'fat_tail_active',
        ]


class FeatureEngine:
    # on cherche les pepites
    # la tuyauterie de donnees

    def __init__(self, window: int = 200, vol_lookback: int = 50):
        self.window = window
        self.vol_lookback = vol_lookback

        # Rolling buffers
        self._returns: deque = deque(maxlen=window * 2)
        self._mid_returns: deque = deque(maxlen=window * 2)  # mid-price returns for autocorrelation
        self._prices: deque = deque(maxlen=window * 2)
        self._volumes: deque = deque(maxlen=window)
        self._spreads: deque = deque(maxlen=window)
        self._sigma_history: deque = deque(maxlen=vol_lookback)
        self._ofi_history: deque = deque(maxlen=window)
        self._timestamps: deque = deque(maxlen=window)

        # Market feature estimators (VPIN, Kyle lambda, Amihud)
        from microstructure.market_features import VPINEstimator, KyleLambdaEstimator, AmihudEstimator
        self._vpin_estimator = VPINEstimator(volume_bucket_size=500, n_buckets=50)
        self._kyle_estimator = KyleLambdaEstimator(window=100)
        self._amihud_estimator = AmihudEstimator(window=200)
        self._last_mid: Optional[float] = None

        self._last_price: Optional[float] = None
        self._tick_count: int = 0

        logger.info(f"FeatureEngine initialized: window={window}")

    def update(self, price: float, volume: float, bid: Optional[float],
               ask: Optional[float], ofi: float, timestamp: float) -> None:
        # la tuyauterie de donnees
        self._tick_count += 1
        self._prices.append(price)
        self._volumes.append(volume)
        self._timestamps.append(timestamp)
        self._ofi_history.append(ofi)

        # Return
        if self._last_price and self._last_price > 0:
            ret = np.log(price / self._last_price)
            if np.isfinite(ret):
                self._returns.append(ret)
        self._last_price = price

        # Spread
        if bid and ask and ask > bid:
            spread_pct = (ask - bid) / ((ask + bid) / 2)
            self._spreads.append(spread_pct)

        # Feed market feature estimators
        self._vpin_estimator.update(price, volume)
        if bid and ask:
            mid = (bid + ask) / 2
            if self._last_mid is not None and self._last_mid > 0:
                mid_change = mid - self._last_mid
                signed_vol = volume if price >= mid else -volume
                self._kyle_estimator.update(mid_change, signed_vol)
                # Mid-price returns for autocorrelation (avoids bid-ask bounce)
                mid_ret = np.log(mid / self._last_mid)
                if np.isfinite(mid_ret):
                    self._mid_returns.append(mid_ret)
            self._last_mid = mid
        if self._returns:
            self._amihud_estimator.update(abs(self._returns[-1]), volume)

    def compute(self, state: StateVector, regime_age: int = 0,
                regime_confidence: float = 0.0, ticks_since_transition: int = 0,
                cluster_distance: float = 0.0, depth_imbalance: float = 0.0,
                l2_features: dict = None, surface_state: Optional['SurfaceState'] = None) -> FeatureVector:
        # la calculette
        # le feu vert
        returns = np.array(self._returns) if self._returns else np.array([0.0])

        fv = FeatureVector()

        # 1. Microstructure moments (from state vector)
        fv.mu = state.mu
        fv.sigma = state.sigma
        fv.skew = state.skew
        fv.kurtosis = state.kurtosis
        fv.tail_slope = state.tail_slope
        fv.entropy = state.entropy

        # 2. Order flow features
        fv.ofi = float(np.sum(list(self._ofi_history))) if self._ofi_history else 0.0
        fv.vpin = self._vpin_estimator.estimate()
        fv.kyle_lambda = self._kyle_estimator.estimate()
        fv.amihud = self._amihud_estimator.estimate()
        fv.depth_imbalance = depth_imbalance

        # Trade flow toxicity: OFI normalized by volume
        total_vol = sum(self._volumes) if self._volumes else 1.0
        fv.trade_flow_toxicity = fv.ofi / max(total_vol, 1.0)

        # 3. Autocorrelation — use mid-price returns to avoid bid-ask bounce contamination
        mid_rets = np.array(self._mid_returns) if self._mid_returns else returns
        mid_lagged = mid_rets[:-1] if len(mid_rets) > 1 else mid_rets
        if len(mid_lagged) >= 20:
            fv.autocorr_1 = self._autocorr(mid_lagged, 1)
            fv.autocorr_5 = self._autocorr(mid_lagged, 5)
            fv.autocorr_10 = self._autocorr(mid_lagged, 10)

        # 4. Cross-window dynamics — lagged to prevent lookahead (trade returns ok for momentum)
        returns_lagged = returns[:-1] if len(returns) > 1 else returns
        if len(returns_lagged) >= 50:
            fv.momentum_short = float(np.sum(returns_lagged[-50:]))
        if len(returns_lagged) >= 200:
            fv.momentum_long = float(np.sum(returns_lagged[-200:]))

        # Mean reversion: Z-score of current price
        if len(self._prices) >= 50:
            prices = np.array(list(self._prices)[-200:])
            p_mean = np.mean(prices)
            p_std = np.std(prices)
            if p_std > 0:
                fv.mean_reversion = (self._prices[-1] - p_mean) / p_std

        # Vol of vol
        self._sigma_history.append(state.sigma)
        if len(self._sigma_history) >= 10:
            fv.vol_of_vol = float(np.std(list(self._sigma_history)))

        # 5. Regime context
        fv.regime_age = regime_age
        fv.regime_confidence = regime_confidence
        fv.ticks_since_transition = ticks_since_transition
        fv.cluster_distance = cluster_distance

        # 6. Derived
        if self._spreads:
            fv.spread_pct = float(np.mean(list(self._spreads)[-20:]))

        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            if dt > 0:
                fv.volume_intensity = float(sum(self._volumes)) / dt

        # 7. L2 Order Book features
        if l2_features:
            fv.liquidity_pull_score = l2_features.get('liquidity_pull_score', 0.0)
            fv.book_pressure = l2_features.get('book_pressure', 0.0)
            fv.spoofing_score = l2_features.get('spoofing_score', 0.0)
            fv.l2_spread_bps = l2_features.get('spread_bps', 0.0)
            fv.l2_depth_imbalance = l2_features.get('depth_imbalance', 0.0)

        # 8. 3D Surface Analytics
        if surface_state:
            fv.density_velocity = surface_state.density_velocity
            fv.regime_trajectory_z = surface_state.regime_trajectory_z

        # 9. Feature interactions — capture nonlinear microstructure edges
        fv.vpin_x_spread = fv.vpin * fv.spread_pct
        fv.kyle_x_regime_conf = fv.kyle_lambda * fv.regime_confidence
        fv.amihud_x_vol_of_vol = fv.amihud * fv.vol_of_vol
        fv.ofi_x_book_pressure = fv.ofi * fv.book_pressure
        fv.vpin_x_kyle = fv.vpin * fv.kyle_lambda

        # 10. Fat tail flag — Hill alpha < 3 means heavy tails present
        # tail_slope = 1/alpha, so fat tails when tail_slope > 1/3 ≈ 0.333
        fv.fat_tail_active = 1.0 if fv.tail_slope > 0.333 else 0.0

        return fv

    @staticmethod
    def _autocorr(x: np.ndarray, lag: int) -> float:
        # la calculette
        if len(x) <= lag:
            return 0.0
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        if var < 1e-15:
            return 0.0
        c = np.sum((x[lag:] - mean) * (x[:n - lag] - mean)) / (n * var)
        return float(np.clip(c, -1.0, 1.0))

    def reset(self) -> None:
        # dans quel etat j'erre
        self._returns.clear()
        self._mid_returns.clear()
        self._prices.clear()
        self._volumes.clear()
        self._spreads.clear()
        self._sigma_history.clear()
        self._ofi_history.clear()
        self._timestamps.clear()
        from microstructure.market_features import VPINEstimator, KyleLambdaEstimator, AmihudEstimator
        self._vpin_estimator = VPINEstimator(volume_bucket_size=500, n_buckets=50)
        self._kyle_estimator = KyleLambdaEstimator(window=100)
        self._amihud_estimator = AmihudEstimator(window=200)
        self._last_mid = None
        self._last_price = None
        self._tick_count = 0
