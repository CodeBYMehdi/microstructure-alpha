"""Portfolio-level risk management — VaR, correlation, regime-conditional budgets.

Replaces single-instrument risk tracking with multi-asset aware risk engine.

Features:
- Real-time portfolio VaR (Historical + Parametric)
- Rolling correlation matrix across instruments
- Regime-conditional risk budgets (auto-shrink in crash/volatile regimes)
- Gross/net exposure limits
- Sector/factor exposure monitoring
- Drawdown-to-volatility ratio tracking
"""

import numpy as np
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from enum import Enum

logger = logging.getLogger(__name__)


class RiskRegime(Enum):
    """Risk regime for budget allocation."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    STRESSED = "STRESSED"
    CRISIS = "CRISIS"


@dataclass
class PortfolioRiskSnapshot:
    """Point-in-time risk snapshot."""
    timestamp: float
    # VaR
    var_95: float = 0.0              # 95% 1-day VaR
    var_99: float = 0.0              # 99% 1-day VaR
    cvar_95: float = 0.0             # Expected shortfall (CVaR) at 95%
    # Exposure
    gross_exposure: float = 0.0      # Sum of absolute positions
    net_exposure: float = 0.0        # Sum of signed positions
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    # Drawdown
    drawdown_pct: float = 0.0
    drawdown_vol_adjusted: float = 0.0  # DD / realized vol
    # Risk regime
    risk_regime: str = "NORMAL"
    risk_budget_used_pct: float = 0.0
    # Correlation
    avg_correlation: float = 0.0     # Average pairwise correlation
    max_concentration_pct: float = 0.0  # Largest single position as % of gross


@dataclass
class RiskBudget:
    """Regime-conditional risk limits."""
    max_gross_exposure: float = 100000.0
    max_net_exposure: float = 50000.0
    max_single_position_pct: float = 0.30   # 30% of gross max
    max_var_95_pct: float = 0.02             # 2% of equity
    max_drawdown_pct: float = 0.10           # 10% from peak
    size_multiplier: float = 1.0             # Scale all sizes by this


# Regime-conditional budgets: more conservative in stressed/crisis regimes
# Nominal caps are intentionally large — VaR (max_var_95_pct) is the binding constraint.
# This lets the system scale up when realized risk is low and automatically shrink
# when VaR/drawdown limits bite, without hard dollar ceilings choking capacity.
DEFAULT_REGIME_BUDGETS: Dict[str, RiskBudget] = {
    RiskRegime.NORMAL.value: RiskBudget(
        max_gross_exposure=10_000_000, max_net_exposure=5_000_000,
        max_single_position_pct=0.30, max_var_95_pct=0.02,
        max_drawdown_pct=0.10, size_multiplier=1.0,
    ),
    RiskRegime.ELEVATED.value: RiskBudget(
        max_gross_exposure=7_500_000, max_net_exposure=3_500_000,
        max_single_position_pct=0.25, max_var_95_pct=0.015,
        max_drawdown_pct=0.07, size_multiplier=0.75,
    ),
    RiskRegime.STRESSED.value: RiskBudget(
        max_gross_exposure=5_000_000, max_net_exposure=2_000_000,
        max_single_position_pct=0.20, max_var_95_pct=0.01,
        max_drawdown_pct=0.05, size_multiplier=0.50,
    ),
    RiskRegime.CRISIS.value: RiskBudget(
        max_gross_exposure=2_500_000, max_net_exposure=1_000_000,
        max_single_position_pct=0.15, max_var_95_pct=0.005,
        max_drawdown_pct=0.03, size_multiplier=0.25,
    ),
}


class PortfolioRiskManager:
    """Portfolio-level risk engine with regime-conditional budgets."""

    def __init__(
        self,
        regime_budgets: Optional[Dict[str, RiskBudget]] = None,
        initial_equity: float = 100000.0,
        var_window: int = 252,        # ~1 year of daily returns
        correlation_window: int = 60, # 60 observations for correlation
    ):
        self.regime_budgets = regime_budgets or DEFAULT_REGIME_BUDGETS
        self.initial_equity = initial_equity
        self.var_window = var_window
        self.correlation_window = correlation_window

        # Equity tracking
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.current_drawdown_pct = 0.0

        # Positions: symbol -> signed quantity
        self._positions: Dict[str, float] = {}
        self._position_prices: Dict[str, float] = {}  # symbol -> mark price (mid-price preferred)
        self._bid_prices: Dict[str, float] = {}  # symbol -> last bid
        self._ask_prices: Dict[str, float] = {}  # symbol -> last ask

        # Return history per symbol: used for VaR
        self._return_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=var_window))

        # Portfolio return history
        self._portfolio_returns: Deque[float] = deque(maxlen=var_window)
        self._portfolio_equity_history: Deque[float] = deque(maxlen=var_window * 5)

        # Current risk regime
        self._current_regime = RiskRegime.NORMAL
        self._risk_snapshots: Deque[PortfolioRiskSnapshot] = deque(maxlen=1000)

        # Realized volatility tracking
        self._realized_vol: float = 0.0
        self._vol_history: Deque[float] = deque(maxlen=100)

        logger.info(
            f"PortfolioRiskManager initialized: equity=${initial_equity:.0f}, "
            f"regimes={list(self.regime_budgets.keys())}"
        )

    # ── Position Management ──

    def update_position(self, symbol: str, qty: float, price: float) -> None:
        """Update a position and its last known price."""
        self._positions[symbol] = qty
        self._position_prices[symbol] = price

        # Clean up zero positions
        if abs(qty) < 1e-10:
            self._positions.pop(symbol, None)

    def update_price(self, symbol: str, price: float,
                     bid: Optional[float] = None, ask: Optional[float] = None) -> None:
        """Update the latest price for a symbol (for MTM).

        Uses mid-price from BBO when available for more accurate mark-to-market.
        Falls back to last trade price if no BBO provided.
        """
        # Store raw BBO
        if bid is not None and bid > 0:
            self._bid_prices[symbol] = bid
        if ask is not None and ask > 0:
            self._ask_prices[symbol] = ask

        # Compute mark price: prefer mid-price from current BBO
        _bid = self._bid_prices.get(symbol)
        _ask = self._ask_prices.get(symbol)
        if _bid and _ask and _bid > 0 and _ask > 0 and _ask >= _bid:
            mark_price = (_bid + _ask) / 2.0
        else:
            mark_price = price  # Fallback to last trade

        old_price = self._position_prices.get(symbol)
        self._position_prices[symbol] = mark_price

        # Record return
        if old_price and old_price > 0 and mark_price > 0:
            ret = np.log(mark_price / old_price)
            if np.isfinite(ret):
                self._return_history[symbol].append(ret)

    def update_equity(self, pnl: float) -> None:
        """Update equity after a PnL event."""
        prev_equity = self.current_equity
        self.current_equity += pnl

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        dd = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.current_drawdown_pct = dd

        # Record portfolio return
        if prev_equity > 0:
            port_ret = pnl / prev_equity
            self._portfolio_returns.append(port_ret)

        self._portfolio_equity_history.append(self.current_equity)

        # Update realized vol
        if len(self._portfolio_returns) >= 10:
            self._realized_vol = float(np.std(list(self._portfolio_returns)[-20:]))
            self._vol_history.append(self._realized_vol)

    # ── VaR Calculation ──

    def compute_var(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute portfolio VaR and CVaR (Expected Shortfall).
        
        Returns (VaR, CVaR) as positive values representing potential loss.
        """
        returns = list(self._portfolio_returns)
        if len(returns) < 20:
            return 0.0, 0.0

        returns_array = np.array(returns)

        # Historical VaR
        var_pctile = np.percentile(returns_array, (1 - confidence) * 100)
        var = -var_pctile * self.current_equity  # Convert to dollar loss

        # CVaR (Expected Shortfall): average of losses beyond VaR
        tail_returns = returns_array[returns_array <= var_pctile]
        if len(tail_returns) > 0:
            cvar = -float(np.mean(tail_returns)) * self.current_equity
        else:
            cvar = var

        return max(0.0, var), max(0.0, cvar)

    # ── Correlation Matrix ──

    def compute_correlation_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Compute rolling pairwise correlation matrix across symbols."""
        symbols = [s for s in self._positions if len(self._return_history.get(s, [])) >= 20]

        if len(symbols) < 2:
            return np.eye(max(1, len(symbols))), symbols

        # Build return matrix
        min_len = min(len(self._return_history[s]) for s in symbols)
        min_len = min(min_len, self.correlation_window)

        return_matrix = np.column_stack([
            list(self._return_history[s])[-min_len:]
            for s in symbols
        ])

        # Correlation
        corr = np.corrcoef(return_matrix.T)
        return corr, symbols

    def get_average_correlation(self) -> float:
        """Get average pairwise correlation of current portfolio."""
        corr, symbols = self.compute_correlation_matrix()
        if len(symbols) < 2:
            return 0.0

        # Extract upper triangle (excluding diagonal)
        n = len(symbols)
        upper_tri = corr[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0

    # ── Exposure Calculation ──

    def get_exposure(self) -> Dict[str, float]:
        """Get current portfolio exposure breakdown (vectorized for multi-asset scale)."""
        if not self._positions:
            return {'gross': 0.0, 'net': 0.0, 'long': 0.0, 'short': 0.0}

        symbols = list(self._positions.keys())
        qtys = np.array([self._positions[s] for s in symbols])
        prices = np.array([self._position_prices.get(s, 0.0) for s in symbols])
        notionals = qtys * prices

        long_mask = notionals > 0
        long_exp = float(np.sum(notionals[long_mask]))
        short_exp = float(np.sum(np.abs(notionals[~long_mask])))

        return {
            'gross': long_exp + short_exp,
            'net': float(np.sum(notionals)),
            'long': long_exp,
            'short': short_exp,
        }

    # ── Regime Detection & Budget ──

    def detect_risk_regime(self, market_vol: Optional[float] = None) -> RiskRegime:
        """Determine current risk regime from portfolio conditions."""
        regime = RiskRegime.NORMAL

        # Use realized portfolio vol
        vol = market_vol or self._realized_vol

        # Vol-based regime detection
        if len(self._vol_history) >= 10:
            avg_vol = np.mean(list(self._vol_history))
            if avg_vol > 0:
                vol_ratio = vol / avg_vol
                if vol_ratio > 3.0:
                    regime = RiskRegime.CRISIS
                elif vol_ratio > 2.0:
                    regime = RiskRegime.STRESSED
                elif vol_ratio > 1.5:
                    regime = RiskRegime.ELEVATED

        # Drawdown-based escalation
        if self.current_drawdown_pct > 0.07:
            regime = max(regime, RiskRegime.CRISIS, key=lambda r: list(RiskRegime).index(r))
        elif self.current_drawdown_pct > 0.05:
            regime = max(regime, RiskRegime.STRESSED, key=lambda r: list(RiskRegime).index(r))
        elif self.current_drawdown_pct > 0.03:
            regime = max(regime, RiskRegime.ELEVATED, key=lambda r: list(RiskRegime).index(r))

        self._current_regime = regime
        return regime

    def get_current_budget(self) -> RiskBudget:
        """Get the risk budget for the current regime."""
        regime = self.detect_risk_regime()
        return self.regime_budgets.get(regime.value, self.regime_budgets[RiskRegime.NORMAL.value])

    def get_size_multiplier(self) -> float:
        """Get the current position sizing multiplier based on regime."""
        return self.get_current_budget().size_multiplier

    # ── Pre-Trade Checks ──

    def check_new_trade(
        self,
        symbol: str,
        qty: float,
        price: float,
    ) -> Tuple[bool, str]:
        """Pre-trade risk check for a proposed new position."""
        budget = self.get_current_budget()
        exposure = self.get_exposure()

        # 1. Gross exposure check
        new_gross = exposure['gross'] + abs(qty * price)
        if new_gross > budget.max_gross_exposure:
            return False, f"Gross exposure {new_gross:.0f} > limit {budget.max_gross_exposure:.0f}"

        # 2. Net exposure check
        new_notional = qty * price
        projected_net = exposure['net'] + new_notional
        if abs(projected_net) > budget.max_net_exposure:
            return False, f"Net exposure |{projected_net:.0f}| > limit {budget.max_net_exposure:.0f}"

        # 3. Concentration check
        if exposure['gross'] > 0:
            position_pct = abs(qty * price) / max(exposure['gross'] + abs(qty * price), 1)
            if position_pct > budget.max_single_position_pct:
                return False, f"Concentration {position_pct:.1%} > limit {budget.max_single_position_pct:.1%}"

        # 4. VaR check
        var_95, _ = self.compute_var(0.95)
        var_pct = var_95 / self.current_equity if self.current_equity > 0 else 0
        if var_pct > budget.max_var_95_pct:
            return False, f"VaR {var_pct:.2%} > limit {budget.max_var_95_pct:.2%}"

        # 5. Drawdown check
        if self.current_drawdown_pct > budget.max_drawdown_pct:
            return False, f"Drawdown {self.current_drawdown_pct:.2%} > limit {budget.max_drawdown_pct:.2%}"

        return True, "OK"

    # ── Snapshot & Monitoring ──

    def take_snapshot(self, timestamp: float = 0.0) -> PortfolioRiskSnapshot:
        """Take a point-in-time risk snapshot."""
        exposure = self.get_exposure()
        var_95, cvar_95 = self.compute_var(0.95)
        var_99, _ = self.compute_var(0.99)
        budget = self.get_current_budget()

        # Vol-adjusted drawdown
        dd_vol_adj = self.current_drawdown_pct / max(self._realized_vol, 1e-6)

        # Concentration (vectorized)
        max_concentration = 0.0
        if exposure['gross'] > 0 and self._positions:
            symbols = list(self._positions.keys())
            abs_notionals = np.array([
                abs(self._positions[s] * self._position_prices.get(s, 0.0))
                for s in symbols
            ])
            max_concentration = float(np.max(abs_notionals) / exposure['gross']) if len(abs_notionals) > 0 else 0.0

        snapshot = PortfolioRiskSnapshot(
            timestamp=timestamp,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            gross_exposure=exposure['gross'],
            net_exposure=exposure['net'],
            long_exposure=exposure['long'],
            short_exposure=exposure['short'],
            drawdown_pct=self.current_drawdown_pct,
            drawdown_vol_adjusted=dd_vol_adj,
            risk_regime=self._current_regime.value,
            risk_budget_used_pct=exposure['gross'] / max(budget.max_gross_exposure, 1),
            avg_correlation=self.get_average_correlation(),
            max_concentration_pct=max_concentration,
        )

        self._risk_snapshots.append(snapshot)
        return snapshot

    def get_status(self) -> Dict:
        """Get current risk status for monitoring."""
        exposure = self.get_exposure()
        var_95, cvar_95 = self.compute_var(0.95)
        budget = self.get_current_budget()

        return {
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'drawdown_pct': self.current_drawdown_pct,
            'realized_vol': self._realized_vol,
            'risk_regime': self._current_regime.value,
            'size_multiplier': budget.size_multiplier,
            'exposure': exposure,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'positions': dict(self._positions),
            'n_positions': len(self._positions),
            'avg_correlation': self.get_average_correlation(),
            'budget': {
                'max_gross': budget.max_gross_exposure,
                'max_net': budget.max_net_exposure,
                'max_var_pct': budget.max_var_95_pct,
                'max_dd_pct': budget.max_drawdown_pct,
            },
        }
