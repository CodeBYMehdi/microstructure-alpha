import numpy as np
from regime.transition import TransitionEvent
from typing import Optional, Dict, Any, List, Tuple
from config.loader import get_config

import logging

logger = logging.getLogger(__name__)

# max 5% des 3 premiers niveaux du carnet — évite le market impact
_MAX_BOOK_CONSUMPTION_PCT = 0.05
_TOP_N_LEVELS = 3


def compute_l2_capacity(
    book_levels: List[Tuple[float, float]],
    n_levels: int = _TOP_N_LEVELS,
    max_pct: float = _MAX_BOOK_CONSUMPTION_PCT,
) -> float:
    if not book_levels:
        return 0.0
    top = book_levels[:n_levels]
    total_depth = sum(size for _, size in top)
    if total_depth <= 0:
        return 0.0
    return total_depth * max_pct


def _kelly_fraction(win_rate: float, profit_factor: float) -> float:
    """f* = p - q/b — Kelly brut, le caller applique le facteur fractionnaire"""
    if not np.isfinite(profit_factor) or profit_factor <= 0:
        profit_factor = 1.0
    if not np.isfinite(win_rate):
        win_rate = 0.0
    f = win_rate - (1.0 - win_rate) / profit_factor
    return max(0.0, min(1.0, f))


class PositionSizer:
    """Kelly ½ + conviction + cap L2 + amortisseur drawdown"""

    def __init__(self, base_size: float = None, config=None):
        self.config = config or get_config()
        sizing_cfg = self.config.thresholds.decision.sizing
        self.base_size = base_size if base_size is not None else sizing_cfg.base_size
        self.max_multiplier = sizing_cfg.max_size_multiplier
        self.min_variance = sizing_cfg.min_variance
        self.dwell_norm = sizing_cfg.dwell_time_norm
        self.kelly_scale = getattr(sizing_cfg, 'kelly_scale', 0.5)  # ½-Kelly = standard industrie

    def calculate(
        self,
        transition: TransitionEvent,
        tail_slope: float,
        regime_stats: Optional[Dict[str, Any]] = None,
        win_rate: float = 0.5,
        profit_factor: float = 1.0,
        kelly_fraction: float = 0.5,
        l2_levels: Optional[List[Tuple[float, float]]] = None,
        equity: float = 0.0,
        price: float = 0.0,
        stop_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        var_95: float = 0.0,
    ) -> float:
        # VaR-adjusted stop floor: stop must cover at least 1.5× historical VaR
        # This prevents stops from being tighter than the observed tail distribution
        if var_95 < 0 and price > 0:
            var_stop_floor = abs(var_95) * 1.5
            if var_stop_floor > stop_pct:
                logger.debug(f"VaR stop floor: {stop_pct:.4%} -> {var_stop_floor:.4%}")
                stop_pct = var_stop_floor

        use_risk_based = (equity > 0 and price > 0 and stop_pct > 1e-9)

        raw_kelly = _kelly_fraction(win_rate, profit_factor)
        risk_fraction = raw_kelly * self.kelly_scale

        conviction = self._compute_conviction(
            transition=transition,
            tail_slope=tail_slope,
            regime_stats=regime_stats,
            drawdown_pct=drawdown_pct,
        )

        if use_risk_based:
            # shares = (equity × kelly_risk) / (price × stop)
            # floor minimal si kelly≈0 — permet de collecter des données
            effective_risk = max(risk_fraction, 0.005) if raw_kelly < 0.01 else risk_fraction
            dollar_risk = equity * effective_risk
            risk_per_share = price * stop_pct
            size = (dollar_risk / risk_per_share) * conviction

            # cap drawdown: perte max sur ce trade ≤ max_dd / pénalité queue
            max_dd = self.config.thresholds.risk.max_drawdown
            tail_penalty = 1.0
            if tail_slope > 0 and np.isfinite(tail_slope):
                # queues grasses → pénalité → taille réduite
                tail_penalty = max(self.config.thresholds.decision.sizing.tail_penalty_floor, min(2.0, tail_slope / 2.0))
            max_loss_per_trade = equity * max_dd / tail_penalty
            dd_cap_shares = max_loss_per_trade / (price * stop_pct)
            if size > dd_cap_shares:
                size = dd_cap_shares

            # pas de levier — notionnel ≤ equity
            max_unleveraged = equity / price
            if size > max_unleveraged:
                size = max_unleveraged

            logger.info(
                f"SIZING: equity=${equity:.0f} kelly={raw_kelly:.3f} "
                f"risk_frac={effective_risk:.3%} stop={stop_pct:.4%} "
                f"conviction={conviction:.2f} -> {size:.1f} shares "
                f"(${size * price:.0f} notional, "
                f"max_loss=${size * price * stop_pct:.0f})"
            )
        else:
            # fallback si equity/price/stop absent (tests unitaires, etc.)
            size = self._legacy_calculate(
                transition, tail_slope, regime_stats,
                win_rate, profit_factor, kelly_fraction,
            )
            size *= conviction

        size = max(size, 0.0)

        if l2_levels:
            l2_cap = compute_l2_capacity(l2_levels)
            if l2_cap > 0 and size > l2_cap:
                logger.info(f"L2 liquidity cap: {size:.1f} -> {l2_cap:.1f}")
                size = l2_cap

        return size

    def _compute_conviction(
        self,
        transition: TransitionEvent,
        tail_slope: float,
        regime_stats: Optional[Dict[str, Any]],
        drawdown_pct: float,
    ) -> float:
        """scalaire conviction ∈ [0.25, 1.5] — force × persistance × DD × queues"""
        strength = transition.strength if np.isfinite(transition.strength) else 0.0
        strength = max(0.0, min(1.0, strength))

        # persistance: dwell time log-scalé → [0,1]
        persistence = 0.5
        if regime_stats and 'dwell_time' in regime_stats:
            dwell = regime_stats.get('dwell_time', 100.0)
            if np.isfinite(dwell) and dwell > 0:
                persistence = min(1.0, np.log1p(dwell / self.dwell_norm) / 3.0)

        # amortisseur DD: linéaire de 1.0 à 15% du max_dd jusqu'à 0.3 au max_dd
        dd_scale = 1.0
        max_dd = self.config.thresholds.risk.max_drawdown
        if drawdown_pct > max_dd * 0.15:
            dd_ratio = min(1.0, drawdown_pct / max_dd)
            dd_scale = max(0.3, 1.0 - dd_ratio * 0.7)

        tail_scale = 1.0
        if tail_slope > 0 and np.isfinite(tail_slope):
            tail_scale = max(0.5, min(1.0, 2.0 / max(tail_slope, 0.5)))

        # blend 50/50, puis normalisation: neutre (0.5 chacun) → 1.0
        raw = strength * 0.5 + persistence * 0.5
        conviction = raw * 2.0 * dd_scale * tail_scale

        return max(0.25, min(1.5, conviction))

    def _legacy_calculate(
        self,
        transition: TransitionEvent,
        tail_slope: float,
        regime_stats: Optional[Dict[str, Any]],
        win_rate: float,
        profit_factor: float,
        kelly_fraction: float,
    ) -> float:
        """sizing legacy sans equity — utilisé en tests et backtest simple"""
        expected_pnl = transition.strength if np.isfinite(transition.strength) else 0.0
        var_pnl = max(self.min_variance, tail_slope / 2.0)

        raw_kelly = _kelly_fraction(win_rate, profit_factor)
        kelly_mult = max(0.1, min(2.0, raw_kelly * kelly_fraction * 10))
        dynamic_base = self.base_size * kelly_mult

        tau = 1.0
        if regime_stats and 'dwell_time' in regime_stats:
            avg_duration = regime_stats.get('dwell_time', 100.0)
            if not np.isfinite(avg_duration) or avg_duration <= 0:
                avg_duration = 100.0
            tau = np.log1p(avg_duration / self.dwell_norm)
        else:
            tau = 0.5 + (transition.strength * 0.5)

        raw_ratio = expected_pnl / max(1e-6, var_pnl)
        size = dynamic_base * raw_ratio * tau

        size = min(size, self.base_size * self.max_multiplier)
        size = max(size, 0.0)
        return size
