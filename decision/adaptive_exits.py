import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from regime.state_vector import StateVector
from regime.transition import TransitionEvent
from core.types import TradeAction
from config.loader import get_config
from microstructure.surface_analytics import SurfaceState
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitParameters:
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_hold_windows: int
    min_rr_ratio: float
    atr_value: float = 0.0


@dataclass
class TrailingState:
    peak_price: float = 0.0
    trough_price: float = 1e9
    best_unrealized_pct: float = 0.0
    # progressif: resserre le trailing à mesure que le profit croît
    trail_tightening_factor: float = 1.0


class ATRTracker:

    def __init__(self, period: int = 14, micro_window: int = 10):
        self.period = period
        self.micro_window = micro_window
        self._prices: List[float] = []
        self._true_ranges: List[float] = []
        self._atr: float = 0.0
        self._prev_close: float = 0.0

    def update(self, price: float) -> None:
        self._prices.append(price)

        # une barre = micro_window ticks — évite le bruit tick-à-tick
        if len(self._prices) >= self.micro_window:
            window = self._prices[-self.micro_window:]
            high = max(window)
            low = min(window)
            close = window[-1]

            if self._prev_close > 0:
                tr = max(
                    high - low,
                    abs(high - self._prev_close),
                    abs(low - self._prev_close),
                )
            else:
                tr = high - low

            self._true_ranges.append(tr)
            self._prev_close = close

            if len(self._true_ranges) > self.period * 3:
                self._true_ranges = self._true_ranges[-(self.period * 2):]

            # ATR = EMA pondérée exponentiellement (biais sur le récent)
            if len(self._true_ranges) >= self.period:
                weights = np.exp(np.linspace(-1, 0, min(len(self._true_ranges), self.period * 2)))
                recent = self._true_ranges[-len(weights):]
                weights = weights[-len(recent):]
                weights /= weights.sum()
                self._atr = float(np.dot(recent, weights))
            elif len(self._true_ranges) > 0:
                self._atr = float(np.mean(self._true_ranges))

            if len(self._prices) > self.micro_window * 5:
                self._prices = self._prices[-(self.micro_window * 3):]

    @property
    def atr(self) -> float:
        return self._atr

    def atr_pct(self, reference_price: float) -> float:
        if reference_price <= 0:
            return 0.0
        return self._atr / reference_price

    @property
    def is_ready(self) -> bool:
        return len(self._true_ranges) >= self.period


class AdaptiveExitEngine:

    def __init__(self, config=None):
        self.config = config or get_config()
        self.exit_cfg = self.config.thresholds.decision.exit
        self._regime_scales = self.config.thresholds.decision.exit_regime_scales
        self.trailing = TrailingState()
        self._recent_sigmas = []
        self._max_sigma_history = 50

        self.atr_tracker = ATRTracker(
            period=getattr(self.exit_cfg, 'atr_period', 14),
            micro_window=getattr(self.exit_cfg, 'atr_micro_window', 10),
        )

    def update_price(self, price: float) -> None:
        self.atr_tracker.update(price)

    def compute_exit_params(
        self,
        state: StateVector,
        regime_stats: Optional[dict] = None,
        current_drawdown_pct: float = 0.0,
        transition_strength: float = 0.0,
        regime_win_rate: float = 0.5,
        reference_price: float = 0.0,
        autocorrelation: float = 0.0,
        transition_volatility: float = 0.0,
    ) -> ExitParameters:
        # guard NaN/Inf sur le vecteur d'état
        _sigma = state.sigma if np.isfinite(state.sigma) else 1e-6
        _kurtosis = state.kurtosis if np.isfinite(state.kurtosis) else 3.0
        _skew = state.skew if np.isfinite(state.skew) else 0.0
        _entropy = state.entropy if np.isfinite(state.entropy) else 1.0

        sigma = max(1e-9, abs(_sigma))
        self._recent_sigmas.append(sigma)
        if len(self._recent_sigmas) > self._max_sigma_history:
            self._recent_sigmas.pop(0)

        weights = np.exp(np.linspace(-1, 0, len(self._recent_sigmas)))
        weights /= weights.sum()
        ema_sigma = float(np.dot(self._recent_sigmas, weights))

        atr_pct = self.atr_tracker.atr_pct(reference_price) if reference_price > 0 else 0.0

        # ATR prioritaire — si sigma >> ATR, distribution fat-tail → élargir
        if self.atr_tracker.is_ready and atr_pct > 0:
            base_vol = atr_pct
            sigma_boost = max(0.0, (ema_sigma / base_vol - 1.0) * 0.3) if base_vol > 0 else 0.0
            base_vol *= (1.0 + sigma_boost)
        else:
            base_vol = ema_sigma  # fallback avant warmup ATR

        excess_kurt = max(0.0, _kurtosis - 3.0)
        tail_factor = 1.0 + excess_kurt * self.exit_cfg.kurtosis_stop_scale

        # skew absolu: distribution asymétrique → plus de room pour le TP
        skew_tp_factor = 1.0 + abs(_skew) * self.exit_cfg.skew_tp_scale

        regime_vol_factor = 1.0
        if regime_stats and 'std' in regime_stats:
            std_sv = regime_stats['std']
            regime_sigma_std = getattr(std_sv, 'sigma', 0.0)
            if base_vol > 0:
                regime_vol_factor = 1.0 + min(2.0, regime_sigma_std / base_vol)

        # amortisseur DD: réduit de 30x → 5x (l'ancien coupait les winners à 1% DD)
        dd_factor = 1.0
        if current_drawdown_pct > 0.01:
            dd_factor = max(0.5, 1.0 - current_drawdown_pct * 5)

        strength_bonus = 1.0 + transition_strength * self.exit_cfg.strength_tp_bonus

        # scaling par type de régime (autocorr-based, config-driven)
        _rs = self._regime_scales
        _ac_thresh = self.config.thresholds.decision.entry_gate.autocorr_trending_threshold
        is_trending = autocorrelation > _ac_thresh
        if is_trending:
            regime_type_stop_scale = _rs.trending_stop_scale
            regime_type_tp_scale = _rs.trending_tp_scale
            regime_type_trail_scale = _rs.trending_trail_scale
        elif autocorrelation < -_ac_thresh:
            regime_type_stop_scale = _rs.reverting_stop_scale
            regime_type_tp_scale = _rs.reverting_tp_scale
            regime_type_trail_scale = _rs.reverting_trail_scale
        else:
            regime_type_stop_scale = 1.0
            regime_type_tp_scale = 1.0
            regime_type_trail_scale = 1.0

        # instabilité de transition → resserre les stops
        if transition_volatility > _rs.transition_vol_threshold:
            instability_scale = max(0.6, 1.0 - transition_volatility * _rs.transition_vol_scale)
            regime_type_trail_scale *= instability_scale
            regime_type_stop_scale *= max(0.7, instability_scale)

        tail_factor_capped = min(1.2, tail_factor)
        regime_vol_capped = min(1.2, regime_vol_factor)
        raw_stop = base_vol * self.exit_cfg.stop_sigma_mult * tail_factor_capped * regime_vol_capped * 0.8 * regime_type_stop_scale
        stop_loss = max(self.exit_cfg.min_stop_pct, raw_stop * dd_factor)

        raw_tp = base_vol * self.exit_cfg.tp_sigma_mult * max(0.8, skew_tp_factor) * strength_bonus * regime_type_tp_scale
        take_profit = max(self.exit_cfg.min_tp_pct, raw_tp)

        trail_factor = 1.0 + excess_kurt * self.exit_cfg.kurtosis_trail_scale
        raw_trail = base_vol * self.exit_cfg.trail_sigma_mult * trail_factor * regime_type_trail_scale
        trailing = max(self.exit_cfg.min_stop_pct, raw_trail * dd_factor)

        # entropy élevée = régime incertain → tenir PLUS longtemps (attendre résolution)
        # logique inversée par rapport à l'ancienne implémentation
        entropy_norm = 1.0 / (1.0 + abs(_entropy))
        entropy_bonus = int(entropy_norm * self.exit_cfg.entropy_hold_scale * 0.5)
        max_hold = max(5, self.exit_cfg.max_hold_base + entropy_bonus)
        max_hold = min(max_hold, self.exit_cfg.max_hold_base * 2)

        if regime_stats and 'centroid' in regime_stats:
            centroid = regime_stats['centroid']
            if hasattr(centroid, 'entropy') and centroid.entropy > state.entropy * 1.5:
                max_hold = max(5, int(max_hold * 0.6))

        # R:R dynamique selon WR du régime
        # WR faible = losses fréquents → R:R serré pour limiter l'impact
        if regime_win_rate >= 0.6:
            base_rr = 1.0
        elif regime_win_rate >= 0.45:
            base_rr = 1.2
        else:
            base_rr = 1.5

        rr_ratio = base_rr * max(0.8, 1.0 - transition_strength * 0.2)

        if current_drawdown_pct > 0.005:
            rr_ratio *= 1.0 + current_drawdown_pct * 5

        # plancher: MR haute WR profitable même à R:R ~0.5 (Kelly positif à WR≥0.70)
        rr_ratio = max(0.5, min(rr_ratio, 3.0))

        min_tp_from_rr = stop_loss * rr_ratio
        if take_profit < min_tp_from_rr:
            take_profit = min_tp_from_rr

        logger.debug(
            f"EXIT PARAMS: ATR={atr_pct:.8f}, σ={ema_sigma:.8f}, base_vol={base_vol:.8f}, "
            f"SL={stop_loss:.6%}, TP={take_profit:.6%}, Trail={trailing:.6%}, "
            f"Hold={max_hold}, R:R={rr_ratio:.2f}, "
            f"tail_f={tail_factor:.2f}, skew_f={skew_tp_factor:.2f}, "
            f"dd_f={dd_factor:.2f}, str_bonus={strength_bonus:.2f}, "
            f"regime_wr={regime_win_rate:.2%}"
        )

        return ExitParameters(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            trailing_stop_pct=trailing,
            max_hold_windows=max_hold,
            min_rr_ratio=rr_ratio,
            atr_value=self.atr_tracker.atr,
        )

    def check_entry_rr(
        self,
        entry_price: float,
        exit_params: ExitParameters,
        side: TradeAction,
    ) -> Tuple[bool, str]:
        risk = entry_price * exit_params.stop_loss_pct
        reward = entry_price * exit_params.take_profit_pct

        if risk <= 0:
            return False, "Zero risk — invalid stop-loss"

        rr = reward / risk
        if rr < exit_params.min_rr_ratio:
            return False, (
                f"R:R {rr:.2f} < minimum {exit_params.min_rr_ratio:.1f} "
                f"(risk=${risk:.4f}, reward=${reward:.4f})"
            )

        return True, (
            f"R:R {rr:.2f} >= {exit_params.min_rr_ratio:.1f}, "
            f"Size={exit_params.stop_loss_pct:.6%}/{exit_params.take_profit_pct:.6%}, "
            f"ATR={exit_params.atr_value:.4f}"
        )

    def reset_trailing(self, entry_price: float, side: TradeAction) -> None:
        self.trailing = TrailingState(
            peak_price=entry_price if side == TradeAction.BUY else 0.0,
            trough_price=entry_price if side == TradeAction.SELL else 1e9,
            best_unrealized_pct=0.0,
            trail_tightening_factor=1.0,
        )

    def check_exit(
        self,
        entry_price: float,
        current_price: float,
        position_side: TradeAction,
        position_windows: int,
        exit_params: ExitParameters,
        transition: Optional[TransitionEvent] = None,
        state: Optional[StateVector] = None,
        surface_state: Optional[SurfaceState] = None,
    ) -> Tuple[bool, str]:
        if entry_price <= 0 or current_price <= 0:
            return False, ""

        if position_side == TradeAction.BUY:
            unrealized_pct = (current_price - entry_price) / entry_price
            if current_price > self.trailing.peak_price:
                self.trailing.peak_price = current_price
            self.trailing.best_unrealized_pct = max(
                self.trailing.best_unrealized_pct,
                unrealized_pct
            )
            trail_drop = (self.trailing.peak_price - current_price) / self.trailing.peak_price
        else:
            unrealized_pct = (entry_price - current_price) / entry_price
            if current_price < self.trailing.trough_price:
                self.trailing.trough_price = current_price
            self.trailing.best_unrealized_pct = max(
                self.trailing.best_unrealized_pct,
                unrealized_pct
            )
            trail_drop = (current_price - self.trailing.trough_price) / self.trailing.trough_price if self.trailing.trough_price > 0 else 0.0

        # 0. SURFACE COLLAPSE — sortie préemptive sur collapse de courbure topologique
        if surface_state and surface_state.is_surface_collapsing:
            return True, (
                f"Information Edge Decay (Surface Collapse): "
                f"trajectory={surface_state.regime_trajectory_z:.4f}, "
                f"curvature={surface_state.surface_curvature:.4f}"
            )

        # 1. STOP-LOSS
        if unrealized_pct <= -exit_params.stop_loss_pct:
            return True, (
                f"Stop-Loss: {unrealized_pct:.4%} <= -{exit_params.stop_loss_pct:.4%}"
            )

        # 2. TRAILING STOP — activation différée pour laisser le trade respirer
        # les trades MR retracent avant de payer — activation trop tôt = sortie prématurée
        if self.trailing.best_unrealized_pct > exit_params.take_profit_pct * self._regime_scales.trailing_activation_pct:
            profit_ratio = self.trailing.best_unrealized_pct / exit_params.take_profit_pct
            if profit_ratio >= 1.5:
                tighten = self._regime_scales.trailing_tight_factor
            else:
                tighten = self._regime_scales.trailing_normal_factor

            effective_trail = exit_params.trailing_stop_pct * tighten

            if trail_drop >= effective_trail:
                return True, (
                    f"Trailing Stop: retraced {trail_drop:.4%} from "
                    f"best {self.trailing.best_unrealized_pct:.4%} "
                    f"(trail={effective_trail:.4%}, tighten={tighten:.1f}x)"
                )

        # 3. TAKE-PROFIT
        if unrealized_pct >= exit_params.take_profit_pct:
            return True, (
                f"Take-Profit: {unrealized_pct:.4%} >= {exit_params.take_profit_pct:.4%}"
            )

        # 4. MAX HOLD
        if position_windows >= exit_params.max_hold_windows:
            return True, (
                f"Max Hold: {position_windows} >= {exit_params.max_hold_windows} windows"
            )

        # 5. INFORMATION EDGE DECAY — désactivé pour MR (TODO: réactiver pour trend)
        if False:
            pass

        return False, ""
