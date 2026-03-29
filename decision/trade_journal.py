# le cerveau de l'operation
# la calculette

import json
import os
import csv
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeJournalEntry:
    # l'usine a gaz
    # Identification
    trade_id: int = 0
    timestamp_entry: float = 0.0
    timestamp_exit: float = 0.0
    
    # Trade details
    symbol: str = ""
    side: str = ""             # BUY / SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    
    # PnL
    gross_pnl: float = 0.0    # Before fees
    net_pnl: float = 0.0      # After fees
    fees: float = 0.0
    
    # Context at entry
    regime_id: int = -1
    regime_confidence: float = 0.0
    entry_reason: str = ""
    transition_strength: float = 0.0
    volatility_at_entry: float = 0.0
    
    # Context at exit
    exit_reason: str = ""
    hold_duration_windows: int = 0
    best_unrealized_pct: float = 0.0
    worst_unrealized_pct: float = 0.0
    
    # MAE/MFE (Maximum Adverse/Favorable Excursion)
    mae_pct: float = 0.0      # Maximum Adverse Excursion as % of entry price
    mfe_pct: float = 0.0      # Maximum Favorable Excursion as % of entry price
    mae_price: float = 0.0    # Worst price reached during trade
    mfe_price: float = 0.0    # Best price reached during trade
    
    # Exit parameters used
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    
    # Quality metrics
    rr_actual: float = 0.0     # Actual reward:risk achieved
    edge_quality: float = 0.0  # Was the direction correct at entry?
    edge_captured_pct: float = 0.0  # net_pnl / mfe — how much of the edge was captured
    
    def to_dict(self) -> dict:
        return asdict(self)


class PerformanceTracker:
    # le cerveau de l'operation
    # la calculette

    def __init__(self, lookback: int = 50, journal_path: str = "trade_journal.csv"):
        self.trades: List[TradeJournalEntry] = []
        self.lookback = lookback
        self.journal_path = journal_path
        self._trade_counter = 0
        
        # Rolling metrics
        self._regime_performance: Dict[int, List[float]] = defaultdict(list)
        self._setup_performance: Dict[str, List[float]] = defaultdict(list)
        self._rolling_win_rate = 0.0
        self._rolling_profit_factor = 0.0
        self._rolling_expectancy = 0.0
        self._consecutive_losses = 0
        self._max_consecutive_losses = 0
        
        # Multi-window rolling metrics (20/50/100-trade windows)
        self._window_metrics: Dict[int, Dict[str, float]] = {
            20: {},
            50: {},
            100: {},
        }
        self._degradation_detected = False
        
        # MAE/MFE aggregates
        self._mae_history: List[float] = []
        self._mfe_history: List[float] = []
        
        # Adaptive state
        self._regime_win_rates: Dict[int, float] = {}
        self._regime_avg_pnl: Dict[int, float] = {}
        self._regime_trade_counts: Dict[int, int] = {}
        self._sizing_multiplier = 1.0  # Reduces during drawdown
        self._should_pause = False     # True if recent performance is disastrous
        
        # Initialize journal file
        self._init_journal()
    
    def _init_journal(self):
        # on prepare le terrain
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trade_id', 'timestamp_entry', 'timestamp_exit',
                    'symbol', 'side', 'entry_price', 'exit_price', 'quantity',
                    'gross_pnl', 'net_pnl', 'fees',
                    'regime_id', 'regime_confidence', 'entry_reason',
                    'transition_strength', 'volatility_at_entry',
                    'exit_reason', 'hold_duration_windows',
                    'best_unrealized_pct', 'worst_unrealized_pct',
                    'mae_pct', 'mfe_pct', 'mae_price', 'mfe_price',
                    'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct',
                    'rr_actual', 'edge_quality', 'edge_captured_pct',
                ])
    
    def record_trade(self, entry: TradeJournalEntry) -> None:
        # l'usine a gaz
        self._trade_counter += 1
        entry.trade_id = self._trade_counter
        
        # Compute edge_captured_pct
        if entry.mfe_pct > 0:
            realized_pct = entry.net_pnl / (entry.entry_price * entry.quantity) if entry.entry_price > 0 and entry.quantity > 0 else 0.0
            entry.edge_captured_pct = realized_pct / entry.mfe_pct if entry.mfe_pct > 0 else 0.0
        
        self.trades.append(entry)
        
        # Track MAE/MFE
        self._mae_history.append(entry.mae_pct)
        self._mfe_history.append(entry.mfe_pct)
        
        # Update regime-level metrics
        self._regime_performance[entry.regime_id].append(entry.net_pnl)
        self._setup_performance[entry.entry_reason[:30]].append(entry.net_pnl)
        
        # Update consecutive loss tracking
        if entry.net_pnl < 0:
            self._consecutive_losses += 1
            self._max_consecutive_losses = max(
                self._max_consecutive_losses, self._consecutive_losses
            )
        else:
            self._consecutive_losses = 0
        
        # Recompute rolling metrics
        self._update_rolling_metrics()
        
        # Write to journal
        self._write_entry(entry)
        
        # Log self-improvement insights
        self._log_insights(entry)
    
    def _update_rolling_metrics(self) -> None:
        # l'usine a gaz
        recent = self.trades[-self.lookback:]
        if not recent:
            return
        
        pnls = [t.net_pnl for t in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        self._rolling_win_rate = len(wins) / len(pnls) if pnls else 0.0
        
        total_wins = sum(wins) if wins else 0.0
        total_losses = abs(sum(losses)) if losses else 1e-10
        self._rolling_profit_factor = total_wins / total_losses
        
        self._rolling_expectancy = np.mean(pnls) if pnls else 0.0
        
        # Update regime-level win rates
        for regime_id, pnl_list in self._regime_performance.items():
            recent_pnls = pnl_list[-20:]  # Last 20 trades in this regime
            if recent_pnls:
                self._regime_win_rates[regime_id] = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
                self._regime_avg_pnl[regime_id] = np.mean(recent_pnls)
                self._regime_trade_counts[regime_id] = len(pnl_list)
        
        # ── Multi-window rolling metrics (20/50/100) ──
        for window_size in [20, 50, 100]:
            if len(self.trades) >= window_size:
                window_trades = self.trades[-window_size:]
                w_pnls = [t.net_pnl for t in window_trades]
                w_wins = [p for p in w_pnls if p > 0]
                w_losses = [p for p in w_pnls if p <= 0]
                
                w_wr = len(w_wins) / len(w_pnls)
                w_total_wins = sum(w_wins) if w_wins else 0.0
                w_total_losses = abs(sum(w_losses)) if w_losses else 1e-10
                w_pf = w_total_wins / w_total_losses
                w_expect = float(np.mean(w_pnls))
                
                # MAE/MFE stats for window
                w_maes = [t.mae_pct for t in window_trades if t.mae_pct != 0]
                w_mfes = [t.mfe_pct for t in window_trades if t.mfe_pct != 0]
                w_avg_mae = float(np.mean(w_maes)) if w_maes else 0.0
                w_avg_mfe = float(np.mean(w_mfes)) if w_mfes else 0.0
                
                # Edge capture: average ratio of realized PnL to MFE
                edge_captures = [t.edge_captured_pct for t in window_trades if t.mfe_pct > 0]
                w_edge_capture = float(np.mean(edge_captures)) if edge_captures else 0.0
                
                self._window_metrics[window_size] = {
                    'win_rate': w_wr,
                    'profit_factor': w_pf,
                    'expectancy': w_expect,
                    'total_pnl': sum(w_pnls),
                    'avg_mae': w_avg_mae,
                    'avg_mfe': w_avg_mfe,
                    'edge_capture': w_edge_capture,
                    'n_trades': window_size,
                }
        
        # ── Degradation Detection ──
        # Compare 20-trade window vs 100-trade window
        # If recent performance is significantly worse, flag it
        self._degradation_detected = False
        if 20 in self._window_metrics and 100 in self._window_metrics:
            w20 = self._window_metrics[20]
            w100 = self._window_metrics[100]
            if w20 and w100:
                # Degradation: 20-trade WR is >15pp below 100-trade WR
                wr_drop = w100.get('win_rate', 0.5) - w20.get('win_rate', 0.5)
                # Or 20-trade expectancy is negative while 100-trade is positive
                exp_flip = w100.get('expectancy', 0) > 0 and w20.get('expectancy', 0) < 0
                
                if wr_drop > 0.15 or exp_flip:
                    self._degradation_detected = True
                    logger.warning(
                        f"[DEGRADATION] Recent 20-trade performance degraded: "
                        f"WR_20={w20.get('win_rate', 0):.1%} vs WR_100={w100.get('win_rate', 0):.1%}, "
                        f"E[PnL]_20={w20.get('expectancy', 0):.4f} vs E[PnL]_100={w100.get('expectancy', 0):.4f}"
                    )
        
        # Adaptive sizing
        if self._consecutive_losses >= 3:
            self._sizing_multiplier = max(0.25, 1.0 - self._consecutive_losses * 0.15)
        elif self._rolling_win_rate > 0.3 and self._rolling_profit_factor > 0.5:
            self._sizing_multiplier = min(1.0, self._sizing_multiplier + 0.1)
        
        # Pause trading if recent performance is catastrophic
        self._should_pause = (
            len(recent) >= 10 and 
            self._rolling_win_rate < 0.1 and 
            self._rolling_profit_factor < 0.05
        )
        
        if self._should_pause:
            logger.warning(
                f"SELF-IMPROVEMENT: Trading PAUSED — "
                f"win_rate={self._rolling_win_rate:.1%}, "
                f"profit_factor={self._rolling_profit_factor:.3f}, "
                f"consec_losses={self._consecutive_losses}"
            )
    
    def _write_entry(self, entry: TradeJournalEntry) -> None:
        # l'usine a gaz
        try:
            with open(self.journal_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.trade_id, entry.timestamp_entry, entry.timestamp_exit,
                    entry.symbol, entry.side, entry.entry_price, entry.exit_price,
                    entry.quantity,
                    entry.gross_pnl, entry.net_pnl, entry.fees,
                    entry.regime_id, entry.regime_confidence, entry.entry_reason,
                    entry.transition_strength, entry.volatility_at_entry,
                    entry.exit_reason, entry.hold_duration_windows,
                    entry.best_unrealized_pct, entry.worst_unrealized_pct,
                    entry.mae_pct, entry.mfe_pct, entry.mae_price, entry.mfe_price,
                    entry.stop_loss_pct, entry.take_profit_pct, entry.trailing_stop_pct,
                    entry.rr_actual, entry.edge_quality, entry.edge_captured_pct,
                ])
        except Exception as e:
            logger.error(f"Failed to write journal entry: {e}")
    
    def _log_insights(self, entry: TradeJournalEntry) -> None:
        # l'usine a gaz
        n = len(self.trades)
        if n % 10 == 0:  # Every 10 trades
            logger.info(
                f"[SELF-IMPROVEMENT] After {n} trades: "
                f"WR={self._rolling_win_rate:.1%}, "
                f"PF={self._rolling_profit_factor:.3f}, "
                f"E[PnL]={self._rolling_expectancy:.4f}, "
                f"sizing_mult={self._sizing_multiplier:.2f}, "
                f"consec_L={self._consecutive_losses}"
            )
            
            # Log regime-level insights
            for rid, wr in sorted(self._regime_win_rates.items()):
                avg = self._regime_avg_pnl.get(rid, 0)
                logger.info(
                    f"  Regime {rid}: WR={wr:.1%}, avg_pnl={avg:.4f}"
                )
    
    # ---- Adaptive Parameter Getters ----
    
    def get_bayesian_kelly(self, regime_id: int) -> float:
        # la calculette
        pnl_list = self._regime_performance.get(regime_id, [])
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        
        # Uninformed Beta(1, 1) prior is 50/50, but we use Beta(2, 2) for slower update
        alpha_prior = 2.0 + len(wins)
        beta_prior  = 2.0 + len(losses)
        post_win_rate = alpha_prior / (alpha_prior + beta_prior)
        
        avg_win = sum(wins)/len(wins) if wins else 1.0
        avg_loss = abs(sum(losses))/len(losses) if losses else 1.0
        
        ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        if ratio <= 0: return 0.1 # Min fallback
        
        kelly = post_win_rate - ((1.0 - post_win_rate) / ratio)
        fractional_kelly = max(0.1, min(1.0, kelly))  # Half or full Kelly bounded
        return fractional_kelly

    def get_sizing_multiplier(self) -> float:
        # l'usine a gaz
        return self._sizing_multiplier
    
    def should_skip_regime(self, regime_id: int) -> Tuple[bool, str]:
        # ca fait bim
        pnl_list = self._regime_performance.get(regime_id, [])
        total_pnl = sum(pnl_list)

        # Immediate massive loss cutoff, regardless of trade count
        if total_pnl < -5.0:
            return True, f"Regime {regime_id} total PnL < -5.0 ({total_pnl:.2f}). Permanently excluded."
            
        if len(pnl_list) < 2:
            return False, f"Insufficient data ({len(pnl_list)}/2 trades)"
        
        # Stricter trailing cutoff — kick out faster
        if total_pnl < -2.5:
            return True, f"Regime {regime_id} total PnL < -2.5 ({total_pnl:.2f}). Excluded."
            
        # Use shorter 10-trade window for faster reaction
        recent = pnl_list[-10:]
        wr = sum(1 for p in recent if p > 0) / len(recent)
        avg = np.mean(recent)
        
        # If it's losing money and win rate is < 50%, cut it immediately
        if wr < 0.50 and avg < 0:
            return True, (
                f"Regime {regime_id} historically unprofitable: "
                f"WR={wr:.1%}, avg_pnl={avg:.4f} (last {len(recent)} trades)"
            )
        
        return False, ""
    
    def should_pause_trading(self) -> bool:
        # l'usine a gaz
        return self._should_pause
    
    @property
    def rolling_stats(self) -> dict:
        # l'usine a gaz
        stats = {
            'total_trades': len(self.trades),
            'rolling_win_rate': self._rolling_win_rate,
            'rolling_profit_factor': self._rolling_profit_factor,
            'rolling_expectancy': self._rolling_expectancy,
            'consecutive_losses': self._consecutive_losses,
            'max_consecutive_losses': self._max_consecutive_losses,
            'sizing_multiplier': self._sizing_multiplier,
            'regime_win_rates': dict(self._regime_win_rates),
            'degradation_detected': self._degradation_detected,
        }
        
        # Add multi-window metrics
        for window_size in [20, 50, 100]:
            w = self._window_metrics.get(window_size, {})
            if w:
                stats[f'window_{window_size}'] = w
        
        # Add MAE/MFE aggregates
        if self._mae_history:
            stats['avg_mae'] = float(np.mean(self._mae_history))
            stats['median_mae'] = float(np.median(self._mae_history))
            stats['p90_mae'] = float(np.percentile(self._mae_history, 90))
        if self._mfe_history:
            stats['avg_mfe'] = float(np.mean(self._mfe_history))
            stats['median_mfe'] = float(np.median(self._mfe_history))
            stats['p10_mfe'] = float(np.percentile(self._mfe_history, 10))
            stats['mfe_to_mae_ratio'] = (
                float(np.mean(self._mfe_history)) / max(abs(float(np.mean(self._mae_history))), 1e-10)
                if self._mae_history else 0.0
            )
        
        return stats

    @property
    def is_degrading(self) -> bool:
        # le cerveau de l'operation
        return self._degradation_detected
