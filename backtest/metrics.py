import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
from core.types import TradeAction
from scipy.stats import skew
from statistics.deflated_sharpe import deflated_sharpe

@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    regime_id: str = "UNDEFINED"
    commission: float = 0.0
    holding_time: float = 0.0  # seconds
    entry_entropy: float = 0.0
    mae_pct: float = 0.0      # Maximum Adverse Excursion as % of entry
    mfe_pct: float = 0.0      # Maximum Favorable Excursion as % of entry

class PerformanceMetrics:
    # l'usine a gaz
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = [{"timestamp": 0, "equity": initial_equity}]
        
        # Analytics Régime
        self.regime_stats = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "dwell_time": 0.0})
        self.last_regime_update = 0
        self.current_regime = "UNDEFINED"
        self.transitions = 0
        
    def record_trade(self, trade: TradeRecord):
        self.trades.append(trade)
        self.equity += trade.pnl
        self.equity_curve.append({
            "timestamp": trade.timestamp, 
            "equity": self.equity
        })
        
        # MAJ stats régime
        self.regime_stats[trade.regime_id]["count"] += 1
        self.regime_stats[trade.regime_id]["total_pnl"] += trade.pnl
        
    def update_regime(self, regime_id: str, timestamp: float):
        if self.current_regime != regime_id:
            # Transition
            if self.last_regime_update > 0:
                duration = timestamp - self.last_regime_update
                self.regime_stats[self.current_regime]["dwell_time"] += duration
            
            self.transitions += 1
            self.current_regime = regime_id
            self.last_regime_update = timestamp
            
    def compute(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "final_equity": self.equity,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "calmar_ratio": 0.0,
                "regime_accuracy": 0.0,
                "avg_dwell_time": 0.0,
                "churn_rate": 0.0,
                "pnl_skewness": 0.0,
                "transitions": self.transitions,
                "regime_analytics": dict(self.regime_stats),
                "dsr_p_value": 1.0,
                "dsr_is_significant": False,
                "dsr_deflated_sharpe": 0.0,
                "status": "No trades"
            }
            
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Stats Base
        total_pnl = df['pnl'].sum()
        win_rate = len(df[df['pnl'] > 0]) / len(df)
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0
        
        # Drawdown (Condi au trading - déjà implicite dans MAJ courbe equity)
        equity_series = equity_df['equity']
        peaks = equity_series.cummax()
        drawdowns = (equity_series - peaks) / peaks
        max_dd = drawdowns.min()
        
        # Sharpe Ratio — from daily aggregated returns (institutional standard)
        # Avoids inflating Sharpe via sqrt(trades_per_day) for HFT strategies
        pnl_mean = df['pnl'].mean()
        pnl_std = df['pnl'].std()

        if 'timestamp' in df.columns and len(df) >= 2:
            df_copy = df.copy()
            df_copy['day'] = (df_copy['timestamp'] // 86400).astype(int)
            daily_pnl = df_copy.groupby('day')['pnl'].sum()
            daily_returns = daily_pnl.values / max(self.initial_equity, 1.0)

            if len(daily_returns) >= 2 and np.std(daily_returns, ddof=1) > 1e-15:
                sharpe_ratio = float(
                    np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0.0
        else:
            # Fallback: per-trade Sharpe (only if no timestamp data)
            sharpe_ratio = float(pnl_mean / pnl_std * np.sqrt(252)) if pnl_std > 1e-10 else 0.0
        
        # Calmar Ratio (total return / max drawdown)
        return_pct = total_pnl / self.initial_equity if self.initial_equity > 0 else 0.0
        calmar_ratio = float(return_pct / abs(max_dd)) if abs(max_dd) > 1e-10 else 0.0
        
        # Regime accuracy: fraction of trades NOT in noise regime (-1)
        noise_trades = len(df[df['regime_id'].astype(str) == '-1'])
        regime_accuracy = 1.0 - (noise_trades / len(df))
        
        # Avg dwell time across regimes
        dwell_times = [v.get('dwell_time', 0.0) for v in self.regime_stats.values()]
        avg_dwell_time = float(np.mean(dwell_times)) if dwell_times else 0.0
        
        # Churn rate: transitions per 1000 trades
        churn_rate = (self.transitions / len(df)) * 1000 if len(df) > 0 else 0.0
        
        # Skewness & Kurtosis PnL
        pnl_skew = skew(df['pnl'])
        from scipy.stats import kurtosis as scipy_kurtosis
        excess_kurt = float(scipy_kurtosis(df['pnl'], fisher=True))  # Excess kurtosis (normal=0)
        
        # Pire Perte Transition
        worst_loss = df['pnl'].min()
        
        # Rendement Ajusté Entropie
        # Somme PnL / Somme Entropie Abs (Proxy pour "Coût du Désordre")
        # Si entry_entropy manquant (0), utilise 1.0 pour éviter div par zero si tout 0
        total_entropy = df['entry_entropy'].abs().sum()
        entropy_adj_return = total_pnl / total_entropy if total_entropy > 0 else total_pnl
        
        # ── Enhanced metrics ──
        
        # Sortino Ratio (downside deviation)
        downside = df[df['pnl'] < 0]['pnl']
        downside_std = downside.std() if len(downside) > 1 else pnl_std
        sortino_ratio = float(pnl_mean / downside_std) if downside_std > 1e-10 else 0.0
        
        # Omega Ratio (sum of gains / sum of losses)
        gains = df[df['pnl'] > 0]['pnl'].sum()
        losses = abs(df[df['pnl'] < 0]['pnl'].sum())
        omega_ratio = float(gains / losses) if losses > 1e-10 else float('inf')
        
        # Consecutive wins/losses
        signs = np.sign(df['pnl'].values)
        max_consec_wins = self._max_consecutive(signs, 1)
        max_consec_losses = self._max_consecutive(signs, -1)
        
        # Expectancy
        expectancy = float(pnl_mean)
        
        # Profit per trade
        avg_pnl = float(df['pnl'].mean())

        # Average trade size (qty) for L2 capacity penalty
        avg_trade_size = float(df['qty'].mean()) if 'qty' in df.columns else 0.0

        # MAE/MFE aggregates
        mae_values = df['mae_pct'].values
        mfe_values = df['mfe_pct'].values
        avg_mae = float(np.mean(mae_values)) if len(mae_values) > 0 else 0.0
        avg_mfe = float(np.mean(mfe_values)) if len(mfe_values) > 0 else 0.0
        # Edge capture: how much of MFE was realized as profit
        edge_capture = float(pnl_mean / avg_mfe) if avg_mfe > 1e-10 else 0.0
        
        return {
            "total_trades": len(df),
            "total_pnl": total_pnl,
            "final_equity": self.equity,
            "win_rate": win_rate,
            "profit_factor": abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(df[df['pnl'] < 0]) > 0 else float('inf'),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_dd,
            
            # OPTIMIZATION METRICS
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "omega_ratio": omega_ratio,
            "calmar_ratio": calmar_ratio,
            "regime_accuracy": regime_accuracy,
            "avg_dwell_time": avg_dwell_time,
            "churn_rate": churn_rate,
            
            # ENHANCED METRICS
            "expectancy": expectancy,
            "avg_pnl": avg_pnl,
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "avg_trade_size": avg_trade_size,
            
            # MAE/MFE ANALYTICS
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "edge_capture_ratio": edge_capture,
            "mfe_to_mae_ratio": float(avg_mfe / max(abs(avg_mae), 1e-10)),
            
            # METRIQUES SUPERIEURES
            "pnl_skewness": pnl_skew,
            "worst_transition_loss": worst_loss,
            "entropy_adjusted_return": entropy_adj_return,
            
            "transitions": self.transitions,
            "regime_analytics": dict(self.regime_stats),

            # Raw PnL series for DSR/CSCV statistical validation
            "trade_pnls": df['pnl'].tolist(),

            # Deflated Sharpe Ratio — auto-wired for multiple-testing correction
            **self._compute_dsr(sharpe_ratio, len(df), pnl_skew, excess_kurt),
        }

    @staticmethod
    def _compute_dsr(sharpe: float, n_obs: int, skewness: float, kurtosis: float, n_trials: int = 1) -> Dict:
        """Compute Deflated Sharpe Ratio for multiple-testing correction."""
        if n_obs < 10 or abs(sharpe) < 1e-10:
            return {"dsr_p_value": 1.0, "dsr_is_significant": False, "dsr_deflated_sharpe": 0.0}
        try:
            result = deflated_sharpe(
                observed_sharpe=sharpe,
                n_trials=max(n_trials, 1),
                n_observations=n_obs,
                skewness=float(skewness),
                kurtosis=float(kurtosis),  # deflated_sharpe expects excess kurtosis (fisher=True)
            )
            return {
                "dsr_p_value": result.p_value,
                "dsr_is_significant": result.is_significant,
                "dsr_deflated_sharpe": result.deflated_sharpe,
            }
        except Exception:
            return {"dsr_p_value": 1.0, "dsr_is_significant": False, "dsr_deflated_sharpe": 0.0}

    @staticmethod
    def _max_consecutive(signs: np.ndarray, target: int) -> int:
        # l'usine a gaz
        max_count = 0
        current = 0
        for s in signs:
            if s == target:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count
