import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class BacktestPlotter:
    # fais peter les graphiques
    # petits dessins
    
    @staticmethod
    def plot_equity_curve(metrics: Dict[str, Any], trades: List[Any], save_path: str = None):
        if not trades:
            print("Pas de trades à tracer.")
            return

        df_trades = pd.DataFrame([t.__dict__ for t in trades])
        # Reconstruire série equity prop si besoin, ou utiliser celle de métriques si passée
        # Supposons calc métriques fait et stats finales dispos. 
        # Mais pour plot besoin série tempo.
        
        # Reconstruisons série equity simple depuis trades pour l'instant
        # Idéalement faudrait courbe equity complète depuis objet métriques
        
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Equity
        ax1.plot(df_trades['timestamp'], df_trades['cumulative_pnl'], label='Equity')
        ax1.set_title('Courbe Equity')
        ax1.set_ylabel('PnL')
        ax1.grid(True)
        
        # Drawdown
        equity = df_trades['cumulative_pnl']
        peaks = equity.cummax()
        drawdown = (equity - peaks) 
        
        ax2.fill_between(df_trades['timestamp'], drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Timestamp')
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_regime_distribution(metrics: Dict[str, Any], save_path: str = None):
        regime_stats = metrics.get('regime_analytics', {})
        if not regime_stats:
            return
            
        labels = list(regime_stats.keys())
        counts = [stats['count'] for stats in regime_stats.values()]
        pnls = [stats['total_pnl'] for stats in regime_stats.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compte Trades par Régime
        ax1.bar(labels, counts)
        ax1.set_title('Nb Trades par Régime')
        ax1.tick_params(axis='x', rotation=45)
        
        # PnL par Régime
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(labels, pnls, color=colors)
        ax2.set_title('PnL Total par Régime')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
