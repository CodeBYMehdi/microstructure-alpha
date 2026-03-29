import logging
import sys
import os
from datetime import datetime

# Ajoute racine projet au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.backtest_agent import BacktestAgent
from data.tick_stream import SyntheticTickStream
from backtest.plotters import BacktestPlotter

# Config logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def main():
    # 1. Setup Flux Données
    # 24 heures de données pour capturer plus de chgts régime et trades
    stream = SyntheticTickStream(
        symbol="BTC-USDT",
        start_time=datetime(2024, 1, 1, 9, 0, 0),
        duration_seconds=86400, 
        seed=42
    )
    
    # 2. Init Agent
    agent = BacktestAgent(data_stream=stream)
    
    # 3. Exéc Backtest
    print("Exécution Backtest...")
    metrics = agent.run()
    
    # 4. Sortie Résultats
    print("\n>>> RÉSULTATS BACKTEST <<<")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"PnL Total: {metrics['total_pnl']:.2f}")
    print(f"Equity Finale: {metrics['final_equity']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Skewness PnL: {metrics.get('pnl_skewness', 0.0):.4f}")
    print(f"Rendement Adj-Entropie: {metrics.get('entropy_adjusted_return', 0.0):.4f}")
    print(f"Pire Perte Transition: {metrics.get('worst_transition_loss', 0.0):.4f}")
    print(f"Transitions: {metrics['transitions']}")
    
    print("\nAnalytics Régime:")
    for regime, stats in metrics['regime_analytics'].items():
        print(f"  {regime}: Count={stats['count']}, PnL={stats['total_pnl']:.2f}")

    # 5. Plot
    print("\nGénération Plots...")
    try:
        BacktestPlotter.plot_equity_curve(metrics, agent.metrics.trades, save_path="backtest_equity.png")
        BacktestPlotter.plot_regime_distribution(metrics, save_path="backtest_regimes.png")
        print("Plots sauvegardés dans backtest_equity.png et backtest_regimes.png")
    except Exception as e:
        print(f"Plotting échoué: {e}")

if __name__ == "__main__":
    main()
