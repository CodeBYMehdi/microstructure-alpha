"""Run a single backtest and output metrics as JSON.

Usage:
    python scripts/run_baseline.py [output_path]
"""
import json
import sys
import os
import logging

logging.basicConfig(level=logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tick_stream import SyntheticTickStream
from backtest.backtest_agent import BacktestAgent
from config.loader import get_config


def run_backtest_metrics(seed=42, duration_seconds=3600):
    config = get_config()
    symbol = config.instruments.instruments[0].symbol
    stream = SyntheticTickStream(symbol=symbol, duration_seconds=duration_seconds, seed=seed)
    ticks = list(stream)

    agent = BacktestAgent(data_stream=iter(ticks))
    agent.run()
    metrics = agent.metrics.compute()

    # Extract key fields
    return {
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "avg_trade_pnl": float(metrics.get("avg_pnl", 0.0)),
        "total_pnl": float(metrics.get("total_pnl", 0.0)),
        "n_trades": int(metrics.get("total_trades", 0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "sortino": float(metrics.get("sortino_ratio", 0.0)),
    }


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results/baseline.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = run_backtest_metrics()
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_path}:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:>12.4f}")
        else:
            print(f"  {k:20s}: {v:>12d}")
