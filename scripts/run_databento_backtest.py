# le cerveau de l'operation
# on cherche les pepites

import logging
import os
import sys
import itertools
import time
from datetime import datetime

# Configure logging BEFORE any other imports to suppress noise
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("main").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("regime.clustering").setLevel(logging.ERROR)
logging.getLogger("regime.transition").setLevel(logging.ERROR)
logging.getLogger("regime.transition_model").setLevel(logging.ERROR)
logging.getLogger("monitoring.alerts").setLevel(logging.ERROR)
logging.getLogger("execution.order_router").setLevel(logging.ERROR)
logging.getLogger("risk.calibration").setLevel(logging.ERROR)
logging.getLogger("microstructure.pdf.normalizing_flow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from data.databento_loader import DatabentoLoader
from backtest.backtest_agent import BacktestAgent
from backtest.metrics import PerformanceMetrics
from config.loader import get_config

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
# Limit total events. None = process ALL data (21 days).
# With TRADES_ONLY=False, book updates outnumber trades ~20:1.
TICK_LIMIT = None

# Symbol override: Databento data is SPY on XNAS
SYMBOL = "SPY"

# Process ALL events including L2 book updates for full microstructure features.
# The L2 data feeds depth_imbalance, book_pressure, liquidity_pull, spoofing
# detection into the decision pipeline. Set True for speed-only runs.
TRADES_ONLY = False


def run_backtest():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "databento_downloads")

    # Check for .dbn.zst files
    dbn_files = [f for f in os.listdir(data_dir) if f.endswith(".dbn.zst")]
    if not dbn_files:
        print("ERROR: No .dbn.zst files found in data/databento_downloads/")
        print("Run download_data.py first, or place Databento files there.")
        sys.exit(1)

    print(f"Found {len(dbn_files)} Databento file(s): {', '.join(dbn_files)}")

    # Override instrument config for SPY
    config = get_config()
    config.instruments.instruments[0].symbol = SYMBOL
    config.instruments.instruments[0].exchange = "XNAS"
    config.instruments.instruments[0].tick_size = 0.01
    config.instruments.instruments[0].lot_size = 1.0
    config.instruments.instruments[0].min_notional = 100.0

    # ── Regime Detection: tuned for equity tick microstructure ──
    config.thresholds.regime.transition_strength_min = 0.15  # Lower: SPY transitions are subtle
    config.thresholds.regime.window_size = 200               # ~5-10 sec of trade ticks
    config.thresholds.regime.update_frequency = 20           # Process every 20 ticks — more decision points
    config.thresholds.regime.min_cluster_size = 10           # Smaller clusters for equity
    config.thresholds.regime.min_samples = 3                 # More sensitive clustering
    config.thresholds.regime.kl_min = 0.0                    # Disabled: KL gate bypassed
    config.thresholds.regime.projection_min = 0.0            # Disabled: projection gate bypassed

    # ── Risk: relaxed for backtest discovery ──
    config.thresholds.risk.confidence_floor = 0.0
    config.thresholds.risk.slippage_tolerance = 0.05
    config.thresholds.risk.max_consecutive_errors = 100

    # ── Entry conditions: tuned for equity microstructure ──
    config.thresholds.decision.long.skew_min = 0.0
    config.thresholds.decision.long.tail_slope_min = 0.5
    config.thresholds.decision.short.volatility_min = 0.00001  # SPY vol much lower than crypto
    config.thresholds.decision.short.skew_max = 0.0
    config.thresholds.decision.short.kurtosis_min = 0.1

    # ── Exit tuning: mean-reversion friendly R:R ──
    config.thresholds.decision.exit.entropy_acceleration_threshold = 3.0  # Very lenient — entropy accel is noisy on SPY
    config.thresholds.decision.exit.fallback_strength_threshold = 0.08    # Lower fallback to catch more opportunities
    config.thresholds.decision.exit.kl_stable_threshold = 0.0             # KL always ~0 on equity

    # ── Liquidity ──
    config.thresholds.decision.liquidity.spread_max = 0.05   # SPY: ~1c spread on $450
    config.thresholds.decision.liquidity.depth_slope_min = -1.0  # More lenient for L2 book pressure

    # ── Sizing ──
    config.thresholds.decision.sizing.base_size = 100.0      # 100 shares — more skin in game
    config.thresholds.execution_sim.base_fee_bps = 0.05     # 0.05 bps fees (maker rebate)

    # Init Loader (trades_only=True skips book updates for faster processing)
    loader = DatabentoLoader(data_dir=data_dir, symbol_override=SYMBOL, trades_only=TRADES_ONLY)

    # Create tick stream (generator)
    raw_stream = loader.load_files()
    if TICK_LIMIT:
        print(f"Limiting to {TICK_LIMIT:,} ticks...")
        tick_stream = itertools.islice(raw_stream, TICK_LIMIT)
    else:
        tick_stream = raw_stream

    # Init Backtest Agent (MUST pass config so overrides propagate to components)
    agent = BacktestAgent(data_stream=tick_stream, config=config)
    # Override the symbol in the agent too
    agent.symbol = SYMBOL
    agent.microstructure.symbol = SYMBOL
    # Execute trades immediately (no persistence confirmation needed)
    agent.strategy.min_persistence = 0

    # Run
    print(f"\nStarting Databento Backtest on {SYMBOL}...")
    print("=" * 60)
    wall_start = time.time()

    try:
        results = agent.run()
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
        results = agent.metrics.compute()
    except Exception as e:
        logging.error(f"Backtest failed: {e}", exc_info=True)
        results = agent.metrics.compute()

    wall_end = time.time()
    duration = wall_end - wall_start

    # -----------------------------------------------------------------
    # Print Results
    # -----------------------------------------------------------------
    trade_ticks = agent.strategy.tick_count
    print(f"\n  Trade ticks processed by strategy: {trade_ticks:,}")
    print(f"  Windows processed: ~{max(0, trade_ticks - config.thresholds.regime.window_size) // config.thresholds.regime.update_frequency}")

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)

    for k, v in results.items():
        if k == "regime_analytics":
            continue
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>12.4f}")
        else:
            print(f"  {k:30s}: {v}")

    regime_analytics = results.get("regime_analytics", {})
    if regime_analytics:
        print(f"\n  {'--- Regime Analytics ---':^42}")
        for rid, stats in sorted(regime_analytics.items(), key=lambda x: str(x[0])):
            cnt = stats.get("count", 0)
            pnl = stats.get("total_pnl", 0.0)
            dwell = stats.get("dwell_time", 0.0)
            print(f"  Regime {str(rid):>6s}: trades={cnt:>4d}  PnL={pnl:>10.2f}  dwell={dwell:>8.1f}s")

    print(f"\n  Wall-clock time: {duration:.1f}s")
    if results.get("total_trades", 0) > 0:
        tps = results["total_trades"] / duration
        print(f"  Throughput: {tps:.0f} trades/sec")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Plot Results
    # -----------------------------------------------------------------
    try:
        _plot_results(agent.metrics, results)
    except Exception as e:
        logging.warning(f"Plotting failed (non-critical): {e}")

    return results


def _plot_results(metrics: PerformanceMetrics, results: dict):
    # on garde une trace
    # fais peter les graphiques
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    trades = metrics.trades
    equity_curve = metrics.equity_curve

    # --- Figure 1: Equity Curve + Drawdown ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    if equity_curve and len(equity_curve) > 1:
        eq_df = pd.DataFrame(equity_curve)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], unit="s", errors="coerce")

        axes[0].plot(eq_df["timestamp"], eq_df["equity"], linewidth=0.8, color="steelblue")
        axes[0].set_title(f"Equity Curve  |  Final: ${results.get('final_equity', 0):,.2f}", fontsize=13)
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        equity_s = eq_df["equity"]
        peak = equity_s.cummax()
        dd = (equity_s - peak) / peak
        axes[1].fill_between(eq_df["timestamp"], dd, 0, color="crimson", alpha=0.4)
        axes[1].set_title(f"Drawdown  |  Max: {results.get('max_drawdown', 0):.2%}", fontsize=13)
        axes[1].set_ylabel("Drawdown %")
        axes[1].set_xlabel("Time")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No equity data", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No drawdown data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig("backtest_equity.png", dpi=150)
    plt.close()
    print("  Saved: backtest_equity.png")

    # --- Figure 2: Regime Distribution ---
    regime_analytics = results.get("regime_analytics", {})
    if regime_analytics:
        labels = [str(k) for k in regime_analytics.keys()]
        counts = [v.get("count", 0) for v in regime_analytics.values()]
        pnls = [v.get("total_pnl", 0.0) for v in regime_analytics.values()]

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.bar(labels, counts, color="steelblue")
        ax1.set_title("Trades per Regime")
        ax1.set_xlabel("Regime ID")
        ax1.set_ylabel("Trade Count")
        ax1.tick_params(axis="x", rotation=45)

        colors = ["green" if p > 0 else "red" for p in pnls]
        ax2.bar(labels, pnls, color=colors)
        ax2.set_title("PnL per Regime")
        ax2.set_xlabel("Regime ID")
        ax2.set_ylabel("Total PnL ($)")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("backtest_regimes.png", dpi=150)
        plt.close()
        print("  Saved: backtest_regimes.png")


if __name__ == "__main__":
    run_backtest()
