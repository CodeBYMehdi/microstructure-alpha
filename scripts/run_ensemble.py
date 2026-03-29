"""Multi-Instrument Ensemble Backtest

Runs independent backtests per symbol, then feeds per-instrument PnL
through PortfolioRiskManager to compute correlation-adjusted capital
allocation and aggregate performance.

The VaR constraint ($10M nominal cap, 2% VaR) is the binding limit.
Low inter-instrument correlation unlocks more gross exposure than any
single-instrument run, which is where the multiplicative PnL scaling
lives.

Usage:
    python scripts/run_ensemble.py --symbols SPY QQQ EEM TLT GLD
    python scripts/run_ensemble.py --symbols SPY QQQ --tick-limit 50000
"""

import argparse
import copy
import itertools
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any

import numpy as np

# Configure logging before heavy imports
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ensemble")
logger.setLevel(logging.INFO)

from config.loader import get_config
from optimization.objective import run_single_backtest
from risk.portfolio import PortfolioRiskManager


# ── Default universe: weakly-correlated macro instruments ──
DEFAULT_UNIVERSE = ["SPY", "QQQ", "EEM", "TLT", "GLD"]

# Approximate starting prices for synthetic sizing (only used for synthetic data)
_APPROX_PRICES = {
    "SPY": 530.0, "QQQ": 470.0, "EEM": 42.0, "TLT": 92.0, "GLD": 220.0,
    "AAPL": 190.0, "MSFT": 420.0, "NVDA": 130.0, "AMZN": 185.0,
    "META": 500.0, "TSLA": 175.0, "GOOGL": 170.0, "IWM": 210.0,
    "XLF": 42.0, "XLE": 88.0, "HYG": 77.0, "LQD": 110.0,
}


def _load_ticks_for_symbol(symbol: str, data_dir: str, tick_limit: int = None) -> list:
    """Load tick data for a symbol. Falls back to synthetic."""
    dbn_dir = os.path.join(data_dir, "data", "databento_downloads")

    if os.path.isdir(dbn_dir):
        dbn_files = [f for f in os.listdir(dbn_dir) if f.endswith(".dbn.zst")]
        if dbn_files:
            from data.databento_loader import DatabentoLoader
            loader = DatabentoLoader(data_dir=dbn_dir, symbol_override=symbol, trades_only=True)
            stream = loader.load_files()
            if tick_limit:
                return list(itertools.islice(stream, tick_limit))
            return list(stream)

    # Synthetic fallback with symbol-specific seed for diversity
    from data.tick_stream import SyntheticTickStream
    seed = hash(symbol) % 100000
    stream = SyntheticTickStream(symbol=symbol, duration_seconds=120, seed=seed)
    return list(itertools.islice(stream, tick_limit or 20_000))


def _run_instrument(args_tuple) -> Dict[str, Any]:
    """Run a single-instrument backtest (picklable for ProcessPoolExecutor)."""
    symbol, tick_list, base_config_dict = args_tuple

    # Rebuild config from dict (configs aren't picklable across processes)
    from config.schema import AppConfig
    config = AppConfig(**base_config_dict)
    config = copy.deepcopy(config)

    # Override instrument
    config.instruments.instruments[0].symbol = symbol
    config.instruments.instruments[0].exchange = "SMART"
    config.instruments.instruments[0].tick_size = 0.01
    config.instruments.instruments[0].lot_size = 1.0
    config.instruments.instruments[0].min_notional = 100.0

    try:
        results = run_single_backtest(config, tick_list, symbol=symbol)
        results["symbol"] = symbol
        return results
    except Exception as e:
        return {
            "symbol": symbol,
            "total_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "error": str(e),
        }


def compute_ensemble_metrics(
    per_instrument: List[Dict[str, Any]],
    initial_equity: float = 100_000.0,
) -> Dict[str, Any]:
    """Aggregate per-instrument results through PortfolioRiskManager.

    Simulates the correlation-adjusted capital allocation that would
    occur if all instruments ran simultaneously.
    """
    prm = PortfolioRiskManager(initial_equity=initial_equity)

    total_pnl = 0.0
    total_trades = 0
    instrument_sharpes = []

    active_symbols = []
    for r in per_instrument:
        symbol = r["symbol"]
        n_trades = r.get("total_trades", 0)
        pnl = r.get("total_pnl", 0.0)
        sharpe = r.get("sharpe_ratio", 0.0)

        if n_trades == 0:
            continue

        active_symbols.append(symbol)
        total_pnl += pnl
        total_trades += n_trades
        instrument_sharpes.append(sharpe)

        # Simulate position updates for correlation matrix
        avg_price = _APPROX_PRICES.get(symbol, 100.0)
        avg_size = r.get("avg_trade_size", 1.0)
        prm.update_position(symbol, avg_size, avg_price)

        # Feed synthetic returns for correlation estimation
        # Use PnL / notional as return proxy
        notional = avg_price * max(avg_size, 1.0)
        per_trade_pnl = pnl / n_trades if n_trades > 0 else 0.0
        per_trade_return = per_trade_pnl / notional if notional > 0 else 0.0

        for _ in range(min(n_trades, 60)):
            noise = np.random.normal(0, abs(per_trade_return) * 0.5)
            prm._return_history[symbol].append(per_trade_return + noise)

        prm.update_equity(pnl)

    # Compute correlation matrix across instruments
    corr_matrix, corr_symbols = prm.compute_correlation_matrix()
    avg_correlation = prm.get_average_correlation()

    # Compute portfolio VaR
    var_95, cvar_95 = prm.compute_var(0.95)

    # Risk regime
    regime = prm.detect_risk_regime()
    budget = prm.get_current_budget()

    # Exposure
    exposure = prm.get_exposure()

    # Diversification ratio: sum of individual vols / portfolio vol
    # Higher = more diversification benefit
    if len(instrument_sharpes) >= 2:
        portfolio_sharpe = np.mean(instrument_sharpes) * np.sqrt(len(instrument_sharpes))
        # Adjust for correlation drag
        if avg_correlation < 1.0:
            correlation_boost = np.sqrt(1.0 / (1.0 + (len(active_symbols) - 1) * avg_correlation))
        else:
            correlation_boost = 1.0
        diversified_sharpe = np.mean(instrument_sharpes) * np.sqrt(len(instrument_sharpes)) * correlation_boost
    else:
        portfolio_sharpe = instrument_sharpes[0] if instrument_sharpes else 0.0
        diversified_sharpe = portfolio_sharpe
        correlation_boost = 1.0

    # Capital utilization: how much of the budget we're using
    budget_utilization = exposure['gross'] / max(budget.max_gross_exposure, 1.0)

    # Theoretical max gross at this correlation level
    # With N instruments at avg_corr, diversification lets us hold more
    n_instruments = len(active_symbols)
    if n_instruments > 1 and avg_correlation < 1.0:
        effective_instruments = 1.0 / (1.0 + (n_instruments - 1) * max(avg_correlation, 0.0))
        theoretical_capacity_mult = np.sqrt(n_instruments * effective_instruments)
    else:
        theoretical_capacity_mult = 1.0

    return {
        "n_instruments": n_instruments,
        "active_symbols": active_symbols,
        "total_pnl": total_pnl,
        "total_trades": total_trades,

        # Per-instrument Sharpes
        "instrument_sharpes": {
            r["symbol"]: r.get("sharpe_ratio", 0.0)
            for r in per_instrument if r.get("total_trades", 0) > 0
        },
        "mean_instrument_sharpe": float(np.mean(instrument_sharpes)) if instrument_sharpes else 0.0,

        # Portfolio-level metrics
        "portfolio_sharpe": float(diversified_sharpe),
        "correlation_boost": float(correlation_boost),
        "avg_pairwise_correlation": float(avg_correlation),
        "theoretical_capacity_multiplier": float(theoretical_capacity_mult),

        # Risk
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "risk_regime": regime.value,
        "drawdown_pct": float(prm.current_drawdown_pct),
        "equity": float(prm.current_equity),
        "peak_equity": float(prm.peak_equity),

        # Exposure
        "gross_exposure": float(exposure['gross']),
        "net_exposure": float(exposure['net']),
        "budget_max_gross": float(budget.max_gross_exposure),
        "budget_utilization_pct": float(budget_utilization),

        # Correlation matrix (for downstream analysis)
        "correlation_matrix": corr_matrix.tolist() if len(corr_matrix) > 1 else [],
        "correlation_symbols": corr_symbols,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Instrument Ensemble Backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE,
                        help=f"Instrument universe (default: {' '.join(DEFAULT_UNIVERSE)})")
    parser.add_argument("--tick-limit", type=int, default=20_000,
                        help="Max ticks per instrument (default: 20000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="Initial portfolio equity (default: 100000)")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                        help="Output directory")
    args = parser.parse_args()

    symbols = args.symbols
    print("=" * 70)
    print("  MULTI-INSTRUMENT ENSEMBLE BACKTEST")
    print("=" * 70)
    print(f"  Universe: {', '.join(symbols)}")
    print(f"  Equity: ${args.equity:,.0f}")
    print(f"  Tick limit: {args.tick_limit:,} per instrument")
    print(f"  Workers: {args.workers}")
    print()

    # Load tick data for each instrument
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("[1/3] Loading tick data...")
    tick_data: Dict[str, list] = {}
    for symbol in symbols:
        ticks = _load_ticks_for_symbol(symbol, base_dir, args.tick_limit)
        tick_data[symbol] = ticks
        print(f"  {symbol}: {len(ticks):,} ticks")

    # Prepare configs (serialize for pickling across processes)
    config = get_config()
    config_dict = config.dict()

    # Run per-instrument backtests in parallel
    print("\n[2/3] Running per-instrument backtests...")
    start_time = time.time()

    tasks = [
        (symbol, tick_data[symbol], config_dict)
        for symbol in symbols
    ]

    per_instrument_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_instrument, task): task[0] for task in tasks}
        for future in as_completed(futures):
            symbol = futures[future]
            result = future.result()
            per_instrument_results.append(result)
            n_trades = result.get("total_trades", 0)
            pnl = result.get("total_pnl", 0.0)
            sharpe = result.get("sharpe_ratio", 0.0)
            error = result.get("error", "")
            status = f"ERROR: {error}" if error else f"{n_trades} trades, PnL=${pnl:+.2f}, Sharpe={sharpe:.3f}"
            print(f"  {symbol}: {status}")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # Ensemble aggregation through PortfolioRiskManager
    print("\n[3/3] Computing ensemble metrics...")
    ensemble = compute_ensemble_metrics(per_instrument_results, initial_equity=args.equity)

    # Print report
    print("\n" + "=" * 70)
    print("  ENSEMBLE RESULTS")
    print("=" * 70)
    print(f"  Active instruments:    {ensemble['n_instruments']} / {len(symbols)}")
    print(f"  Total trades:          {ensemble['total_trades']:,}")
    print(f"  Total PnL:             ${ensemble['total_pnl']:+,.2f}")
    print(f"  Equity:                ${ensemble['equity']:,.2f}")
    print(f"  Drawdown:              {ensemble['drawdown_pct']:.2%}")
    print()

    print(f"  {'--- Per-Instrument Sharpe ---':^66}")
    for sym, sr in ensemble['instrument_sharpes'].items():
        print(f"    {sym:>8s}: {sr:>10.4f}")
    print(f"    {'Mean':>8s}: {ensemble['mean_instrument_sharpe']:>10.4f}")

    print()
    print(f"  {'--- Portfolio Diversification ---':^66}")
    print(f"  Avg pairwise corr:     {ensemble['avg_pairwise_correlation']:>10.4f}")
    print(f"  Correlation boost:     {ensemble['correlation_boost']:>10.4f}x")
    print(f"  Capacity multiplier:   {ensemble['theoretical_capacity_multiplier']:>10.4f}x")
    print(f"  Portfolio Sharpe:      {ensemble['portfolio_sharpe']:>10.4f}")

    print()
    print(f"  {'--- Risk Budget ---':^66}")
    print(f"  Risk regime:           {ensemble['risk_regime']:>10s}")
    print(f"  Gross exposure:        ${ensemble['gross_exposure']:>12,.2f}")
    print(f"  Net exposure:          ${ensemble['net_exposure']:>12,.2f}")
    print(f"  Budget max gross:      ${ensemble['budget_max_gross']:>12,.2f}")
    print(f"  Budget utilization:    {ensemble['budget_utilization_pct']:>10.2%}")
    print(f"  VaR (95%):             ${ensemble['var_95']:>12,.2f}")
    print(f"  CVaR (95%):            ${ensemble['cvar_95']:>12,.2f}")

    if ensemble['correlation_matrix']:
        print()
        print(f"  {'--- Correlation Matrix ---':^66}")
        syms = ensemble['correlation_symbols']
        header = "          " + "  ".join(f"{s:>8s}" for s in syms)
        print(header)
        for i, row in enumerate(ensemble['correlation_matrix']):
            vals = "  ".join(f"{v:>8.3f}" for v in row)
            print(f"    {syms[i]:>6s}  {vals}")

    print("\n" + "=" * 70)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ensemble_results.json")
    with open(output_path, "w") as f:
        json.dump(ensemble, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    return ensemble


if __name__ == "__main__":
    main()
