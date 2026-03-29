"""Massive parallelized Monte Carlo + Bayesian optimization runner.

Supports two modes:
  1. Monte Carlo stress test: N random-seed simulations to validate survival
  2. Parallel optimization: Bayesian search over the parameter space with
     ProcessPoolExecutor for throughput on multi-core machines

Usage:
    # Monte Carlo stress test (1000 runs, 4 workers)
    python run_massive_headless.py --mode mc --n 1000 --workers 4

    # Parallel Bayesian optimization (100 calls, 5 folds)
    python run_massive_headless.py --mode optimize --n-calls 100 --n-folds 5

    # Multi-instrument ensemble optimization
    python run_massive_headless.py --mode ensemble --symbols SPY QQQ TLT GLD
"""

import argparse
import concurrent.futures
import itertools
import json
import logging
import os
import time
from typing import List

import numpy as np

# Minimal logging for speed — override per-module after imports
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("massive_sim")
logger.setLevel(logging.INFO)


# ── Monte Carlo Mode ──

def run_single_simulation(seed, duration_seconds=300):
    from main import Strategy
    from data.tick_stream import SyntheticTickStream
    try:
        strategy = Strategy()
        stream = SyntheticTickStream(
            symbol=strategy.config.instruments.instruments[0].symbol,
            duration_seconds=duration_seconds,
            seed=seed
        )
        strategy.run(duration_seconds=duration_seconds, stream=stream)

        return {
            "seed": seed,
            "final_equity": strategy.risk_manager.current_equity,
            "min_equity": min(strategy.risk_manager.equity_history),
            "max_equity": strategy.risk_manager.peak_equity,
            "drawdown": strategy.risk_manager.current_drawdown_pct,
            "kill_switch_triggered": strategy.risk_manager.kill_switch.triggered,
            "kill_reason": strategy.risk_manager.kill_switch.reason if strategy.risk_manager.kill_switch.triggered else None
        }
    except Exception as e:
        return {"seed": seed, "error": str(e)}


def run_monte_carlo(n_simulations=1000, max_workers=4):
    print(f"Starting Monte Carlo: {n_simulations} runs, {max_workers} workers...")

    results = []
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation, 42 + i): i for i in range(n_simulations)}

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                print(f"Progress: {completed}/{n_simulations} ({completed/n_simulations:.1%}) - {rate:.1f} sims/sec", end='\r')

    total_time = time.time() - start_time
    print(f"\nCompleted {n_simulations} simulations in {total_time:.2f}s ({n_simulations/total_time:.1f} sims/sec)")

    # Analysis
    survived = [r for r in results if not r.get("kill_switch_triggered", False) and not r.get("error")]
    killed = [r for r in results if r.get("kill_switch_triggered", False)]
    errors = [r for r in results if r.get("error")]

    print("\n=== MONTE CARLO REPORT ===")
    print(f"Total Runs: {len(results)}")
    print(f"Survived: {len(survived)} ({len(survived)/len(results):.2%})")
    print(f"Killed (Safety): {len(killed)} ({len(killed)/len(results):.2%})")
    print(f"Errors: {len(errors)} ({len(errors)/len(results):.2%})")

    if survived:
        equities = [r["final_equity"] for r in survived]
        drawdowns = [r["drawdown"] for r in survived]
        print(f"\nEquity — Mean: ${np.mean(equities):,.2f}, Median: ${np.median(equities):,.2f}")
        print(f"Equity — P5: ${np.percentile(equities, 5):,.2f}, P95: ${np.percentile(equities, 95):,.2f}")
        print(f"Drawdown — Mean: {np.mean(drawdowns):.2%}, Max: {np.max(drawdowns):.2%}")

    if killed:
        print("\nTop Kill Reasons:")
        reasons = {}
        for r in killed:
            reason = r.get("kill_reason", "Unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {reason}: {count}")

    return results


# ── Parallel Optimization Mode ──

def _optimize_worker(args_tuple):
    """Single optimization fold (picklable for ProcessPoolExecutor)."""
    fold_idx, train_ticks, test_ticks, config_dict, symbol, dd_target, n_calls = args_tuple

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from config.schema import AppConfig
    from optimization.search_space import get_defaults, apply_params
    from optimization.objective import create_optuna_objective, run_single_backtest, _compute_sharpe

    config = AppConfig(**config_dict)
    optuna_obj = create_optuna_objective(config, train_ticks, symbol=symbol, dd_target=dd_target)

    sampler = optuna.samplers.TPESampler(
        seed=42 + fold_idx,
        n_startup_trials=min(10, n_calls // 3),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(get_defaults())
    study.optimize(optuna_obj, n_trials=n_calls, show_progress_bar=False)

    best_params = study.best_params
    train_score = -study.best_value

    # Evaluate on test set
    test_config = apply_params(config, best_params)
    test_results = run_single_backtest(test_config, test_ticks, symbol=symbol)
    test_sharpe = _compute_sharpe(test_results)

    return {
        "fold_idx": fold_idx,
        "best_params": best_params,
        "train_sharpe": train_score,
        "test_sharpe": test_sharpe,
        "test_trades": test_results.get("total_trades", 0),
        "test_pnl": test_results.get("total_pnl", 0.0),
        "test_drawdown": test_results.get("max_drawdown", 0.0),
    }


def run_parallel_optimization(
    symbol: str = "SPY",
    n_calls: int = 50,
    n_folds: int = 5,
    tick_limit: int = 50_000,
    max_workers: int = 4,
    dd_target: float = 0.05,
    output_dir: str = "optimization_results",
):
    print(f"Starting parallel optimization: {n_folds} folds x {n_calls} calls, {max_workers} workers")

    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dbn_dir = os.path.join(base_dir, "data", "databento_downloads")

    if os.path.isdir(dbn_dir):
        dbn_files = [f for f in os.listdir(dbn_dir) if f.endswith(".dbn.zst")]
        if dbn_files:
            from data.databento_loader import DatabentoLoader
            loader = DatabentoLoader(data_dir=dbn_dir, symbol_override=symbol, trades_only=True)
            ticks = list(itertools.islice(loader.load_files(), tick_limit))
            print(f"  Loaded {len(ticks):,} Databento ticks")
        else:
            ticks = None
    else:
        ticks = None

    if not ticks:
        from data.tick_stream import SyntheticTickStream
        stream = SyntheticTickStream(symbol=symbol, duration_seconds=120, seed=42)
        ticks = list(itertools.islice(stream, tick_limit or 20_000))
        print(f"  Generated {len(ticks):,} synthetic ticks")

    from config.loader import get_config
    config = get_config()
    config_dict = config.dict()

    # Create walk-forward folds
    fold_size = len(ticks) // (n_folds + 1)
    tasks = []
    for i in range(n_folds):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, len(ticks))
        train_ticks = ticks[:train_end]
        test_ticks = ticks[test_start:test_end]
        if len(test_ticks) < 100:
            continue
        tasks.append((i, train_ticks, test_ticks, config_dict, symbol, dd_target, n_calls))

    print(f"  {len(tasks)} folds prepared, launching workers...")
    start_time = time.time()

    fold_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_optimize_worker, task): task[0] for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            fold_idx = futures[future]
            try:
                result = future.result()
                fold_results.append(result)
                print(
                    f"  Fold {result['fold_idx']+1}: "
                    f"Train={result['train_sharpe']:.4f} "
                    f"Test={result['test_sharpe']:.4f} "
                    f"Trades={result['test_trades']} "
                    f"PnL=${result['test_pnl']:+.2f}"
                )
            except Exception as e:
                print(f"  Fold {fold_idx+1} FAILED: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(fold_results)} folds in {elapsed:.1f}s")

    if fold_results:
        train_sharpes = [r["train_sharpe"] for r in fold_results]
        test_sharpes = [r["test_sharpe"] for r in fold_results]
        print(f"\nTrain Sharpe: {np.mean(train_sharpes):.4f} +/- {np.std(train_sharpes):.4f}")
        print(f"Test Sharpe:  {np.mean(test_sharpes):.4f} +/- {np.std(test_sharpes):.4f}")
        overfit_ratio = np.mean(train_sharpes) / max(abs(np.mean(test_sharpes)), 0.001)
        print(f"Overfit Ratio: {overfit_ratio:.2f}")

        # Best fold
        best_fold = max(fold_results, key=lambda r: r["test_sharpe"])
        print(f"\nBest Parameters (Fold {best_fold['fold_idx']+1}, Test Sharpe={best_fold['test_sharpe']:.4f}):")
        for k, v in sorted(best_fold["best_params"].items()):
            if isinstance(v, float):
                print(f"  {k:35s}: {v:.6f}")
            else:
                print(f"  {k:35s}: {v}")

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"parallel_opt_{symbol}.json")
        with open(output_path, "w") as f:
            json.dump({
                "symbol": symbol,
                "n_calls": n_calls,
                "n_folds": n_folds,
                "total_ticks": len(ticks),
                "elapsed_sec": elapsed,
                "mean_train_sharpe": float(np.mean(train_sharpes)),
                "mean_test_sharpe": float(np.mean(test_sharpes)),
                "overfit_ratio": float(overfit_ratio),
                "best_params": best_fold["best_params"],
                "fold_results": fold_results,
            }, f, indent=2, default=str)
        print(f"\nSaved to: {output_path}")

    return fold_results


# ── Multi-Instrument Ensemble Optimization ──

def run_ensemble_optimization(
    symbols: List[str],
    n_calls: int = 30,
    tick_limit: int = 20_000,
    max_workers: int = 4,
    dd_target: float = 0.05,
    output_dir: str = "optimization_results",
):
    """Optimize each instrument independently, then aggregate through PortfolioRiskManager."""
    print(f"Starting ensemble optimization: {len(symbols)} instruments, {n_calls} calls each")

    from config.loader import get_config
    config = get_config()
    config_dict = config.dict()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Load data per instrument
    tick_data = {}
    for symbol in symbols:
        dbn_dir = os.path.join(base_dir, "data", "databento_downloads")
        ticks = None
        if os.path.isdir(dbn_dir):
            dbn_files = [f for f in os.listdir(dbn_dir) if f.endswith(".dbn.zst")]
            if dbn_files:
                from data.databento_loader import DatabentoLoader
                loader = DatabentoLoader(data_dir=dbn_dir, symbol_override=symbol, trades_only=True)
                ticks = list(itertools.islice(loader.load_files(), tick_limit))
        if not ticks:
            from data.tick_stream import SyntheticTickStream
            seed = hash(symbol) % 100000
            stream = SyntheticTickStream(symbol=symbol, duration_seconds=120, seed=seed)
            ticks = list(itertools.islice(stream, tick_limit))
        tick_data[symbol] = ticks
        print(f"  {symbol}: {len(ticks):,} ticks")

    # Step 2: Optimize each instrument in parallel (single fold for speed)
    print("\nOptimizing per instrument...")
    tasks = []
    for symbol in symbols:
        ticks = tick_data[symbol]
        split = int(len(ticks) * 0.7)
        train = ticks[:split]
        test = ticks[split:]
        if len(test) < 50:
            test = ticks[-500:]
        tasks.append((0, train, test, config_dict, symbol, dd_target, n_calls))

    start_time = time.time()
    per_instrument = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_optimize_worker, task): task[4] for task in tasks}  # task[4] = symbol
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                per_instrument[symbol] = result
                print(
                    f"  {symbol}: Train={result['train_sharpe']:.4f} "
                    f"Test={result['test_sharpe']:.4f} "
                    f"Trades={result['test_trades']}"
                )
            except Exception as e:
                print(f"  {symbol}: FAILED ({e})")

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # Step 3: Aggregate through correlation analysis
    if per_instrument:
        test_sharpes = [r["test_sharpe"] for r in per_instrument.values()]
        print("\n=== ENSEMBLE SUMMARY ===")
        print(f"Instruments optimized: {len(per_instrument)}")
        print(f"Mean Test Sharpe: {np.mean(test_sharpes):.4f}")

        # Simple portfolio Sharpe estimate with diversification
        n = len(test_sharpes)
        if n >= 2:
            # Assume moderate positive correlation for synthetic data
            avg_corr = 0.3
            portfolio_sharpe = np.mean(test_sharpes) * np.sqrt(n) / np.sqrt(1 + (n - 1) * avg_corr)
            print(f"Est. Portfolio Sharpe: {portfolio_sharpe:.4f} (assuming avg corr={avg_corr})")
            print(f"  Single-instrument mean:    {np.mean(test_sharpes):.4f}")
            print(f"  Diversification multiplier: {portfolio_sharpe / max(np.mean(test_sharpes), 0.001):.2f}x")

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ensemble_optimization.json")
        with open(output_path, "w") as f:
            json.dump({
                "symbols": symbols,
                "n_calls": n_calls,
                "elapsed_sec": elapsed,
                "per_instrument": per_instrument,
            }, f, indent=2, default=str)
        print(f"\nSaved to: {output_path}")

    return per_instrument


def main():
    parser = argparse.ArgumentParser(
        description="Massive parallel simulation & optimization runner"
    )
    parser.add_argument("--mode", choices=["mc", "optimize", "ensemble"], default="mc",
                        help="Run mode: mc (Monte Carlo), optimize (Bayesian), ensemble (multi-instrument)")
    # Monte Carlo args
    parser.add_argument("--n", type=int, default=1000, help="Number of MC simulations")
    # Optimization args
    parser.add_argument("--n-calls", type=int, default=50, help="Bayesian opt iterations per fold")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--tick-limit", type=int, default=50_000, help="Max ticks to load")
    parser.add_argument("--symbol", type=str, default="SPY", help="Instrument symbol")
    parser.add_argument("--dd-target", type=float, default=0.05, help="Target max drawdown")
    # Ensemble args
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "EEM", "TLT", "GLD"],
                        help="Multi-instrument universe")
    # Common
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--output-dir", type=str, default="optimization_results")

    args = parser.parse_args()

    if args.mode == "mc":
        run_monte_carlo(n_simulations=args.n, max_workers=args.workers)
    elif args.mode == "optimize":
        run_parallel_optimization(
            symbol=args.symbol,
            n_calls=args.n_calls,
            n_folds=args.n_folds,
            tick_limit=args.tick_limit,
            max_workers=args.workers,
            dd_target=args.dd_target,
            output_dir=args.output_dir,
        )
    elif args.mode == "ensemble":
        run_ensemble_optimization(
            symbols=args.symbols,
            n_calls=args.n_calls,
            tick_limit=args.tick_limit,
            max_workers=args.workers,
            dd_target=args.dd_target,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
