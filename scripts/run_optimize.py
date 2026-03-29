# attention aux degats
# le cerveau de l'operation

import argparse
import json
import logging
import os
import sys
import time
import itertools

# Configure logging BEFORE other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy loggers
for name in [
    "matplotlib", "regime.clustering", "regime.transition",
    "regime.transition_model", "monitoring.alerts",
    "execution.order_router", "risk.calibration",
    "microstructure.pdf.normalizing_flow",
]:
    logging.getLogger(name).setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds")

from config.loader import get_config
from optimization.search_space import get_param_names, get_defaults
from optimization.walk_forward import WalkForwardValidator
from optimization.sensitivity import SensitivityAnalyzer
from optimization.diagnostics import generate_diagnostics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------
DEFAULT_N_CALLS = 30
DEFAULT_N_FOLDS = 5
DEFAULT_TICK_LIMIT = 50_000
DEFAULT_SYMBOL = "SPY"
DEFAULT_DD_TARGET = 0.05
OUTPUT_DIR = "optimization_results"


def load_tick_data(data_dir: str, symbol: str, tick_limit: int = None) -> list:
    # la tuyauterie de donnees
    # on ramene les datas
    dbn_dir = os.path.join(data_dir, "data", "databento_downloads")

    if os.path.isdir(dbn_dir):
        dbn_files = [f for f in os.listdir(dbn_dir) if f.endswith(".dbn.zst")]
        if dbn_files:
            logger.info(f"Loading Databento data from {dbn_dir} ({len(dbn_files)} files)")
            from data.databento_loader import DatabentoLoader
            loader = DatabentoLoader(data_dir=dbn_dir, symbol_override=symbol, trades_only=True)
            stream = loader.load_files()
            if tick_limit:
                ticks = list(itertools.islice(stream, tick_limit))
            else:
                ticks = list(stream)
            logger.info(f"Loaded {len(ticks):,} ticks from Databento")
            return ticks

    # Fallback to synthetic data
    logger.warning("No Databento data found. Using synthetic tick stream.")
    from data.tick_stream import SyntheticTickStream
    stream = SyntheticTickStream(symbol=symbol, duration_seconds=60, seed=42)
    ticks = list(itertools.islice(stream, tick_limit or 10_000))
    logger.info(f"Generated {len(ticks):,} synthetic ticks")
    return ticks


def save_results(output_dir: str, **results):
    # on garde une trace
    os.makedirs(output_dir, exist_ok=True)
    for name, data in results.items():
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"  Saved: {filepath}")


def print_summary(wf_report, sensitivity_report, diagnostics_report):
    # l'usine a gaz
    print("\n" + "=" * 70)
    print("  BAYESIAN OPTIMIZATION — RESULTS SUMMARY")
    print("=" * 70)

    # Walk-Forward
    print(f"\n  {'--- Walk-Forward Validation ---':^66}")
    print(f"  Folds:              {wf_report.n_folds}")
    print(f"  Calls per fold:     {wf_report.n_calls_per_fold}")
    print(f"  Mean OOS Sharpe:    {wf_report.mean_oos_sharpe:>10.4f} ± {wf_report.std_oos_sharpe:.4f}")
    print(f"  Mean Overfit Ratio: {wf_report.mean_overfitting_ratio:>10.2f}")
    print(f"  Total Time:         {wf_report.total_time_sec:>10.1f}s")

    for fold in wf_report.folds:
        print(
            f"    Fold {fold.fold_idx + 1}: "
            f"Train Sharpe={fold.train_sharpe:>7.4f}  "
            f"Test Sharpe={fold.test_sharpe:>7.4f}  "
            f"Trades={fold.test_total_trades:>4d}  "
            f"DD={fold.test_max_drawdown:>6.2%}"
        )

    # Best Parameters
    print(f"\n  {'--- Best Parameters ---':^66}")
    for k, v in wf_report.best_overall_params.items():
        if isinstance(v, float):
            print(f"    {k:35s}: {v:>12.6f}")
        else:
            print(f"    {k:35s}: {v:>12}")

    # Sensitivity
    print(f"\n  {'--- Sensitivity Analysis ---':^66}")
    if sensitivity_report:
        sr = sensitivity_report
        robust = [p["name"] for p in sr["parameters"] if p["classification"] == "robust"]
        sensitive = [p["name"] for p in sr["parameters"] if p["classification"] == "sensitive"]
        unstable = [p["name"] for p in sr["parameters"] if p["classification"] == "unstable"]
        print(f"    Robust ({len(robust)}):    {', '.join(robust[:5]) or 'None'}")
        print(f"    Sensitive ({len(sensitive)}): {', '.join(sensitive[:5]) or 'None'}")
        print(f"    Unstable ({len(unstable)}):  {', '.join(unstable[:5]) or 'None'}")

    # Diagnostics
    print(f"\n  {'--- Risk Diagnostics ---':^66}")
    if diagnostics_report:
        dr = diagnostics_report
        tr = dr.get("tail_risk", {})
        rs = dr.get("regime_stability", {})
        dd = dr.get("drawdown", {})
        of = dr.get("overfit", {})
        print(f"    Sharpe (OOS):      {dr.get('sharpe_ratio', 0):>10.4f}")
        print(f"    Max Drawdown:      {dd.get('max_drawdown', 0):>10.2%}")
        print(f"    Win Rate:          {tr.get('win_rate', 0):>10.2%}")
        print(f"    Profit Factor:     {tr.get('profit_factor', 0):>10.2f}")
        print(f"    Kelly Fraction:    {tr.get('kelly_fraction', 0):>10.4f}")
        print(f"    Regimes Detected:  {rs.get('n_regimes_detected', 0):>10d}")
        print(f"    Overfit Status:    {of.get('assessment', 'N/A'):>10s}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Microstructure-Alpha")
    parser.add_argument("--n-calls", type=int, default=DEFAULT_N_CALLS,
                        help=f"Bayesian optimization iterations per fold (default: {DEFAULT_N_CALLS})")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS,
                        help=f"Walk-forward folds (default: {DEFAULT_N_FOLDS})")
    parser.add_argument("--tick-limit", type=int, default=DEFAULT_TICK_LIMIT,
                        help=f"Max ticks to load (default: {DEFAULT_TICK_LIMIT})")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                        help=f"Instrument symbol (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--dd-target", type=float, default=DEFAULT_DD_TARGET,
                        help=f"Target max drawdown (default: {DEFAULT_DD_TARGET})")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip sensitivity analysis (saves time)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    print("=" * 70)
    print("  MICROSTRUCTURE-ALPHA: BAYESIAN OPTIMIZATION")
    print("=" * 70)
    print(f"  Config: n_calls={args.n_calls}, n_folds={args.n_folds}, "
          f"tick_limit={args.tick_limit:,}, symbol={args.symbol}")
    print()

    # 1. Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ticks = load_tick_data(base_dir, args.symbol, args.tick_limit)

    if len(ticks) < 200:
        logger.error(f"Not enough ticks ({len(ticks)}). Need at least 200.")
        sys.exit(1)

    # 2. Get base config
    config = get_config()

    # Override instrument for SPY
    config.instruments.instruments[0].symbol = args.symbol
    config.instruments.instruments[0].exchange = "XNAS"
    config.instruments.instruments[0].tick_size = 0.01
    config.instruments.instruments[0].lot_size = 1.0
    config.instruments.instruments[0].min_notional = 100.0

    # 3. Walk-Forward Validation
    print("\n[1/3] Running Walk-Forward Validation...")
    wf = WalkForwardValidator(
        base_config=config,
        tick_list=ticks,
        n_folds=args.n_folds,
        n_calls=args.n_calls,
        tick_limit_per_run=None,  # Use all ticks in each fold
        symbol=args.symbol,
        dd_target=args.dd_target,
    )
    wf_report = wf.run()

    # Extract best params (already a dict after Optuna migration)
    best_params = wf_report.best_overall_params

    # 4. Sensitivity Analysis (optional)
    sensitivity_dict = None
    if not args.skip_sensitivity:
        print("\n[2/3] Running Sensitivity Analysis...")
        sa = SensitivityAnalyzer(
            base_config=config,
            tick_list=ticks,
            tick_limit=args.tick_limit,
            symbol=args.symbol,
            dd_target=args.dd_target,
        )
        sensitivity_report = sa.analyze(best_params)
        sensitivity_dict = sensitivity_report.to_dict()
    else:
        print("\n[2/3] Sensitivity Analysis SKIPPED")

    # 5. Diagnostics
    print("\n[3/3] Generating Risk Diagnostics...")
    splits = wf.get_fold_splits()
    last_train_end = splits[-1][0]
    train_ticks = ticks[:last_train_end]
    test_ticks = ticks[last_train_end:]

    if len(test_ticks) < 50:
        # Fallback: use last 20% as test
        split_idx = int(len(ticks) * 0.8)
        train_ticks = ticks[:split_idx]
        test_ticks = ticks[split_idx:]

    diag_report = generate_diagnostics(
        best_params, config, train_ticks, test_ticks,
        tick_limit=None, symbol=args.symbol,
    )
    diagnostics_dict = diag_report.to_dict()

    # 6. Save results
    print(f"\nSaving results to {args.output_dir}/...")
    save_results(
        args.output_dir,
        best_params=wf_report.best_overall_params,
        walk_forward_summary=wf_report.to_dict(),
        sensitivity_report=sensitivity_dict or {},
        diagnostics=diagnostics_dict,
    )

    # 7. Print summary
    print_summary(wf_report, sensitivity_dict, diagnostics_dict)

    return wf_report, sensitivity_dict, diagnostics_dict


if __name__ == "__main__":
    main()
