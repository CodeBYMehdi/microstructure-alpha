# attention aux degats

import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TearsheetGenerator:
    # le cerveau de l'operation

    def __init__(self, output_dir: str = ".", initial_capital: float = 100000.0):
        self.output_dir = output_dir
        self.initial_capital = initial_capital

    def generate(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: Optional[List[float]] = None,
        validation_report: Optional[Any] = None,
        benchmark_results: Optional[Dict] = None,
        attribution_summary: Optional[str] = None,
        predictor_metrics: Optional[Dict] = None,
        signal_weights: Optional[Dict] = None,
        decay_summary: Optional[Dict] = None,
        regime_stationarity: Optional[Dict] = None,
    ) -> str:
        # le bif
        lines = []
        lines.append("=" * 70)
        lines.append("    MICROSTRUCTURE ALPHA — STRATEGY TEARSHEET")
        lines.append(f"    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        if not trades:
            lines.append("  No trades to analyze.")
            return "\n".join(lines)

        pnls = [t.get("pnl", 0.0) for t in trades]

        # ── SECTION 1: SUMMARY ──
        lines.extend(self._section_summary(pnls, trades))

        # ── SECTION 2: RISK METRICS ──
        lines.extend(self._section_risk(pnls, equity_curve))

        # ── SECTION 3: TRADE ANALYSIS ──
        lines.extend(self._section_trades(trades, pnls))

        # ── SECTION 4: STATISTICAL VALIDATION ──
        if validation_report:
            lines.extend(self._section_validation(validation_report))

        # ── SECTION 5: BENCHMARK COMPARISON ──
        if benchmark_results:
            lines.extend(self._section_benchmarks(benchmark_results))

        # ── SECTION 6: ATTRIBUTION ──
        if attribution_summary:
            lines.append("\n" + "─" * 50)
            lines.append("  SECTION 6: PnL ATTRIBUTION")
            lines.append("─" * 50)
            lines.append(attribution_summary)

        # ── SECTION 7: ALPHA MODEL ──
        lines.extend(self._section_alpha_model(predictor_metrics, signal_weights, decay_summary))

        # ── SECTION 8: REGIME STATIONARITY ──
        if regime_stationarity:
            lines.extend(self._section_stationarity(regime_stationarity))

        lines.append("\n" + "=" * 70)
        lines.append("    END OF TEARSHEET")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # Save to file
        filepath = os.path.join(self.output_dir, "strategy_tearsheet.txt")
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            logger.info(f"Tearsheet saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save tearsheet: {e}")

        return report

    def _section_summary(self, pnls, trades) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 1: PERFORMANCE SUMMARY", "─" * 50]
        pnl_array = np.array(pnls)

        total_pnl = float(np.sum(pnl_array))
        total_return_pct = total_pnl / self.initial_capital * 100

        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]

        lines.append(f"  Total PnL:           {total_pnl:>+12.2f}")
        lines.append(f"  Total Return:        {total_return_pct:>+11.2f}%")
        lines.append(f"  # Trades:            {len(pnls):>12}")
        lines.append(f"  Win Rate:            {len(wins)/len(pnls)*100:>11.1f}%")

        if len(losses) > 0:
            pf = abs(np.sum(wins)) / abs(np.sum(losses))
            lines.append(f"  Profit Factor:       {pf:>12.2f}")
        else:
            lines.append(f"  Profit Factor:          ∞")

        lines.append(f"  Avg Win:             {float(np.mean(wins)):>+12.4f}" if len(wins) else "")
        lines.append(f"  Avg Loss:            {float(np.mean(losses)):>+12.4f}" if len(losses) else "")

        # Expectancy
        expectancy = float(np.mean(pnl_array))
        lines.append(f"  Expectancy:          {expectancy:>+12.4f}")

        return lines

    def _section_risk(self, pnls, equity_curve) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 2: RISK-ADJUSTED METRICS", "─" * 50]
        pnl_array = np.array(pnls)

        if len(pnl_array) < 2:
            lines.append("  Insufficient trades for risk metrics")
            return lines

        mean_pnl = np.mean(pnl_array)
        std_pnl = np.std(pnl_array)

        # Sharpe
        sharpe = float(mean_pnl / std_pnl) if std_pnl > 1e-10 else 0.0
        lines.append(f"  Sharpe Ratio:        {sharpe:>12.3f}")

        # Sortino (downside deviation)
        downside = pnl_array[pnl_array < 0]
        down_std = float(np.std(downside)) if len(downside) > 1 else std_pnl
        sortino = float(mean_pnl / down_std) if down_std > 1e-10 else 0.0
        lines.append(f"  Sortino Ratio:       {sortino:>12.3f}")

        # Omega (ratio of gains integral to losses integral above threshold 0)
        gains = np.sum(pnl_array[pnl_array > 0])
        loss = abs(np.sum(pnl_array[pnl_array < 0]))
        omega = float(gains / loss) if loss > 0 else float('inf')
        lines.append(f"  Omega Ratio:         {omega:>12.3f}")

        # Max Drawdown
        if equity_curve and len(equity_curve) > 1:
            eq = np.array(equity_curve)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.maximum(peak, 1e-10)
            max_dd = float(np.max(dd))
        else:
            cum_pnl = np.cumsum(pnl_array) + self.initial_capital
            peak = np.maximum.accumulate(cum_pnl)
            dd = (peak - cum_pnl) / np.maximum(peak, 1e-10)
            max_dd = float(np.max(dd))

        lines.append(f"  Max Drawdown:        {max_dd*100:>11.2f}%")

        # Calmar
        total_return = np.sum(pnl_array) / self.initial_capital
        calmar = float(total_return / max_dd) if max_dd > 1e-10 else 0.0
        lines.append(f"  Calmar Ratio:        {calmar:>12.3f}")

        # PnL Skewness
        from scipy.stats import skew, kurtosis
        pnl_skew = float(skew(pnl_array))
        pnl_kurt = float(kurtosis(pnl_array))
        lines.append(f"  PnL Skewness:        {pnl_skew:>+12.3f}")
        lines.append(f"  PnL Kurtosis:        {pnl_kurt:>+12.3f}")

        return lines

    def _section_trades(self, trades, pnls) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 3: TRADE ANALYSIS", "─" * 50]

        pnl_array = np.array(pnls)

        # Consecutive wins/losses
        signs = np.sign(pnl_array)
        max_consec_wins = self._max_consecutive(signs, 1)
        max_consec_losses = self._max_consecutive(signs, -1)
        lines.append(f"  Max Consecutive Wins:  {max_consec_wins:>10}")
        lines.append(f"  Max Consecutive Losses:{max_consec_losses:>10}")

        # Holding period stats
        holds = [t.get("hold_duration_windows", 0) for t in trades]
        if holds:
            lines.append(f"  Avg Hold (windows):    {np.mean(holds):>10.1f}")
            lines.append(f"  Max Hold (windows):    {max(holds):>10}")

        # MAE / MFE
        maes = [t.get("worst_unrealized_pct", 0.0) for t in trades if "worst_unrealized_pct" in t]
        mfes = [t.get("best_unrealized_pct", 0.0) for t in trades if "best_unrealized_pct" in t]
        if maes:
            lines.append(f"  Avg MAE:               {np.mean(maes)*100:>+9.3f}%")
        if mfes:
            lines.append(f"  Avg MFE:               {np.mean(mfes)*100:>+9.3f}%")

        # Top 5 best / worst trades
        sorted_pnls = sorted(enumerate(pnls), key=lambda x: x[1])
        lines.append("\n  Top 5 Worst Trades:")
        for i, (idx, pnl) in enumerate(sorted_pnls[:5]):
            lines.append(f"    #{idx}: PnL={pnl:+.4f}")
        lines.append("\n  Top 5 Best Trades:")
        for i, (idx, pnl) in enumerate(sorted_pnls[-5:][::-1]):
            lines.append(f"    #{idx}: PnL={pnl:+.4f}")

        return lines

    def _section_validation(self, report) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 4: STATISTICAL VALIDATION", "─" * 50]

        verdict = "✅ SIGNIFICANT" if report.is_statistically_significant else "❌ NOT SIGNIFICANT"
        lines.append(f"  Overall: {verdict}")

        if hasattr(report, 'bootstrap_ci'):
            ci = report.bootstrap_ci
            lines.append(f"\n  Bootstrap ({ci.confidence_level:.0%} CI):")
            lines.append(f"    Point Estimate: {ci.point_estimate:+.4f}")
            lines.append(f"    CI: [{ci.ci_lower:+.4f}, {ci.ci_upper:+.4f}]")
            lines.append(f"    Significant: {'Yes' if ci.is_significant else 'No'}")

        if hasattr(report, 'permutation_test'):
            pt = report.permutation_test
            lines.append(f"\n  Permutation Test:")
            lines.append(f"    p-value: {pt.p_value:.4f}")
            lines.append(f"    Significant at α={pt.significance_level}: {'Yes' if pt.is_significant else 'No'}")

        if hasattr(report, 'cost_sensitivity') and report.cost_sensitivity:
            lines.append(f"\n  Transaction Cost Sensitivity:")
            for label, value in report.cost_sensitivity.items():
                status = "✅" if value > 0 else "❌"
                lines.append(f"    {label}: {value:+.2f} {status}")

        return lines

    def _section_benchmarks(self, results) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 5: BENCHMARK COMPARISON", "─" * 50]
        for name, r in results.items():
            verdict = "✅" if r.excess_return > 0 else "❌"
            lines.append(f"  vs {r.benchmark_name}: {verdict} Excess={r.excess_return:+.2f} IR={r.information_ratio:.3f}")
        return lines

    def _section_alpha_model(self, predictor, signals, decay) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 7: ALPHA MODEL", "─" * 50]

        if predictor:
            lines.append(f"\n  Return Predictor:")
            lines.append(f"    Updates: {predictor.get('n_updates', 0)}")
            lines.append(f"    MAE: {predictor.get('mae', 0):.6f}")
            lines.append(f"    Direction Accuracy: {predictor.get('directional_accuracy', 0):.1%}")
            lines.append(f"    IC: {predictor.get('ic', 0):.4f}")
            lines.append(f"    R²: {predictor.get('r_squared', 0):.4f}")

        if signals:
            lines.append(f"\n  Signal Weights:")
            for name, weight in sorted(signals.items(), key=lambda x: -x[1]):
                lines.append(f"    {name}: {weight:.3f}")

        if decay:
            lines.append(f"\n  Alpha Decay Profiles:")
            for trans, info in decay.items():
                hl = info.get('half_life')
                if hl:
                    lines.append(f"    {trans}: half-life={hl:.1f}w, conf={info.get('confidence', 0):.2f}")

        return lines

    def _section_stationarity(self, results) -> List[str]:
        lines = ["\n" + "─" * 50, "  SECTION 8: REGIME STATIONARITY", "─" * 50]
        for regime_id, tests in results.items():
            conclusion = tests.get("joint_conclusion", "Unknown")
            lines.append(f"  Regime {regime_id}: {conclusion}")
        return lines

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
