"""Transaction cost sensitivity analysis.

Answers the critical question: "Does my strategy survive realistic costs?"

Runs the same backtest at multiple cost assumptions and produces a
cost frontier showing where profitability breaks down.

Usage:
    analyzer = CostSensitivityAnalyzer(base_config, tick_data, symbol="SPY")
    report = analyzer.run(cost_bps=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    
    print(f"Break-even cost: {report.break_even_bps:.1f} bps")
"""

import numpy as np
import logging
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class CostScenario:
    """Result of a single cost scenario."""
    cost_bps: float
    total_pnl: float
    net_pnl: float           # PnL after costs
    total_costs: float       # Total costs at this level
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    cost_as_pct_of_gross_pnl: float  # How much costs eat into gross PnL


@dataclass
class CostSensitivityReport:
    """Full cost sensitivity analysis report."""
    scenarios: List[CostScenario] = field(default_factory=list)
    break_even_bps: float = 0.0         # Cost level where PnL crosses zero
    profitable_at_3bps: bool = False     # The minimum institutional bar
    profitable_at_5bps: bool = False     # Conservative bar
    gross_pnl: float = 0.0              # PnL with zero costs
    cost_elasticity: float = 0.0        # d(PnL) / d(cost_bps)

    def to_dict(self) -> Dict:
        return {
            'break_even_bps': self.break_even_bps,
            'profitable_at_3bps': self.profitable_at_3bps,
            'profitable_at_5bps': self.profitable_at_5bps,
            'gross_pnl': self.gross_pnl,
            'cost_elasticity': self.cost_elasticity,
            'scenarios': [
                {
                    'cost_bps': s.cost_bps,
                    'net_pnl': s.net_pnl,
                    'sharpe': s.sharpe_ratio,
                    'trades': s.total_trades,
                    'win_rate': s.win_rate,
                    'max_drawdown': s.max_drawdown,
                }
                for s in self.scenarios
            ],
        }


class CostSensitivityAnalyzer:
    """Runs backtest at multiple cost levels to find break-even point."""

    def __init__(
        self,
        base_config,         # AppConfig
        tick_list: List,     # List of ticks for backtesting
        symbol: str = "SPY",
        tick_limit: Optional[int] = None,
    ):
        self.base_config = base_config
        self.tick_list = tick_list
        self.symbol = symbol
        self.tick_limit = tick_limit

    def run(
        self,
        cost_bps: Optional[List[float]] = None,
    ) -> CostSensitivityReport:
        """Run backtest at each cost level.
        
        Args:
            cost_bps: List of cost levels (in basis points) to test.
                      Default: [0, 0.5, 1, 2, 3, 5, 7, 10]
        """
        if cost_bps is None:
            cost_bps = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        scenarios: List[CostScenario] = []
        gross_pnl = 0.0

        for bps in sorted(cost_bps):
            logger.info(f"Running cost scenario: {bps} bps...")
            try:
                result = self._run_backtest_at_cost(bps)
                scenario = CostScenario(
                    cost_bps=bps,
                    total_pnl=result.get('total_pnl', 0.0),
                    net_pnl=result.get('total_pnl', 0.0),  # Already includes costs
                    total_costs=result.get('total_fees', 0.0),
                    sharpe_ratio=result.get('sharpe_ratio', 0.0),
                    total_trades=result.get('total_trades', 0),
                    win_rate=result.get('win_rate', 0.0),
                    profit_factor=result.get('profit_factor', 0.0),
                    max_drawdown=abs(result.get('max_drawdown', 0.0)),
                    cost_as_pct_of_gross_pnl=0.0,
                )
                scenarios.append(scenario)

                if bps == 0:
                    gross_pnl = scenario.total_pnl

                logger.info(
                    f"  {bps} bps -> PnL=${scenario.net_pnl:.2f}, "
                    f"Sharpe={scenario.sharpe_ratio:.3f}, "
                    f"Trades={scenario.total_trades}"
                )

            except Exception as e:
                logger.error(f"Cost scenario {bps} bps failed: {e}")

        # Fill in cost percentages
        for s in scenarios:
            if gross_pnl > 0:
                s.cost_as_pct_of_gross_pnl = 1.0 - (s.net_pnl / gross_pnl)

        # Find break-even
        break_even = self._find_break_even(scenarios)

        # Check profitability thresholds
        profitable_3 = any(s.net_pnl > 0 for s in scenarios if s.cost_bps == 3.0)
        profitable_5 = any(s.net_pnl > 0 for s in scenarios if s.cost_bps == 5.0)

        # Cost elasticity (linear slope of PnL vs cost)
        if len(scenarios) >= 2:
            costs = [s.cost_bps for s in scenarios]
            pnls = [s.net_pnl for s in scenarios]
            elasticity = float(np.polyfit(costs, pnls, 1)[0])
        else:
            elasticity = 0.0

        report = CostSensitivityReport(
            scenarios=scenarios,
            break_even_bps=break_even,
            profitable_at_3bps=profitable_3,
            profitable_at_5bps=profitable_5,
            gross_pnl=gross_pnl,
            cost_elasticity=elasticity,
        )

        self._print_report(report)
        return report

    def _run_backtest_at_cost(self, cost_bps: float) -> Dict:
        """Run a single backtest with modified cost parameters."""
        from optimization.objective import run_single_backtest

        config = copy.deepcopy(self.base_config)

        # Override execution simulation costs
        if hasattr(config, 'thresholds') and hasattr(config.thresholds, 'execution_sim'):
            config.thresholds.execution_sim.base_fee_bps = cost_bps
            config.thresholds.execution_sim.slippage_std_bps = cost_bps * 0.3  # Slippage scales with costs

        return run_single_backtest(
            config, self.tick_list, self.tick_limit, self.symbol,
        )

    @staticmethod
    def _find_break_even(scenarios: List[CostScenario]) -> float:
        """Find the cost level where PnL crosses zero via interpolation."""
        if not scenarios:
            return 0.0

        # Sort by cost
        sorted_scenarios = sorted(scenarios, key=lambda s: s.cost_bps)

        # Find the crossing point
        for i in range(len(sorted_scenarios) - 1):
            s1 = sorted_scenarios[i]
            s2 = sorted_scenarios[i + 1]

            if s1.net_pnl >= 0 and s2.net_pnl < 0:
                # Linear interpolation
                if s1.net_pnl != s2.net_pnl:
                    frac = s1.net_pnl / (s1.net_pnl - s2.net_pnl)
                    break_even = s1.cost_bps + frac * (s2.cost_bps - s1.cost_bps)
                    return float(break_even)

        # All profitable or all unprofitable
        if all(s.net_pnl > 0 for s in sorted_scenarios):
            return float(sorted_scenarios[-1].cost_bps) * 1.5  # Extrapolate
        return 0.0

    @staticmethod
    def _print_report(report: CostSensitivityReport) -> None:
        print("\n" + "=" * 60)
        print("  TRANSACTION COST SENSITIVITY ANALYSIS")
        print("=" * 60)
        print(f"{'Cost (bps)':>12} {'Net PnL':>12} {'Sharpe':>10} {'Win Rate':>10} {'Max DD':>10}")
        print("-" * 60)
        for s in report.scenarios:
            marker = " ⚠" if s.net_pnl < 0 else " ✓"
            print(
                f"{s.cost_bps:>10.1f}   ${s.net_pnl:>10.2f} "
                f"{s.sharpe_ratio:>9.3f} {s.win_rate:>9.1%} "
                f"{s.max_drawdown:>9.2%}{marker}"
            )
        print("-" * 60)
        print(f"  Break-even cost:      {report.break_even_bps:.1f} bps")
        print(f"  Profitable at 3 bps:  {'YES ✓' if report.profitable_at_3bps else 'NO ✗'}")
        print(f"  Profitable at 5 bps:  {'YES ✓' if report.profitable_at_5bps else 'NO ✗'}")
        print(f"  Cost elasticity:      ${report.cost_elasticity:.2f} per bps")
        print(f"  Gross PnL (0 cost):   ${report.gross_pnl:.2f}")
        print("=" * 60 + "\n")
