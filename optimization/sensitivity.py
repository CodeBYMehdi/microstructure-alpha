# le cerveau de l'operation

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from config.schema import AppConfig
from optimization.search_space import (
    get_param_names,
    get_param_bounds,
    get_param_groups,
    apply_params,
)
from optimization.objective import objective

logger = logging.getLogger(__name__)


@dataclass
class ParameterSensitivity:
    name: str
    group: str
    optimal_value: Any
    perturbations: List[Dict[str, Any]]  # [{pct, value, score, delta_score}]
    sensitivity_score: float             # |Δscore / Δparam| normalized
    classification: str                  # "robust" / "sensitive" / "unstable"


@dataclass
class SensitivityReport:
    n_parameters: int
    n_perturbations_per_param: int
    base_score: float
    parameters: List[ParameterSensitivity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_robust_params(self) -> List[str]:
        return [p.name for p in self.parameters if p.classification == "robust"]

    def get_sensitive_params(self) -> List[str]:
        return [p.name for p in self.parameters if p.classification == "sensitive"]

    def get_unstable_params(self) -> List[str]:
        return [p.name for p in self.parameters if p.classification == "unstable"]


class SensitivityAnalyzer:

    def __init__(
        self,
        base_config: AppConfig,
        tick_list: List[Any],
        perturbation_pcts: List[float] = None,
        tick_limit: Optional[int] = None,
        symbol: str = "SPY",
        dd_target: float = 0.05,
    ):
        self.base_config = base_config
        self.tick_list = tick_list
        self.perturbation_pcts = perturbation_pcts or [-0.20, -0.10, 0.10, 0.20]
        self.tick_limit = tick_limit
        self.symbol = symbol
        self.dd_target = dd_target

    def analyze(self, optimal_params: Dict[str, Any]) -> SensitivityReport:
        names = get_param_names()
        bounds = get_param_bounds()
        param_groups = get_param_groups()
        # Invert to name -> group
        name_to_group = {}
        for group, group_names in param_groups.items():
            for n in group_names:
                name_to_group[n] = group

        # Compute base score
        base_score = objective(
            optimal_params, self.base_config, self.tick_list,
            self.tick_limit, self.symbol, self.dd_target,
        )
        logger.info(f"Base score at optimum: {base_score:.4f}")

        report = SensitivityReport(
            n_parameters=len(names),
            n_perturbations_per_param=len(self.perturbation_pcts),
            base_score=base_score,
        )

        for name in names:
            ptype, low, high = bounds[name]
            optimal_val = optimal_params[name]
            perturbations = []
            delta_scores = []

            for pct in self.perturbation_pcts:
                perturbed_val = self._perturb_value(optimal_val, pct, ptype, low, high)

                perturbed_params = dict(optimal_params)
                perturbed_params[name] = perturbed_val

                score = objective(
                    perturbed_params, self.base_config, self.tick_list,
                    self.tick_limit, self.symbol, self.dd_target,
                )

                delta_score = score - base_score
                delta_scores.append(abs(delta_score))

                perturbations.append({
                    "pct": pct,
                    "value": perturbed_val,
                    "score": score,
                    "delta_score": delta_score,
                })

            sensitivity = float(np.mean(delta_scores)) if delta_scores else 0.0

            if sensitivity < 0.1:
                classification = "robust"
            elif sensitivity < 0.5:
                classification = "sensitive"
            else:
                classification = "unstable"

            param_sensitivity = ParameterSensitivity(
                name=name,
                group=name_to_group.get(name, "unknown"),
                optimal_value=optimal_val,
                perturbations=perturbations,
                sensitivity_score=sensitivity,
                classification=classification,
            )
            report.parameters.append(param_sensitivity)

            logger.info(
                f"  {name}: sensitivity={sensitivity:.4f} [{classification}]"
            )

        report.parameters.sort(key=lambda p: p.sensitivity_score, reverse=True)

        logger.info(
            f"\nSensitivity Analysis Complete:\n"
            f"  Robust: {len(report.get_robust_params())} params\n"
            f"  Sensitive: {len(report.get_sensitive_params())} params\n"
            f"  Unstable: {len(report.get_unstable_params())} params"
        )

        return report

    @staticmethod
    def _perturb_value(value: Any, pct: float, ptype: str, low: Any, high: Any) -> Any:
        if ptype == "int" or isinstance(value, int):
            perturbed = int(round(value * (1 + pct)))
            perturbed = max(low, min(high, perturbed))
            if perturbed == value and pct > 0:
                perturbed = min(value + 1, high)
            elif perturbed == value and pct < 0:
                perturbed = max(value - 1, low)
            return perturbed
        else:
            perturbed = float(value * (1 + pct))
            return max(low, min(high, perturbed))
