"""Parameter space for walk-forward optimization (Optuna)."""

import copy
from typing import Any, Dict, List, Tuple

import optuna

from config.schema import AppConfig


# Parameter definitions: (section, name, type, low, high, kwargs)
_PARAM_DEFS: List[Tuple[str, str, str, Any, Any, dict]] = [
    # Regime Detection
    ("regime", "min_cluster_size",         "int",   5,     50,    {}),
    ("regime", "min_samples",              "int",   2,     15,    {}),
    ("regime", "window_size",              "int",   50,    500,   {}),
    ("regime", "update_frequency",         "int",   10,    100,   {}),
    ("regime", "transition_strength_min",  "float", 0.05,  0.5,   {}),
    ("regime", "kl_boost_threshold",       "float", 0.1,   1.0,   {}),
    ("regime", "kl_boost_amount",          "float", 0.05,  0.5,   {}),
    ("regime", "persistence_window",       "int",   5,     50,    {}),
    # Decision Thresholds
    ("decision", "long_skew_min",          "float", 0.001, 0.2,   {}),
    ("decision", "long_tail_slope_min",    "float", 0.1,   2.0,   {}),
    ("decision", "short_volatility_min",   "float", 1e-6,  0.001, {"log": True}),
    ("decision", "short_skew_max",         "float", -0.5,  -0.001, {}),
    ("decision", "short_kurtosis_min",     "float", 0.05,  2.0,   {}),
    # Sizing
    ("sizing", "base_size",               "float", 0.1,   5.0,   {}),
    ("sizing", "max_size_multiplier",     "float", 1.0,   10.0,  {}),
    # Risk
    ("risk", "max_drawdown",              "float", 0.01,  0.10,  {}),
    ("risk", "confidence_floor",          "float", 0.0,   0.5,   {}),
    ("risk", "regime_churn_limit",        "int",   2,     20,    {}),
    # Calibration
    ("calibration", "entropy_reduce_threshold",  "float", 1.0, 8.0,   {}),
    ("calibration", "volatility_high_threshold", "float", 0.001, 0.05, {"log": True}),
    ("calibration", "tail_slope_tighten",        "float", 1.0, 5.0,   {}),
    ("calibration", "kurtosis_tighten",          "float", 1.0, 8.0,   {}),
]


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Map 22-parameter space to Optuna trial suggestions."""
    params = {}
    for _section, name, ptype, low, high, kwargs in _PARAM_DEFS:
        if ptype == "int":
            params[name] = trial.suggest_int(name, low, high)
        elif ptype == "float":
            params[name] = trial.suggest_float(name, low, high, **kwargs)
    return params


def get_param_names() -> List[str]:
    return [name for _, name, *_ in _PARAM_DEFS]


def get_param_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for group, name, *_ in _PARAM_DEFS:
        groups.setdefault(group, []).append(name)
    return groups


def get_param_bounds() -> Dict[str, Tuple[str, Any, Any]]:
    """Return {name: (type, low, high)} for each parameter.

    Used by SensitivityAnalyzer to clamp perturbed values.
    """
    return {name: (ptype, low, high) for _, name, ptype, low, high, _ in _PARAM_DEFS}


def get_defaults() -> Dict[str, Any]:
    """Return default parameter values as a dict."""
    return {
        "min_cluster_size": 10, "min_samples": 3, "window_size": 200,
        "update_frequency": 50, "transition_strength_min": 0.2,
        "kl_boost_threshold": 0.5, "kl_boost_amount": 0.2,
        "persistence_window": 20,
        "long_skew_min": 0.01, "long_tail_slope_min": 0.5,
        "short_volatility_min": 0.00001, "short_skew_max": -0.01,
        "short_kurtosis_min": 0.1,
        "base_size": 1.0, "max_size_multiplier": 3.0,
        "max_drawdown": 0.02, "confidence_floor": 0.0,
        "regime_churn_limit": 5,
        "entropy_reduce_threshold": 4.0, "volatility_high_threshold": 0.01,
        "tail_slope_tighten": 2.0, "kurtosis_tighten": 3.0,
    }


def apply_params(config: AppConfig, params: Dict[str, Any]) -> AppConfig:
    """Apply parameter dict to config.thresholds. Returns deepcopy."""
    cfg = copy.deepcopy(config)

    # Regime
    cfg.thresholds.regime.min_cluster_size = int(params["min_cluster_size"])
    cfg.thresholds.regime.min_samples = int(params["min_samples"])
    cfg.thresholds.regime.window_size = int(params["window_size"])
    cfg.thresholds.regime.update_frequency = int(params["update_frequency"])
    cfg.thresholds.regime.transition_strength_min = float(params["transition_strength_min"])
    cfg.thresholds.regime.kl_boost_threshold = float(params["kl_boost_threshold"])
    cfg.thresholds.regime.kl_boost_amount = float(params["kl_boost_amount"])
    cfg.thresholds.regime.persistence_window = int(params["persistence_window"])

    # Decision
    cfg.thresholds.decision.long.skew_min = float(params["long_skew_min"])
    cfg.thresholds.decision.long.tail_slope_min = float(params["long_tail_slope_min"])
    cfg.thresholds.decision.short.volatility_min = float(params["short_volatility_min"])
    cfg.thresholds.decision.short.skew_max = float(params["short_skew_max"])
    cfg.thresholds.decision.short.kurtosis_min = float(params["short_kurtosis_min"])

    # Sizing
    cfg.thresholds.decision.sizing.base_size = float(params["base_size"])
    cfg.thresholds.decision.sizing.max_size_multiplier = float(params["max_size_multiplier"])

    # Risk
    cfg.thresholds.risk.max_drawdown = float(params["max_drawdown"])
    cfg.thresholds.risk.confidence_floor = float(params["confidence_floor"])
    cfg.thresholds.risk.regime_churn_limit = int(params["regime_churn_limit"])

    # Calibration
    cfg.thresholds.calibration.entropy_reduce_threshold = float(params["entropy_reduce_threshold"])
    cfg.thresholds.calibration.volatility_high_threshold = float(params["volatility_high_threshold"])
    cfg.thresholds.calibration.tail_slope_tighten = float(params["tail_slope_tighten"])
    cfg.thresholds.calibration.kurtosis_tighten = float(params["kurtosis_tighten"])

    return cfg
