from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, confloat, validator

# --- Execution & Routing ---

class RouterConfig(BaseModel):
    max_retries: PositiveInt = Field(..., description="Max order routing retries")
    timeout_ms: PositiveInt = Field(..., description="Order routing timeout in ms")

class StrategyConfig(BaseModel):
    style: str
    order_type: Literal["limit", "market"]
    urgency: Literal["high", "low", "medium"] = "medium"
    duration_ms: Optional[PositiveInt] = None

class ExecutionConfig(BaseModel):
    mode: Literal["simulation", "live", "replay", "paper"]
    data_source: Literal["synthetic", "hdf5", "csv", "ibkr"] = "synthetic"
    data_path: Optional[str] = None
    router: RouterConfig
    strategies: Dict[str, StrategyConfig]

# --- Instruments ---

class InstrumentConfig(BaseModel):
    symbol: str
    exchange: str
    tick_size: PositiveFloat
    lot_size: PositiveFloat
    min_notional: PositiveFloat

class InstrumentsConfig(BaseModel):
    instruments: List[InstrumentConfig]

# --- HMM Regime Detection ---

class HMMConfig(BaseModel):
    n_states: PositiveInt = Field(default=4, description="Number of hidden regime states")
    learning_rate: float = Field(default=0.01, ge=0.001, le=0.1, description="Online EM learning rate")
    warmup_ticks: PositiveInt = Field(default=200, description="Ticks before HMM predictions are trusted")
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0, description="Min posterior probability to accept regime")
    prior_strength: float = Field(default=1.0, ge=0.1, le=10.0, description="Dirichlet prior strength for transition matrix")
    emission_reg: float = Field(default=1e-4, ge=1e-8, le=1e-1, description="Regularization for emission covariance diagonal")
    sticky_prior_multiplier: float = Field(default=4.0, ge=1.0, le=20.0, description="Self-transition prior boost (regimes persist)")
    kalman_vel_noise_mult: float = Field(default=10.0, ge=1.0, le=100.0, description="Kalman velocity process noise multiplier")
    kalman_acc_noise_mult: float = Field(default=100.0, ge=1.0, le=1000.0, description="Kalman acceleration process noise multiplier")

# --- Regime Detection ---

class RegimeConfig(BaseModel):
    transition_strength_min: float = Field(..., ge=0.0, le=1.0)
    stability_window: PositiveInt
    min_cluster_size: PositiveInt
    min_samples: PositiveInt
    window_size: PositiveInt
    update_frequency: PositiveInt
    # Weights for transition strength calculation: mu, sigma, skew, kurt, tail, entropy
    transition_weights: List[float] = [1.0, 5.0, 1.0, 0.5, 0.5, 2.0]
    kl_min: float = 0.05
    projection_min: float = 1.0
    # KL divergence boost threshold for transition strength
    kl_boost_threshold: float = Field(default=0.5, ge=0.0, description="KL threshold above which transition strength is boosted")
    kl_boost_amount: float = Field(default=0.2, ge=0.0, le=1.0, description="Amount to boost strength when KL exceeds threshold")
    # Persistence window for clustering confidence
    persistence_window: PositiveInt = Field(default=20, description="Window size for persistence estimation")
    # Max history for clustering
    max_history: PositiveInt = Field(default=2000, description="Max state vectors kept for clustering")
    # HMM config
    hmm: HMMConfig = Field(default_factory=HMMConfig, description="HMM regime detection config")

    @validator('transition_weights')
    def weights_must_have_six_elements(cls, v):
        if len(v) != 6:
            raise ValueError('transition_weights must have exactly 6 elements')
        return v

# --- Risk Management ---

class RiskConfig(BaseModel):
    max_position_size: PositiveFloat
    max_drawdown: float = Field(..., ge=0.0, le=1.0)
    tail_risk_limit: float = Field(..., ge=0.0, le=1.0)
    confidence_floor: float = Field(..., ge=0.0, le=1.0)
    regime_churn_limit: PositiveInt
    slippage_tolerance: float = Field(..., ge=0.0)
    live_slippage_tolerance: float = Field(default=0.001, ge=0.0, description="Tight slippage tolerance for live trading (1 bps)")
    var_position_limit_pct: float = Field(default=0.02, ge=0.001, le=0.10,
        description="Max position size scaled by tail risk — heavier tails = smaller size")
    latency_budget_ms: PositiveFloat
    max_error_rate: float = Field(..., ge=0.0, le=1.0)
    # Max consecutive errors before kill switch
    max_consecutive_errors: PositiveInt = Field(default=10, description="Max errors before triggering kill switch")
    # Confidence EMA parameters
    confidence_ema_alpha: float = Field(default=0.1, ge=0.0, le=1.0, description="EMA smoothing for confidence")
    confidence_warmup: PositiveInt = Field(default=20, description="Min samples before confidence EMA is active")

# --- PDF Model ---

class PDFConfig(BaseModel):
    entropy_threshold: float
    tail_slope_min: float
    # Grid resolution for PDF evaluation
    grid_points: PositiveInt = Field(default=1000, description="Number of grid points for PDF evaluation")
    # Sigma range for PDF grid
    sigma_range: PositiveFloat = Field(default=4.0, description="Number of std deviations for PDF grid range")
    # Min data points for density model fit
    min_data_points: PositiveInt = Field(default=20, description="Minimum data points for density model")
    # Max GMM components
    max_gmm_components: PositiveInt = Field(default=4, description="Maximum Gaussian mixture components")
    # Mode collapse threshold
    mode_collapse_threshold: PositiveFloat = Field(default=1000.0, description="PDF peak threshold for mode collapse")
    # Min data for moments calculation
    min_moments_data: PositiveInt = Field(default=10, description="Minimum data points for moments computation")

# --- Decision Logic ---

class LongEntryConfig(BaseModel):
    skew_min: float
    tail_slope_min: float

class ShortEntryConfig(BaseModel):
    volatility_min: float
    skew_max: float
    kurtosis_min: float

class LiquidityConfig(BaseModel):
    spread_max: float
    depth_slope_min: float

class ExitConfig(BaseModel):
    # les petits reglages
    kl_stable_threshold: float = Field(default=0.01, ge=0.0, description="KL divergence threshold below which edge is considered decayed")
    entropy_acceleration_threshold: float = Field(default=0.5, ge=0.0, description="Threshold for entropy acceleration in entry conditions")
    fallback_strength_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum transition strength required for the relaxed fallback entry path. "
            "When strict second-order conditions are not met, a transition whose "
            "strength exceeds this threshold will still generate a trade signal "
            "based on mu_velocity direction alone."
        ),
    )
    # Adaptive Exit Dynamic Parameters
    stop_sigma_mult: float = Field(default=2.0, ge=0.0, description="Stop-loss at N sigma (widened from 1.0)")
    tp_sigma_mult: float = Field(default=2.5, ge=0.0, description="Take-profit at N sigma")
    trail_sigma_mult: float = Field(default=1.5, ge=0.0, description="Trailing distance at N sigma (widened from 0.8)")
    kurtosis_stop_scale: float = Field(default=0.05, ge=0.0, description="Each unit of excess kurtosis widens stop by this factor")
    kurtosis_trail_scale: float = Field(default=0.05, ge=0.0, description="Each unit of excess kurtosis widens trailing distance")
    skew_tp_scale: float = Field(default=0.2, ge=0.0, description="Each unit of skew shifts TP")
    strength_tp_bonus: float = Field(default=0.5, ge=0.0, description="Strong transitions extend TP by up to this factor")
    max_hold_base: int = Field(default=80, ge=1, description="Max hold base windows (increased from 60)")
    entropy_hold_scale: float = Field(default=2.0, ge=0.0, description="Higher entropy -> fewer windows")
    min_stop_pct: float = Field(default=0.0015, ge=0.0, description="Absolute minimum stop loss percentage (widened from 0.03%)")
    min_tp_pct: float = Field(default=0.0030, ge=0.0, description="Absolute minimum take profit percentage (widened from 0.06%)")
    # ATR-based modeling parameters
    atr_period: int = Field(default=14, ge=2, description="Number of micro-bars for ATR EMA computation")
    atr_micro_window: int = Field(default=10, ge=2, description="Number of ticks per micro-bar for ATR (H/L/C approximation)")

class ConfidenceConfig(BaseModel):
    """Confidence scorer magic numbers."""
    dynamics_alignment_high: float = Field(default=0.8, ge=0.0, le=1.0, description="Alignment score when velocity and acceleration agree")
    dynamics_alignment_low: float = Field(default=0.2, ge=0.0, le=1.0, description="Alignment score when velocity and acceleration disagree")
    entropy_stable_boost: float = Field(default=0.2, ge=0.0, le=1.0, description="Boost when entropy acceleration is low")
    spread_normalizer: float = Field(default=100.0, ge=1.0, le=10000.0, description="Spread-to-score normalizer (spread * N)")

class MicrostructureConfig(BaseModel):
    """L2 orderbook microstructure thresholds."""
    liq_pull_threshold: float = Field(default=-0.15, le=0.0, description="Liquidity pull detection threshold (% drop)")
    spoofing_size_mult: float = Field(default=5.0, ge=2.0, le=50.0, description="Size multiple vs median for spoofing detection")

class SizingConfig(BaseModel):
    kelly_scale: float = Field(default=0.5, ge=0.1, le=1.0,
        description="Fraction of full Kelly to use (0.5 = half-Kelly, industry standard)")
    base_size: PositiveFloat = Field(default=1.0, description="Fallback base position size when equity unavailable")
    max_size_multiplier: PositiveFloat = Field(default=3.0, description="Maximum size as multiple of base")
    min_variance: PositiveFloat = Field(default=0.5, description="Minimum variance floor for sizing")
    dwell_time_norm: PositiveFloat = Field(default=50.0, description="Normalization factor for dwell time")
    tail_penalty_floor: float = Field(default=0.2, ge=0.0, le=1.0, description="Floor for tail penalty in conviction calc")

class SignalGateConfig(BaseModel):
    """Signal quality gates — every entry must pass all gates."""
    cost_edge_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="Predicted move must exceed cost by this multiple")
    min_rolling_ic: float = Field(default=0.02, ge=0.0, le=1.0, description="Min rolling IC over last 20 observations")
    min_snr: float = Field(default=0.5, ge=0.0, le=10.0, description="Min signal-to-noise ratio (|predicted| / realized_vol)")
    rejection_rate_critical: float = Field(default=0.80, ge=0.5, le=1.0, description="Rejection rate above which SIGNAL_DEGRADED fires")
    ic_lookback: PositiveInt = Field(default=20, description="Rolling window for IC computation")
    regime_predictiveness_alpha: float = Field(default=0.05, ge=0.001, le=0.20, description="p-value threshold for regime t-test")
    min_regime_trades_for_ban: PositiveInt = Field(default=10, description="Min trades before regime can be banned")
    toxic_regime_wr: float = Field(default=0.30, ge=0.0, le=1.0, description="Win rate below which regime is toxic")
    toxic_regime_pnl: float = Field(default=-50.0, description="Cumulative PnL below which regime is toxic")

class EntryGateConfig(BaseModel):
    """Magic numbers extracted from entry logic."""
    autocorr_trending_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Autocorrelation above which regime is trending")
    entropy_acc_extreme_mult: float = Field(default=3.0, ge=1.0, le=10.0, description="Multiplier on entropy acceleration threshold for extreme rejection")
    flow_imbalance_threshold: float = Field(default=0.10, ge=0.0, le=1.0, description="Flow imbalance threshold for directional confirmation")
    min_confidence_gate: float = Field(default=0.20, ge=0.0, le=1.0, description="Min confidence to allow entry")
    min_hold_periods: PositiveInt = Field(default=3, description="Min process_window cycles before exit allowed")
    bootstrap_mu_vel_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Min mu_velocity/sigma ratio for bootstrap signal")
    bootstrap_strength_scale: float = Field(default=0.5, ge=0.0, le=2.0, description="Scale factor for bootstrap signal strength")
    hard_stop_multiplier: float = Field(default=1.5, ge=1.0, le=5.0, description="Hard stop as multiple of stop_loss_pct")
    min_dollar_stop: float = Field(default=50.0, ge=0.0, description="Minimum dollar stop loss floor")

class ExitRegimeScaleConfig(BaseModel):
    """Regime-type exit parameter scales (extracted from adaptive_exits.py magic numbers)."""
    trending_stop_scale: float = Field(default=0.7, ge=0.1, le=2.0)
    trending_tp_scale: float = Field(default=0.8, ge=0.1, le=2.0)
    trending_trail_scale: float = Field(default=0.6, ge=0.1, le=2.0)
    reverting_stop_scale: float = Field(default=1.2, ge=0.1, le=3.0)
    reverting_tp_scale: float = Field(default=1.3, ge=0.1, le=3.0)
    reverting_trail_scale: float = Field(default=1.2, ge=0.1, le=3.0)
    transition_vol_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    transition_vol_scale: float = Field(default=0.8, ge=0.0, le=2.0)
    trailing_activation_pct: float = Field(default=0.6, ge=0.0, le=1.0, description="TP fraction to activate trailing")
    trailing_tight_factor: float = Field(default=0.5, ge=0.0, le=1.0, description="Tighten factor when profit exceeds 1.5x TP")
    trailing_normal_factor: float = Field(default=0.7, ge=0.0, le=1.0, description="Normal trailing tighten factor")

class DecisionConfig(BaseModel):
    long: LongEntryConfig
    short: ShortEntryConfig
    liquidity: LiquidityConfig
    exit: ExitConfig = ExitConfig()
    sizing: SizingConfig = SizingConfig()
    signal_gate: SignalGateConfig = SignalGateConfig()
    entry_gate: EntryGateConfig = EntryGateConfig()
    exit_regime_scales: ExitRegimeScaleConfig = ExitRegimeScaleConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    microstructure: MicrostructureConfig = MicrostructureConfig()

# --- Calibration ---

class CalibrationConfig(BaseModel):
    # attention aux degats
    # les petits reglages
    tail_slope_tighten: float = Field(default=2.0, description="Tail slope above which stops tighten")
    kurtosis_tighten: float = Field(default=3.0, description="Kurtosis above which stops tighten")
    entropy_reduce_threshold: float = Field(default=4.0, description="Entropy above which size reduces")
    volatility_high_threshold: float = Field(default=0.01, description="Sigma above which exposure reduces")
    volatility_trend_scale: float = Field(default=10.0, description="Scaling factor for vol trend penalty")

# --- Tail Risk ---

class TailRiskConfig(BaseModel):
    # attention aux degats
    # les petits reglages
    min_tail_points: PositiveInt = Field(default=5, description="Minimum tail points for Hill estimator")
    tail_fraction: float = Field(default=0.1, ge=0.01, le=0.5, description="Fraction of sorted returns used for tail")
    default_tail_slope: float = Field(default=2.0, description="Default tail slope when estimation fails")
    alpha_clamp_min: float = Field(default=1.0, description="Min alpha clamp for tail slope")
    alpha_clamp_max: float = Field(default=10.0, description="Max alpha clamp for tail slope")
    fat_tail_kurtosis_threshold: float = Field(default=3.0, description="Kurtosis threshold for fat tail warning")
    severe_kurtosis: float = Field(default=5.0, description="Severe kurtosis warning threshold")
    low_tail_slope: float = Field(default=2.5, description="Low tail slope warning threshold")
    severe_cvar: float = Field(default=-0.05, description="Severe CVaR warning threshold")
    gap_risk_factor: float = Field(default=0.6, ge=0.0, le=1.0, description="Empirical gap factor: gaps are ~60% of continuous-time prediction")

# --- Drift Monitor ---

class DriftConfig(BaseModel):
    # les petits reglages
    structural_break_threshold: float = Field(default=2.0, ge=0.0, description="Distance threshold for structural break detection")
    drift_window: PositiveInt = Field(default=100, description="Window size for drift history")

# --- Execution Simulation ---

class RegimeSlippageConfig(BaseModel):
    """Regime-conditional slippage multipliers."""
    low_vol_mult: float = Field(default=1.0, ge=0.1, le=5.0, description="Slippage multiplier in low-vol regime")
    normal_mult: float = Field(default=1.5, ge=0.5, le=5.0, description="Slippage multiplier in normal regime")
    high_vol_mult: float = Field(default=2.5, ge=1.0, le=10.0, description="Slippage multiplier in high-vol regime")
    crisis_mult: float = Field(default=5.0, ge=1.0, le=20.0, description="Slippage multiplier in crisis regime")
    vol_threshold_high: float = Field(default=0.02, ge=0.001, description="Volatility above which high-vol multiplier applies")
    vol_threshold_crisis: float = Field(default=0.05, ge=0.001, description="Volatility above which crisis multiplier applies")

class ImpactCalibrationConfig(BaseModel):
    """Impact model calibration — conservative priors until calibrated from fills."""
    temporary_coeff: float = Field(default=0.5, ge=0.01, le=5.0, description="Temporary impact coefficient (Almgren-Chriss eta)")
    permanent_coeff: float = Field(default=0.1, ge=0.001, le=2.0, description="Permanent impact coefficient")
    max_impact_bps: float = Field(default=50.0, ge=1.0, description="Max acceptable impact in bps")
    is_calibrated: bool = Field(default=False, description="False = conservative priors, True = calibrated from fills")

class ExecutionSimConfig(BaseModel):
    # simu pour pas pleurer en live
    # verif rapide
    base_fee_bps: float = Field(default=1.0, ge=0.0, description="Base fee in basis points")
    slippage_std_bps: float = Field(default=1.0, ge=0.0, description="Slippage noise std in basis points")
    impact_coefficient: float = Field(default=0.00001, ge=0.0, description="Market impact coefficient (k * sqrt(size))")
    regime_slippage: RegimeSlippageConfig = RegimeSlippageConfig()
    impact_calibration: ImpactCalibrationConfig = ImpactCalibrationConfig()

# --- Backtest ---

class BacktestConfig(BaseModel):
    # simu pour pas pleurer en live
    # verif rapide
    order_entry_latency_sec: float = Field(default=0.005, ge=0.0, description="Simulated order entry latency")
    market_data_latency_sec: float = Field(default=0.001, ge=0.0, description="Simulated market data latency")
    relaxed_confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence floor override for backtest")
    relaxed_slippage_tolerance: float = Field(default=0.05, ge=0.0, description="Slippage tolerance override for backtest")

# --- Alerting ---

class AlertConfig(BaseModel):
    # les petits reglages
    max_history: PositiveInt = Field(default=1000, description="Max alert history entries")
    cooldown_seconds: float = Field(default=60.0, ge=0.0, description="Cooldown between repeated alerts of same type")
    drawdown_warning: float = Field(default=0.05, ge=0.0, le=1.0, description="Drawdown level for warning alert")
    drawdown_critical: float = Field(default=0.10, ge=0.0, le=1.0, description="Drawdown level for critical alert")
    latency_warning_ms: PositiveFloat = Field(default=50.0, description="Latency threshold for warning")
    error_rate_warning: float = Field(default=0.05, ge=0.0, le=1.0, description="Error rate threshold for warning")

# --- Alpha Layer ---

class AlphaConfig(BaseModel):
    # on cherche les pepites
    # les petits reglages
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Return predictor SGD learning rate")
    l2_lambda: float = Field(default=0.01, ge=0.0, le=1.0, description="L2 regularization strength")
    forgetting_factor: float = Field(default=0.995, ge=0.9, le=1.0, description="Exponential forgetting factor")
    min_samples: PositiveInt = Field(default=30, description="Min samples before predictions are valid")
    signal_actionable_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Minimum signal strength to act on")
    decay_max_lookback: PositiveInt = Field(default=50, description="Max windows to track alpha decay")

# --- Statistical Validation ---

class StatisticsConfig(BaseModel):
    # les petits reglages
    n_bootstrap: PositiveInt = Field(default=5000, description="Number of bootstrap resamples")
    n_permutations: PositiveInt = Field(default=5000, description="Number of permutation test iterations")
    significance_level: float = Field(default=0.05, ge=0.01, le=0.20, description="Statistical significance threshold (alpha)")

# --- Compliance ---

class ComplianceConfig(BaseModel):
    price_collar_pct: float = Field(default=0.05, ge=0.0, le=1.0, description="Reject orders deviating more than this from last price")
    max_order_notional: PositiveFloat = Field(default=500_000.0, description="Max notional per single order")
    max_position_notional: PositiveFloat = Field(default=2_000_000.0, description="Max notional per symbol")
    daily_loss_limit_pct: float = Field(default=0.05, ge=0.0, le=1.0, description="Max daily loss as % of session start equity")
    daily_loss_limit_abs: PositiveFloat = Field(default=5000.0, description="Max daily loss in absolute dollars")
    min_trade_interval_s: float = Field(default=10.0, ge=0.0, description="Min seconds between opposite-direction trades on same symbol")
    max_quote_age_s: float = Field(default=5.0, ge=0.0, description="Reject orders when last quote is older than this")

# --- Portfolio Risk ---

class PortfolioRiskConfig(BaseModel):
    initial_equity: PositiveFloat = Field(default=100000.0, description="Starting portfolio equity")
    var_window: PositiveInt = Field(default=252, description="Lookback window for VaR computation")
    correlation_window: PositiveInt = Field(default=60, description="Lookback for correlation matrix")
    # Regime budget overrides — nominal caps are large; VaR is the binding constraint
    normal_max_gross: PositiveFloat = Field(default=10_000_000.0, description="Max gross exposure in NORMAL regime")
    elevated_max_gross: PositiveFloat = Field(default=7_500_000.0, description="Max gross exposure in ELEVATED regime")
    stressed_max_gross: PositiveFloat = Field(default=5_000_000.0, description="Max gross exposure in STRESSED regime")
    crisis_max_gross: PositiveFloat = Field(default=2_500_000.0, description="Max gross exposure in CRISIS regime")
    max_var_95_pct: float = Field(default=0.02, ge=0.0, le=1.0, description="Max 95% VaR as pct of equity")
    max_single_position_pct: float = Field(default=0.30, ge=0.0, le=1.0, description="Max single position as pct of gross")

# --- Data Quality ---

class DataQualityConfig(BaseModel):
    max_tick_gap_ms: PositiveFloat = Field(default=5000.0, description="Warning threshold for tick gaps (ms)")
    critical_tick_gap_ms: PositiveFloat = Field(default=30000.0, description="Critical threshold for tick gaps (ms)")
    max_quote_age_s: float = Field(default=5.0, ge=0.0, description="Max quote age before warning")
    price_jump_warning_sigma: float = Field(default=5.0, ge=1.0, description="Price jump warning in sigma")
    price_jump_critical_sigma: float = Field(default=10.0, ge=1.0, description="Price jump critical in sigma")
    volume_spike_mult: float = Field(default=20.0, ge=2.0, description="Volume spike multiplier vs average")
    max_clock_drift_ms: PositiveFloat = Field(default=500.0, description="Max exchange-receive clock drift (ms)")

# --- Monitoring ---

class MonitoringConfig(BaseModel):
    dashboard_enabled: bool = Field(default=True, description="Enable HTTP monitoring dashboard")
    dashboard_port: PositiveInt = Field(default=8080, description="Dashboard HTTP port")
    webhook_cooldown_s: float = Field(default=60.0, ge=0.0, description="Min time between same-type alerts")
    max_alerts_per_hour: PositiveInt = Field(default=30, description="Max webhook alerts per hour")

# --- Alpha Ensemble ---

class EnsembleConfig(BaseModel):
    enabled: bool = Field(default=True, description="Use ensemble instead of single predictor")
    min_samples: PositiveInt = Field(default=50, description="Min observations before ensemble predictions are valid")
    gbt_retrain_interval: PositiveInt = Field(default=500, description="Retrain GBT model every N updates")
    weight_decay: float = Field(default=0.98, ge=0.9, le=1.0, description="Meta-weight EMA decay")
    rollback_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Performance degradation threshold for rollback")

# --- Execution Analytics ---

class ExecutionAnalyticsConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable execution quality tracking")
    max_history: PositiveInt = Field(default=5000, description="Max fill records to keep")
    twap_threshold_qty: PositiveFloat = Field(default=100.0, description="Order qty above which TWAP is used")
    twap_base_duration_s: float = Field(default=30.0, ge=1.0, description="Base TWAP duration in seconds")

# --- Volatility ---

class VolatilityConfig(BaseModel):
    garch_refit_interval: int = 200
    garch_min_obs: int = 100

# --- Microstructure Features ---

class MicrostructureFeaturesConfig(BaseModel):
    vpin_bucket_size: int = 500
    vpin_n_buckets: int = 50
    kyle_lambda_window: int = 100
    amihud_window: int = 200

# --- Top-Level ---

class ThresholdsConfig(BaseModel):
    regime: RegimeConfig
    risk: RiskConfig
    pdf: PDFConfig
    decision: DecisionConfig
    calibration: CalibrationConfig = CalibrationConfig()
    tail_risk: TailRiskConfig = TailRiskConfig()
    drift: DriftConfig = DriftConfig()
    execution_sim: ExecutionSimConfig = ExecutionSimConfig()
    backtest: BacktestConfig = BacktestConfig()
    alerts: AlertConfig = AlertConfig()
    alpha: AlphaConfig = AlphaConfig()
    statistics: StatisticsConfig = StatisticsConfig()
    compliance: ComplianceConfig = ComplianceConfig()
    portfolio_risk: PortfolioRiskConfig = PortfolioRiskConfig()
    data_quality: DataQualityConfig = DataQualityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    execution_analytics: ExecutionAnalyticsConfig = ExecutionAnalyticsConfig()
    volatility: VolatilityConfig = VolatilityConfig()
    microstructure_features: MicrostructureFeaturesConfig = MicrostructureFeaturesConfig()

class AppConfig(BaseModel):
    execution: ExecutionConfig
    instruments: InstrumentsConfig
    thresholds: ThresholdsConfig

    @validator('thresholds')
    def cross_validate_thresholds(cls, v):
        # window_size must be > min_cluster_size (otherwise clustering never has enough data)
        if v.regime.window_size <= v.regime.min_cluster_size:
            raise ValueError(
                f"regime.window_size ({v.regime.window_size}) must be > "
                f"regime.min_cluster_size ({v.regime.min_cluster_size})"
            )
        # Stop loss must be < take profit (otherwise R:R is always < 1)
        if v.decision.exit.min_stop_pct >= v.decision.exit.min_tp_pct:
            raise ValueError(
                f"exit.min_stop_pct ({v.decision.exit.min_stop_pct}) must be < "
                f"exit.min_tp_pct ({v.decision.exit.min_tp_pct})"
            )
        # Drawdown warning must be < drawdown critical
        if v.alerts.drawdown_warning >= v.alerts.drawdown_critical:
            raise ValueError(
                f"alerts.drawdown_warning ({v.alerts.drawdown_warning}) must be < "
                f"alerts.drawdown_critical ({v.alerts.drawdown_critical})"
            )
        return v
