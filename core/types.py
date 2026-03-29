from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import datetime
import numpy as np

# --- Market Data ---

@dataclass(frozen=True)
class Tick:
    # la tuyauterie de donnees
    timestamp: datetime.datetime
    symbol: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    exchange: str = "generic"
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    bids: Optional[List[tuple]] = None # L2 representation: list of (price, size)
    asks: Optional[List[tuple]] = None # L2 representation: list of (price, size)

# --- Regime ---

class RegimeType(Enum):
    UNDEFINED = "UNDEFINED"
    STABLE_BULL = "STABLE_BULL"
    STABLE_BEAR = "STABLE_BEAR"
    VOLATILE_NEUTRAL = "VOLATILE_NEUTRAL"
    CRASH = "CRASH"
    RALLY = "RALLY"

@dataclass(frozen=True)
class RegimeState:
    # dans quel etat j'erre
    regime: RegimeType
    confidence: float
    entropy: float
    timestamp: datetime.datetime
    metadata: Dict[str, Any]

# --- Trading ---

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    FLAT = "FLAT"

@dataclass(frozen=True)
class TradeProposal:
    # l'usine a gaz
    action: TradeAction
    symbol: str
    quantity: float
    price: Optional[float]
    reason: str
    timestamp: datetime.datetime
    regime_state: RegimeState

@dataclass(frozen=True)
class OrderResult:
    # on passe a la caisse
    # execution sans pitie
    order_id: str
    status: str
    filled_price: float
    filled_quantity: float
    timestamp: datetime.datetime
    fees: float

# --- PDF / Model ---

@dataclass(frozen=True)
class PDFData:
    # la tuyauterie de donnees
    x: np.ndarray  # Domain points (returns)
    y: np.ndarray  # Density values
    method: str     # 'KDE', 'GMMDensity', 'Parametric'
    bandwidth: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelHealth:
    # la grosse machine
    data_drift_score: float
    output_stability: float
    confidence_trend: float
    error_rate: float
    is_healthy: bool = True

@dataclass
class PDFDiagnostics:
    # l'usine a gaz
    mode_collapse: bool
    tail_instability: bool
    entropy_jump: bool

@dataclass
class PDFModelOutput:
    # la grosse machine
    pdf_callable: Callable  # Callable[[np.ndarray], np.ndarray]
    log_likelihood: float
    entropy: float
    tail_slope: float
    diagnostics: PDFDiagnostics
    valid: bool = True

# --- Regime Output ---

@dataclass
class RegimeOutput:
    # l'usine a gaz
    regime_id: str
    confidence: float
    cluster_density: float
    persistence_estimate: float
    is_noise: bool = False

# --- Transition ---

@dataclass
class TransitionProbability:
    # la grosse machine
    probability: float  # [0, 1]
    is_significant: bool
    model_used: str

# --- Liquidity ---

@dataclass
class LiquidityState:
    # dans quel etat j'erre
    spread: float
    depth_imbalance: float  # (BidVol - AskVol) / (BidVol + AskVol)
    depth_slope: float      # Rate of change of depth
    trade_intensity: float  # Volume per second
    is_liquid: bool = True

# --- Decision Log ---

@dataclass
class DecisionLog:
    # l'usine a gaz
    timestamp: float
    regime_id: str
    action: str
    reason: str
    delta_mu: float
    delta_sigma: float
    delta_entropy: float
    delta_skew: float
    regime_age: int
    transition_strength: float
    tail_risk: float
    l2_liquidity_slope: float
    result: str

# --- Risk ---

@dataclass
class RiskAdjustments:
    # attention aux degats
    # dans quel etat j'erre
    stop_multiplier: float
    size_scaler: float
    max_exposure: float
    valid: bool = True
