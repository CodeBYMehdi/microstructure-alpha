from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Any
from .types import Tick, RegimeState, TradeProposal, OrderResult, TradeAction, PDFData

class IDataSource(ABC):
    @abstractmethod
    def stream(self) -> Iterator[Tick]:
        # le flux en scred
        pass

class IMicrostructureModel(ABC):
    @abstractmethod
    def update(self, tick: Tick) -> None:
        # dans quel etat j'erre
        pass

    @abstractmethod
    def get_current_pdf(self) -> PDFData:
        # le bif
        pass

class IRegimeClassifier(ABC):
    @abstractmethod
    def update(self, pdf_data: PDFData) -> RegimeState:
        # la tuyauterie de donnees
        pass

    @abstractmethod
    def get_current_regime(self) -> RegimeState:
        # le bif
        # dans quel etat j'erre
        pass

class IRiskManager(ABC):
    @abstractmethod
    def validate(self, proposal: TradeProposal) -> bool:
        # attention aux degats
        # entrainement intensif
        pass

    @abstractmethod
    def check_kill_switch(self) -> bool:
        # stop le massacre
        # le bif
        pass

class IDecisionEngine(ABC):
    @abstractmethod
    def evaluate(self, regime: RegimeState) -> Optional[TradeProposal]:
        # l'usine a gaz
        pass

class IExecutionHandler(ABC):
    @abstractmethod
    def execute(self, proposal: TradeProposal) -> OrderResult:
        # l'usine a gaz
        pass
