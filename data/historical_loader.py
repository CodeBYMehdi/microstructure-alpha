import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Callable
from ibapi.contract import Contract
from data.ib_client import IBClient
from data.event_bus import event_bus, MarketEvent, EventType

logger = logging.getLogger(__name__)

class HistoricalLoader:
    def __init__(self, ib_client_instance: IBClient):
        self.ib = ib_client_instance

    async def load_historical_data(self, 
                                   symbol: str, 
                                   end_time: datetime, 
                                   duration_str: str = "1 D", 
                                   bar_size_setting: str = "1 min", 
                                   what_to_show: str = "TRADES", 
                                   use_rth: int = 1,
                                   sec_type: str = "STK",
                                   exchange: str = "SMART",
                                   currency: str = "USD"):
        # la tuyauterie de donnees
        # on ramene les datas
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        req_id = self.ib.get_req_id()
        
        # API IB req format chaîne pour endDateTime: yyyyMMdd HH:mm:ss
        end_str = end_time.strftime("%Y%m%d %H:%M:%S") + " UTC"
        
        logger.info(f"Demande données histo pour {symbol} jusqu'à {end_str}")
        
        # Créer future pour attendre complétion (optionnel, ou laisser couler événements)
        # Pour simplicité dans arch événementielle, on déclenche juste requête.
        # IBClient émettra MarketEvents (BAR).
        
        self.ib.reqHistoricalData(
            req_id, 
            contract, 
            end_str, 
            duration_str, 
            bar_size_setting, 
            what_to_show, 
            use_rth, 
            1, # formatDate: 1 = yyyyMMdd HH:mm:ss
            False, # KeepUpToDate
            []
        )
        
        return req_id

    async def load_historical_ticks(self,
                                    symbol: str,
                                    start_time: datetime,
                                    end_time: datetime,
                                    number_of_ticks: int = 1000,
                                    what_to_show: str = "TRADES",
                                    sec_type: str = "STK",
                                    exchange: str = "SMART",
                                    currency: str = "USD"):
        # on ramene les datas
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency

        req_id = self.ib.get_req_id()
        
        start_str = start_time.strftime("%Y%m%d %H:%M:%S")
        end_str = end_time.strftime("%Y%m%d %H:%M:%S")
        
        logger.info(f"Demande ticks histo pour {symbol}")
        
        # reqHistoricalTicks(reqId, contract, startDateTime, endDateTime, numberOfTicks, whatToShow, useRth, ignoreSize, miscOptions)
        self.ib.reqHistoricalTicks(
            req_id,
            contract,
            start_str,
            "", # endDateTime généralement null si start défini, ou vice versa. 
            number_of_ticks,
            what_to_show,
            1,
            True,
            []
        )
        
        return req_id
