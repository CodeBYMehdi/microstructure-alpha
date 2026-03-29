import asyncio
import logging
import sys
import unittest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Add project root to path
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.event_bus import event_bus, MarketEvent, EventType
from data.ib_client import IBClient
from data.realtime_stream import RealTimeStream
from data.l2_orderbook import L2OrderBook
from data.normalizer import DataNormalizer
from monitoring.viz_connector import VizConnector
from monitoring.viz_manager import VisualizationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineValidator")

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Reset abonnés bus événts singleton pour isolation (bricolage mais requis pour tests)
        event_bus._subscribers = {}
        event_bus._async_subscribers = {}
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.ib_client = IBClient()
        # Mock connexion et boucle
        self.ib_client._loop = self.loop
        self.ib_client._connected = True
        # Mock send pour éviter erreurs "Non connecté"
        self.ib_client.send = MagicMock()
        
        self.stream = RealTimeStream(self.ib_client)
        # Set min_history à 1 pour tests
        self.normalizer = DataNormalizer(min_history=1)

    def tearDown(self):
        self.loop.close()

    def test_demo_mode(self):
        # petits dessins
        # verif rapide
        if not os.environ.get("DEMO_MODE"):
            logger.info("Skipping Demo Mode (Set DEMO_MODE=1 to run)")
            return

        import random
        
        # 1. Démarrer Gest Viz
        viz_mgr = VisualizationManager()
        viz_mgr.start()
        
        # 2. Config Connecteur
        connector = VizConnector(
            self.normalizer, 
            viz_mgr.q_surface, 
            viz_mgr.q_density, 
            instrument_id="1"
        )
        
        # 3. Config Instrument
        symbol = "DEMO"
        # Mock souscription pour générer req_ids
        self.stream.subscribe_instrument(symbol)
        req_id_l2 = self.stream.active_subscriptions[symbol]['l2']
        req_id_l1 = self.stream.active_subscriptions[symbol]['l1']
        
        # Surcharger instrument_id connecteur pour correspondre req_id mocké
        connector.instrument_id = str(req_id_l2)
        
        book = self.stream.get_order_book(symbol)
        self.normalizer.register_book(str(req_id_l2), book)
        
        logger.info("Starting Demo Loop... Press Ctrl+C to stop")
        
        async def run_demo():
            price = 100.0
            try:
                # Simuler Boucle Données
                for i in range(10000):
                    # Marche Aléatoire
                    shock = random.gauss(0, 0.1)
                    price += shock
                    
                    # MAJ L2
                    self.ib_client.updateMktDepth(req_id_l2, 0, 0, 1, price - 0.01, 100) # Bid
                    self.ib_client.updateMktDepth(req_id_l2, 0, 0, 0, price + 0.01, 100) # Ask
                    
                    # Trade
                    if random.random() < 0.3:
                        self.ib_client.tickPrice(req_id_l1, 4, price, 0)
                    
                    # Allow event loop to process events
                    await asyncio.sleep(0.05)
                    
            except KeyboardInterrupt:
                pass
            finally:
                viz_mgr.stop()

        # Run the async demo loop
        try:
            self.loop.run_until_complete(run_demo())
        except KeyboardInterrupt:
            viz_mgr.stop()

    def test_l1_flow(self):
        async def run():
            logger.info("Testing Level 1 Data Flow...")
            
            # Capture events
            received_events = []
            def on_event(event):
                received_events.append(event)
            event_bus.subscribe(EventType.QUOTE, on_event)
            event_bus.subscribe(EventType.TRADE, on_event)

            # Simulate IB tickPrice (Ask)
            # reqId=1, tickType=2 (Ask), price=100.5, attrib=0
            self.ib_client.tickPrice(1, 2, 100.5, 0)
            
            # Allow loop to process
            await asyncio.sleep(0.01)
            
            # Simulate IB tickSize (Ask Size)
            # reqId=1, tickType=3 (Ask Size), size=1000
            self.ib_client.tickSize(1, 3, 1000)
            
            await asyncio.sleep(0.01)
            
            # Verify events
            self.assertEqual(len(received_events), 2)
            self.assertEqual(received_events[0].event_type, EventType.QUOTE)
            self.assertEqual(received_events[0].price, 100.5)
            self.assertEqual(received_events[0].side, 'ASK')
            
            self.assertEqual(received_events[1].event_type, EventType.QUOTE)
            self.assertEqual(received_events[1].size, 1000)
            self.assertEqual(received_events[1].side, 'ASK')

        self.loop.run_until_complete(run())

    def test_full_integration(self):
        # verif rapide
        async def run_test():
            logger.info("Starting Integration Test")
            
            # 1. Setup Instrument
            symbol = "AAPL"
            # Mock subscribe to generate req_ids
            self.stream.subscribe_instrument(symbol)
            
            req_id_l1 = self.stream.active_subscriptions[symbol]['l1']
            req_id_l2 = self.stream.active_subscriptions[symbol]['l2']
            
            # Register book with normalizer
            book = self.stream.get_order_book(symbol)
            self.normalizer.register_book(str(req_id_l2), book)
            
            # 2. Simulate L2 Data (Order Book Build)
            logger.info("Simulating L2 Data...")
            # Insert Bids
            self.ib_client.updateMktDepth(req_id_l2, 0, 0, 1, 150.00, 100) # Bid @ 150
            self.ib_client.updateMktDepth(req_id_l2, 1, 0, 1, 149.95, 200) # Bid @ 149.95
            
            # Insert Asks
            self.ib_client.updateMktDepth(req_id_l2, 0, 0, 0, 150.05, 100) # Ask @ 150.05
            self.ib_client.updateMktDepth(req_id_l2, 1, 0, 0, 150.10, 300) # Ask @ 150.10
            
            # Give time for async bus to process
            await asyncio.sleep(0.1)
            
            # Check Order Book State
            best_bid = book.state.get_best_bid()
            best_ask = book.state.get_best_ask()
            logger.info(f"Book State: Bid={best_bid}, Ask={best_ask}")
            
            self.assertEqual(best_bid, (150.00, 100))
            self.assertEqual(best_ask, (150.05, 100))
            
            # 3. Simulate L1 Trade (Trigger Feature Computation)
            logger.info("Simulating Trade...")
            # TickPrice (Last)
            self.ib_client.tickPrice(req_id_l1, 4, 150.02, 0)
            await asyncio.sleep(0.1)
            
            # 4. Check Normalizer Output
            features = self.normalizer.compute_features(str(req_id_l2))
            logger.info(f"Features: {features}")
            
            self.assertIsNotNone(features)
            self.assertAlmostEqual(features.mid_price, 150.025)
            self.assertAlmostEqual(features.spread, 0.05)
            
            # Check Imbalance: (300 - 400) / 700 = -100/700 = -0.1428
            # Wait, book state: 
            # Bids: 150.00(100), 149.95(200) -> Total 300
            # Asks: 150.05(100), 150.10(300) -> Total 400
            # Imbalance = (300 - 400) / 700 = -0.142857
            self.assertAlmostEqual(features.order_book_imbalance, -1.0/7.0, places=4)
            
        self.loop.run_until_complete(run_test())

if __name__ == '__main__':
    unittest.main()
