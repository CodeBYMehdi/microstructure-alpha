
import unittest
import os
from datetime import datetime
from data.tick_stream import SyntheticTickStream, RealTickStream
from data.storage import TimeSeriesStorage

class TestStorageReplay(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_ticks.h5"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_storage_and_replay(self):
        # 1. Générer ticks synth.
        symbol = "BTC-USDT"
        count = 100
        gen_stream = SyntheticTickStream(symbol=symbol, duration_seconds=60, seed=42)
        ticks = list(gen_stream.stream())[:count] # Limiter compte pour vitesse
        
        # 2. Stocker ticks
        storage = TimeSeriesStorage(self.test_file, mode='w')
        storage.store_ticks(ticks)
        
        stored_count = storage.get_tick_count()
        self.assertEqual(stored_count, len(ticks))
        
        # 3. Rejouer ticks
        replay_stream = RealTickStream(self.test_file, symbol=symbol)
        replayed_ticks = list(replay_stream.stream())
        
        # 4. Vérifier
        self.assertEqual(len(replayed_ticks), len(ticks))
        
        for t1, t2 in zip(ticks, replayed_ticks):
            self.assertEqual(t1.timestamp, t2.timestamp)
            self.assertEqual(t1.price, t2.price)
            self.assertEqual(t1.symbol, t2.symbol)

if __name__ == '__main__':
    unittest.main()
