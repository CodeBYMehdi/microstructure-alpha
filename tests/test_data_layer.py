# la tuyauterie de donnees
# verif rapide

import unittest
import os
import tempfile
from datetime import datetime
import numpy as np
from data.tick_stream import SyntheticTickStream
from data.storage import TimeSeriesStorage
from core.types import Tick


class TestSyntheticStream(unittest.TestCase):

    def test_determinism(self):
        # verif rapide
        # le flux en scred
        start_time = datetime(2024, 1, 1, 10, 0, 0)
        stream1 = SyntheticTickStream(seed=123, duration_seconds=10, start_time=start_time)
        stream2 = SyntheticTickStream(seed=123, duration_seconds=10, start_time=start_time)
        ticks1 = list(stream1.stream())
        ticks2 = list(stream2.stream())
        self.assertEqual(len(ticks1), len(ticks2))
        for t1, t2 in zip(ticks1, ticks2):
            self.assertEqual(t1, t2)

    def test_different_seeds(self):
        # verif rapide
        # le flux en scred
        stream1 = SyntheticTickStream(seed=123, duration_seconds=10)
        stream2 = SyntheticTickStream(seed=456, duration_seconds=10)
        ticks1 = list(stream1.stream())
        ticks2 = list(stream2.stream())
        self.assertNotEqual(ticks1, ticks2)

    def test_positive_prices(self):
        # verif rapide
        stream = SyntheticTickStream(seed=42, duration_seconds=5)
        for tick in stream.stream():
            self.assertGreater(tick.price, 0)
            self.assertGreaterEqual(tick.volume, 0)

    def test_tick_type(self):
        # verif rapide
        stream = SyntheticTickStream(seed=42, duration_seconds=1)
        for tick in stream.stream():
            self.assertIsInstance(tick, Tick)
            break  # Just check first


class TestTimeSeriesStorage(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.tmpdir, "test_storage.h5")

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        os.rmdir(self.tmpdir)

    def test_store_and_load(self):
        # on ramene les datas
        # verif rapide
        storage = TimeSeriesStorage(self.filepath)
        ticks = [
            Tick(timestamp=datetime(2024, 1, 1, 10, 0, i), symbol="TEST",
                 price=100.0 + i, volume=10.0)
            for i in range(5)
        ]
        storage.store_ticks(ticks)

        loaded = list(storage.load_ticks())
        self.assertEqual(len(loaded), 5)
        self.assertEqual(loaded[0].symbol, "TEST")
        self.assertAlmostEqual(loaded[0].price, 100.0, places=1)

    def test_symbol_filter(self):
        # on ramene les datas
        # verif rapide
        storage = TimeSeriesStorage(self.filepath)
        ticks = [
            Tick(timestamp=datetime(2024, 1, 1, 10, 0, 0), symbol="A", price=100.0, volume=1.0),
            Tick(timestamp=datetime(2024, 1, 1, 10, 0, 1), symbol="B", price=200.0, volume=2.0),
        ]
        storage.store_ticks(ticks)
        loaded_a = list(storage.load_ticks(symbol="A"))
        self.assertEqual(len(loaded_a), 1)
        self.assertEqual(loaded_a[0].symbol, "A")

    def test_tick_count(self):
        # verif rapide
        # le bif
        storage = TimeSeriesStorage(self.filepath)
        ticks = [
            Tick(timestamp=datetime(2024, 1, 1, 10, 0, i), symbol="TEST",
                 price=100.0, volume=1.0)
            for i in range(10)
        ]
        storage.store_ticks(ticks)
        self.assertEqual(storage.get_tick_count(), 10)

    def test_empty_store(self):
        # verif rapide
        storage = TimeSeriesStorage(self.filepath)
        storage.store_ticks([])
        self.assertFalse(os.path.exists(self.filepath))


class TestTickDataIntegrity(unittest.TestCase):

    def test_tick_immutability(self):
        # la tuyauterie de donnees
        # verif rapide
        tick = Tick(timestamp=datetime.now(), symbol="TEST", price=100.0, volume=10.0)
        with self.assertRaises(AttributeError):
            tick.price = 200.0


if __name__ == '__main__':
    unittest.main()
