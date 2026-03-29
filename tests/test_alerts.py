# verif rapide

import unittest
import time
from monitoring.alerts import AlertManager, AlertLevel, Alert


class TestAlertManager(unittest.TestCase):

    def test_fire_alert(self):
        mgr = AlertManager(cooldown_seconds=0)
        fired = mgr.fire(AlertLevel.WARNING, "test", "Test alert")
        self.assertTrue(fired)

    def test_cooldown(self):
        mgr = AlertManager(cooldown_seconds=60)
        mgr.fire(AlertLevel.WARNING, "test", "First")
        # Second alert of same category should be suppressed
        fired = mgr.fire(AlertLevel.WARNING, "test", "Second")
        self.assertFalse(fired)

    def test_different_categories(self):
        mgr = AlertManager(cooldown_seconds=60)
        mgr.fire(AlertLevel.WARNING, "cat_a", "First")
        fired = mgr.fire(AlertLevel.WARNING, "cat_b", "Second")
        self.assertTrue(fired)

    def test_bypass_cooldown(self):
        mgr = AlertManager(cooldown_seconds=60)
        mgr.fire(AlertLevel.WARNING, "test", "First")
        fired = mgr.fire(AlertLevel.CRITICAL, "test", "Urgent", bypass_cooldown=True)
        self.assertTrue(fired)

    def test_history(self):
        mgr = AlertManager(cooldown_seconds=0, max_history=10)
        for i in range(5):
            mgr.fire(AlertLevel.INFO, f"cat_{i}", f"Alert {i}")
        history = mgr.get_history()
        self.assertEqual(len(history), 5)

    def test_history_filter(self):
        mgr = AlertManager(cooldown_seconds=0)
        mgr.fire(AlertLevel.INFO, "a", "Info")
        mgr.fire(AlertLevel.WARNING, "b", "Warning")
        mgr.fire(AlertLevel.CRITICAL, "c", "Critical")
        warnings = mgr.get_history(level=AlertLevel.WARNING)
        self.assertEqual(len(warnings), 1)

    def test_clear_history(self):
        mgr = AlertManager(cooldown_seconds=0)
        mgr.fire(AlertLevel.INFO, "test", "Test")
        mgr.clear_history()
        self.assertEqual(len(mgr.get_history()), 0)

    def test_drawdown_check(self):
        mgr = AlertManager(cooldown_seconds=0)
        mgr.check_drawdown(0.12, warning=0.05, critical=0.10)
        history = mgr.get_history(level=AlertLevel.CRITICAL)
        self.assertEqual(len(history), 1)

    def test_latency_check(self):
        mgr = AlertManager(cooldown_seconds=0)
        mgr.check_latency(100.0, threshold_ms=50.0)
        history = mgr.get_history(level=AlertLevel.WARNING)
        self.assertEqual(len(history), 1)

    def test_custom_handler(self):
        mgr = AlertManager(cooldown_seconds=0)
        captured = []
        mgr.register_handler(lambda a: captured.append(a))
        mgr.fire(AlertLevel.INFO, "test", "Hello")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].message, "Hello")


if __name__ == '__main__':
    unittest.main()
