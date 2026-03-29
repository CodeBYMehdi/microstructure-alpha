# verif rapide

import unittest
import numpy as np
from regime.state_vector import StateVector
from regime.transition import TransitionDetector, TransitionEvent, SimpleKalmanFilter1D
from regime.labels import RegimeLabelManager
from monitoring.regime_drift import RegimeDriftMonitor


class TestStateVector(unittest.TestCase):

    def test_creation(self):
        sv = StateVector(0.001, 0.01, 0.5, 1.0, 3.0, -5.0)
        self.assertEqual(sv.mu, 0.001)
        self.assertEqual(len(sv.to_array()), 6)

    def test_immutability(self):
        sv = StateVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaises(AttributeError):
            sv.mu = 1.0

    def test_to_dict(self):
        sv = StateVector(0.001, 0.01, 0.5, 1.0, 3.0, -5.0)
        d = sv.to_dict()
        self.assertEqual(d['mu'], 0.001)
        self.assertEqual(d['entropy'], -5.0)

    def test_repr(self):
        sv = StateVector(0.001, 0.01, 0.5, 1.0, 3.0, -5.0)
        r = repr(sv)
        self.assertIn("StateVector", r)
        self.assertIn("mu=", r)


class TestTransitionDetector(unittest.TestCase):

    def test_no_transition_same_regime(self):
        detector = TransitionDetector()
        s1 = StateVector(0.0, 0.01, 0.0, 0.0, 2.0, -5.0)
        s2 = StateVector(0.001, 0.01, 0.0, 0.0, 2.0, -5.0)
        self.assertIsNone(detector.update(0, s1))
        self.assertIsNone(detector.update(0, s2))

    def test_transition_detected(self):
        detector = TransitionDetector()
        s1 = StateVector(0.0, 0.01, 0.0, 0.0, 2.0, -5.0)
        s2 = StateVector(0.01, 0.05, 1.0, 2.0, 4.0, -3.0)
        s3 = StateVector(0.02, 0.06, 1.5, 3.0, 5.0, -2.0)
        detector.update(0, s1)
        detector.update(0, s2)
        event = detector.update(1, s3)
        self.assertIsNotNone(event)
        self.assertEqual(event.from_regime, 0)
        self.assertEqual(event.to_regime, 1)

    def test_transition_significance(self):
        detector = TransitionDetector(strength_threshold=0.3)
        s1 = StateVector(0.0, 0.01, 0.0, 0.0, 2.0, -5.0)
        s2 = StateVector(0.001, 0.011, 0.01, 0.01, 2.01, -4.99)
        detector.update(0, s1)
        event = detector.update(1, s2)
        self.assertIsNotNone(event)
        # Small change -> may not be significant
        self.assertIsInstance(event.is_significant, bool)

    def test_kalman_smoothing(self):
        detector = TransitionDetector()
        # Feed several states to populate history
        for i in range(5):
            detector.update(0, StateVector(i * 0.001, 0.01, 0.0, 0.0, 2.0, -5.0))
        event = detector.update(1, StateVector(0.1, 0.05, 1.0, 2.0, 4.0, -3.0))
        self.assertIsNotNone(event)
        # Kalman-filtered values should be finite
        self.assertTrue(np.isfinite(event.mu_velocity))
        self.assertTrue(np.isfinite(event.mu_acceleration))


class TestSimpleKalmanFilter1D(unittest.TestCase):

    def test_initial_measurement(self):
        kf = SimpleKalmanFilter1D()
        result = kf.update(5.0)
        self.assertEqual(result, 5.0)

    def test_smoothing(self):
        kf = SimpleKalmanFilter1D()
        kf.update(0.0)
        # Noisy measurement
        result = kf.update(10.0)
        # Should be smoothed toward 0 and 10
        self.assertGreater(result, 0.0)
        self.assertLess(result, 10.0)


class TestRegimeLabelManager(unittest.TestCase):

    def test_create_profile(self):
        mgr = RegimeLabelManager()
        centroid = np.array([0.001, 0.01, 0.5, 1.0, 3.0, -5.0])
        mgr.update_profile(0, centroid, 10)
        profile = mgr.get_profile(0)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.id, 0)

    def test_noise_ignored(self):
        mgr = RegimeLabelManager()
        mgr.update_profile(-1, np.zeros(6), 1)
        self.assertIsNone(mgr.get_profile(-1))

    def test_describe(self):
        mgr = RegimeLabelManager()
        mgr.update_profile(0, np.array([0.001, 0.01, 0.5, 1.0, 3.0, -3.0]), 10)
        desc = mgr.describe(0)
        self.assertIn("Regime 0", desc)
        self.assertIn("HighVol", desc)

    def test_describe_noise(self):
        mgr = RegimeLabelManager()
        self.assertEqual(mgr.describe(-1), "NOISE/TRANSITION")

    def test_describe_unknown(self):
        mgr = RegimeLabelManager()
        desc = mgr.describe(99)
        self.assertIn("Unprofiled", desc)


class TestRegimeDriftMonitor(unittest.TestCase):

    def test_no_drift_initially(self):
        mgr = RegimeLabelManager()
        monitor = RegimeDriftMonitor(mgr)
        self.assertEqual(monitor.check_drift(0), 0.0)

    def test_drift_detection(self):
        mgr = RegimeLabelManager()
        centroid = np.array([0.0, 0.01, 0.0, 0.0, 2.0, -5.0])
        mgr.update_profile(0, centroid, 10)
        monitor = RegimeDriftMonitor(mgr)

        # Record observations far from baseline
        for i in range(20):
            far_state = StateVector(0.1, 0.1, 1.0, 5.0, 8.0, -1.0)
            monitor.record_observation(0, far_state)

        drift = monitor.check_drift(0)
        self.assertGreater(drift, 0.0)

    def test_structural_break(self):
        mgr = RegimeLabelManager()
        centroid = np.array([0.0, 0.01, 0.0, 0.0, 2.0, -5.0])
        mgr.update_profile(0, centroid, 10)
        monitor = RegimeDriftMonitor(mgr)

        for _ in range(20):
            monitor.record_observation(0, StateVector(1.0, 1.0, 5.0, 10.0, 10.0, 0.0))

        self.assertTrue(monitor.detect_structural_break(0))


if __name__ == '__main__':
    unittest.main()
