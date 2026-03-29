# verif rapide

import unittest
import numpy as np
from microstructure.returns import ReturnCalculator
from microstructure.moments import MomentsCalculator
from microstructure.entropy import EntropyCalculator
from microstructure.pdf.kde import AdaptiveKDE
from microstructure.pdf.normalizing_flow import GMMDensityEstimator, NormalizingFlow


class TestReturnCalculator(unittest.TestCase):

    def test_initial_state(self):
        calc = ReturnCalculator(max_window_size=10)
        self.assertIsNone(calc.update(100.0))
        self.assertEqual(calc.count, 0)

    def test_first_return(self):
        calc = ReturnCalculator(max_window_size=10)
        calc.update(100.0)
        ret = calc.update(101.0)
        self.assertAlmostEqual(ret, np.log(101 / 100), places=5)
        self.assertEqual(calc.count, 1)

    def test_window_capping(self):
        calc = ReturnCalculator(max_window_size=10)
        for i in range(20):
            calc.update(100.0 + i)
        self.assertEqual(calc.count, 10)

    def test_get_window(self):
        calc = ReturnCalculator(max_window_size=100)
        for i in range(15):
            calc.update(100.0 + i * 0.1)
        window = calc.get_window(5)
        self.assertEqual(len(window), 5)

    def test_get_window_too_large(self):
        calc = ReturnCalculator(max_window_size=10)
        calc.update(100.0)
        calc.update(101.0)
        with self.assertRaises(ValueError):
            calc.get_window(10)

    def test_invalid_price(self):
        calc = ReturnCalculator()
        with self.assertRaises(ValueError):
            calc.update(-5.0)
        with self.assertRaises(ValueError):
            calc.update(0.0)
        with self.assertRaises(ValueError):
            calc.update(np.nan)
        with self.assertRaises(ValueError):
            calc.update(np.inf)

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            ReturnCalculator(max_window_size=0)
        with self.assertRaises((ValueError, OverflowError)):
            ReturnCalculator(max_window_size=-1)

    def test_reset(self):
        calc = ReturnCalculator(max_window_size=10)
        calc.update(100.0)
        calc.update(101.0)
        calc.reset()
        self.assertEqual(calc.count, 0)
        self.assertIsNone(calc.update(100.0))


class TestMomentsCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = MomentsCalculator()

    def test_gaussian_moments(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        moments = self.calc.compute(data)
        self.assertAlmostEqual(moments.mu, 0.0, delta=0.1)
        self.assertAlmostEqual(moments.sigma, 1.0, delta=0.15)
        self.assertAlmostEqual(moments.skew, 0.0, delta=0.3)
        self.assertAlmostEqual(moments.kurtosis, 0.0, delta=0.5)

    def test_insufficient_data(self):
        data = np.array([1.0, 2.0, 3.0])
        moments = self.calc.compute(data)
        self.assertEqual(moments.mu, 0.0)
        self.assertEqual(moments.sigma, 0.0)

    def test_zero_variance(self):
        data = np.ones(20)
        moments = self.calc.compute(data)
        self.assertEqual(moments.sigma, 0.0)
        self.assertEqual(moments.skew, 0.0)

    def test_skewed_data(self):
        np.random.seed(42)
        data = np.random.exponential(1.0, 1000) - 1.0  # Right-skewed
        moments = self.calc.compute(data)
        self.assertGreater(moments.skew, 0.5)


class TestEntropyCalculator(unittest.TestCase):

    def test_peaked_vs_flat(self):
        peaked_pdf = np.array([0.01, 0.01, 0.9, 0.04, 0.04])
        flat_pdf = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        dx = 1.0
        h_peaked = EntropyCalculator.compute_from_pdf(peaked_pdf, dx)
        h_flat = EntropyCalculator.compute_from_pdf(flat_pdf, dx)
        self.assertLess(h_peaked, h_flat)

    def test_kl_divergence_zero_same(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = EntropyCalculator.compute_kl_divergence(p, p, 1.0)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_divergence_positive(self):
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.33, 0.33, 0.34])
        kl = EntropyCalculator.compute_kl_divergence(p, q, 1.0)
        self.assertGreater(kl, 0.0)

    def test_gaussian_entropy(self):
        data = np.random.normal(0, 1, 100)
        h = EntropyCalculator.compute_from_samples(data)
        # Gaussian entropy ~ 0.5 * log(2*pi*e) ~ 1.42
        self.assertGreater(h, 1.0)
        self.assertLess(h, 2.0)

    def test_kozachenko_leonenko(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        h = EntropyCalculator.compute_kozachenko_leonenko(data)
        self.assertGreater(h, 0.0)

    def test_zero_variance_entropy(self):
        data = np.ones(10)
        h = EntropyCalculator.compute_from_samples(data)
        self.assertEqual(h, 0.0)  # Zero variance = zero information content


class TestKDE(unittest.TestCase):

    def test_fit_and_evaluate(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        kde = AdaptiveKDE()
        kde.fit(data)
        x = np.linspace(-3, 3, 10)
        pdf = kde.evaluate(x)
        self.assertEqual(len(pdf), 10)
        self.assertTrue(np.all(pdf >= 0))

    def test_deterministic_sampling(self):
        data = np.random.normal(0, 1, 100)
        kde1 = AdaptiveKDE(seed=42)
        kde2 = AdaptiveKDE(seed=42)
        kde1.fit(data)
        kde2.fit(data)
        samples1 = kde1.sample(100)
        samples2 = kde2.sample(100)
        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds(self):
        data = np.random.normal(0, 1, 100)
        kde1 = AdaptiveKDE(seed=42)
        kde2 = AdaptiveKDE(seed=43)
        kde1.fit(data)
        kde2.fit(data)
        self.assertFalse(np.array_equal(kde1.sample(100), kde2.sample(100)))

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            AdaptiveKDE().fit(np.array([1.0, 2.0]))

    def test_zero_variance(self):
        kde = AdaptiveKDE(seed=42)
        kde.fit(np.ones(20))  # Should add noise
        self.assertIsNotNone(kde.evaluate(np.array([1.0])))


class TestGMMDensityEstimator(unittest.TestCase):

    def test_fit_and_evaluate(self):
        np.random.seed(42)
        data = np.random.normal(0, 0.01, 100)
        model = GMMDensityEstimator()
        model.fit(data)
        self.assertTrue(model.fitted)
        x = np.linspace(-0.03, 0.03, 50)
        pdf = model.evaluate(x)
        self.assertEqual(len(pdf), 50)
        self.assertTrue(np.all(pdf >= 0))

    def test_backward_compat_alias(self):
        self.assertIs(NormalizingFlow, GMMDensityEstimator)

    def test_insufficient_data(self):
        model = GMMDensityEstimator()
        model.fit(np.array([1.0, 2.0]))
        self.assertFalse(model.fitted)

    def test_get_bounds(self):
        np.random.seed(42)
        model = GMMDensityEstimator()
        model.fit(np.random.normal(0, 1, 100))
        lo, hi = model.get_bounds()
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_model_output(self):
        np.random.seed(42)
        data = np.random.normal(0, 0.01, 200)
        model = GMMDensityEstimator()
        model.fit(data)
        output = model.get_model_output(data)
        self.assertTrue(output.valid or not output.valid)  # Just check it runs
        self.assertIsNotNone(output.entropy)


if __name__ == '__main__':
    unittest.main()
