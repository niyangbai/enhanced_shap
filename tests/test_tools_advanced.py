import unittest
import numpy as np
import time
from unittest.mock import patch, Mock
from shap_enhanced.tools.timer import Timer
from shap_enhanced.tools.datasets import generate_synthetic_seqregression, generate_synthetic_tabular


class TestTimer(unittest.TestCase):
    def test_timer_context_manager(self):
        with Timer("test", verbose=False) as timer:
            time.sleep(0.01)
        
        self.assertIsNotNone(timer.elapsed)
        self.assertGreaterEqual(timer.elapsed, 0.01)
        self.assertLess(timer.elapsed, 0.1)

    def test_timer_verbose_output(self):
        with patch('builtins.print') as mock_print:
            with Timer("test task", verbose=True):
                time.sleep(0.01)
            
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn("[Timer]", call_args)
            self.assertIn("test task", call_args)
            self.assertIn("seconds", call_args)

    def test_timer_silent_mode(self):
        with patch('builtins.print') as mock_print:
            with Timer("silent test", verbose=False) as timer:
                pass
            
            mock_print.assert_not_called()
            self.assertIsNotNone(timer.elapsed)

    def test_timer_default_label(self):
        with Timer(verbose=False) as timer:
            pass
        
        self.assertEqual(timer.label, "")
        self.assertIsNotNone(timer.elapsed)

    def test_timer_exception_handling(self):
        with self.assertRaises(ValueError):
            with Timer("error test", verbose=False) as timer:
                raise ValueError("test error")
        
        self.assertIsNotNone(timer.elapsed)


class TestDatasets(unittest.TestCase):
    def test_generate_synthetic_seqregression_default(self):
        X, y = generate_synthetic_seqregression()
        
        self.assertEqual(X.shape, (200, 10, 3))
        self.assertEqual(y.shape, (200,))
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isfinite(y).all())

    def test_generate_synthetic_seqregression_custom_params(self):
        X, y = generate_synthetic_seqregression(seq_len=15, n_features=5, n_samples=100, seed=42)
        
        self.assertEqual(X.shape, (100, 15, 5))
        self.assertEqual(y.shape, (100,))
        
        # Test reproducibility
        X2, y2 = generate_synthetic_seqregression(seq_len=15, n_features=5, n_samples=100, seed=42)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(y, y2)

    def test_generate_synthetic_seqregression_target_function(self):
        # Test that target follows sin(sum(first_feature))
        X, y = generate_synthetic_seqregression(n_samples=10, seed=123)
        
        expected_y_base = np.sin(X[:, :, 0].sum(axis=1))
        # Allow for noise in comparison
        correlation = np.corrcoef(y, expected_y_base)[0, 1]
        self.assertGreater(correlation, 0.8)

    def test_generate_synthetic_tabular_linear(self):
        X, y, w = generate_synthetic_tabular(n_samples=100, n_features=5, model_type="linear", sparse=False)
        
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(w.shape, (5,))
        
        # For linear model, y should equal X @ w
        expected_y = X @ w
        np.testing.assert_array_almost_equal(y, expected_y)

    def test_generate_synthetic_tabular_nonlinear(self):
        X, y, w = generate_synthetic_tabular(n_samples=50, n_features=3, model_type="nonlinear", sparse=False)
        
        self.assertEqual(X.shape, (50, 3))
        self.assertEqual(y.shape, (50,))
        self.assertEqual(w.shape, (3,))
        
        # For nonlinear, y should not simply equal X @ w
        linear_y = X @ w
        self.assertFalse(np.allclose(y, linear_y))

    def test_generate_synthetic_tabular_sparse(self):
        X, y, w = generate_synthetic_tabular(n_samples=100, n_features=10, sparse=True, sparsity=0.9)
        
        # Check that approximately 90% of elements are zero
        zero_fraction = (X == 0).mean()
        self.assertGreater(zero_fraction, 0.8)
        self.assertLess(zero_fraction, 0.95)

    def test_generate_synthetic_tabular_reproducibility(self):
        X1, y1, w1 = generate_synthetic_tabular(random_seed=456)
        X2, y2, w2 = generate_synthetic_tabular(random_seed=456)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(w1, w2)

    def test_generate_synthetic_tabular_weight_range(self):
        _, _, w = generate_synthetic_tabular(n_features=20, random_seed=789)
        
        # Weights should be in range [-2, 3]
        self.assertTrue((w >= -2).all())
        self.assertTrue((w <= 3).all())

    def test_invalid_model_type(self):
        # Test that function handles invalid model_type gracefully
        X, y, w = generate_synthetic_tabular(model_type="invalid")
        
        # Should default to linear behavior or handle gracefully
        self.assertEqual(X.shape[1], 5)  # default n_features
        self.assertEqual(len(y), 500)   # default n_samples


if __name__ == '__main__':
    unittest.main()