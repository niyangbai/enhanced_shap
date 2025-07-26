import unittest

import numpy as np

from shap_enhanced.tools import comparison, evaluation, predefined_models, visulization


class TestToolsModules(unittest.TestCase):
    def test_evaluation_module_import(self):
        """Test that evaluation module imports successfully"""
        self.assertIsNotNone(evaluation)
        self.assertTrue(hasattr(evaluation, "__name__"))

    def test_comparison_module_import(self):
        """Test that comparison module imports successfully"""
        self.assertIsNotNone(comparison)
        self.assertTrue(hasattr(comparison, "__name__"))

    def test_predefined_models_module_import(self):
        """Test that predefined_models module imports successfully"""
        self.assertIsNotNone(predefined_models)
        self.assertTrue(hasattr(predefined_models, "__name__"))

    def test_visulization_module_import(self):
        """Test that visulization module imports successfully"""
        self.assertIsNotNone(visulization)
        self.assertTrue(hasattr(visulization, "__name__"))

    def test_tools_module_structure(self):
        """Test the overall structure of tools modules"""
        import shap_enhanced.tools

        # Test main tools module
        self.assertTrue(hasattr(shap_enhanced.tools, "__name__"))
        self.assertTrue(hasattr(shap_enhanced.tools, "__file__"))

        # Check that it contains expected submodules
        expected_modules = [
            "timer",
            "datasets",
            "evaluation",
            "comparison",
            "predefined_models",
            "visulization",
        ]

        for module_name in expected_modules:
            with self.subTest(module=module_name):
                self.assertTrue(
                    hasattr(shap_enhanced.tools, module_name),
                    f"tools.{module_name} not found",
                )

    def test_tools_init_imports(self):
        """Test that tools __init__.py properly imports submodules"""
        from shap_enhanced import tools

        # These should be accessible from tools
        modules_to_check = ["timer", "datasets", "evaluation", "comparison"]

        for module_name in modules_to_check:
            with self.subTest(module=module_name):
                self.assertTrue(
                    hasattr(tools, module_name),
                    f"tools.{module_name} not accessible from main tools module",
                )


class TestToolsIntegration(unittest.TestCase):
    def test_datasets_timer_integration(self):
        """Test using timer with datasets generation"""
        from shap_enhanced.tools.datasets import generate_synthetic_tabular
        from shap_enhanced.tools.timer import Timer

        with Timer("data generation", verbose=False) as timer:
            X, y, w = generate_synthetic_tabular(n_samples=100)

        self.assertGreater(timer.elapsed, 0)
        self.assertEqual(X.shape, (100, 5))

    def test_datasets_with_different_random_states(self):
        """Test datasets with various random state configurations"""
        from shap_enhanced.tools.datasets import generate_synthetic_seqregression

        # Test different seed values
        seeds = [0, 42, 123, 999]
        results = []

        for seed in seeds:
            X, y = generate_synthetic_seqregression(n_samples=50, seed=seed)
            results.append((X, y))

        # All should have same shape
        for X, y in results:
            self.assertEqual(X.shape, (50, 10, 3))
            self.assertEqual(y.shape, (50,))

        # Different seeds should produce different data
        X1, y1 = results[0]
        X2, y2 = results[1]
        self.assertFalse(np.allclose(X1, X2))
        self.assertFalse(np.allclose(y1, y2))

    def test_error_handling_in_modules(self):
        """Test error handling in various tools modules"""
        from shap_enhanced.tools.datasets import generate_synthetic_tabular

        # Test with edge case parameters
        try:
            X, y, w = generate_synthetic_tabular(n_samples=1, n_features=1)
            self.assertEqual(X.shape, (1, 1))
        except Exception as e:
            self.fail(f"Minimal parameters should work: {e}")

        # Test with large sparsity
        try:
            X, y, w = generate_synthetic_tabular(sparsity=0.99, sparse=True)
            sparsity_actual = (X == 0).mean()
            self.assertGreater(sparsity_actual, 0.95)
        except Exception as e:
            self.fail(f"High sparsity should work: {e}")


class TestToolsPerformance(unittest.TestCase):
    def test_timer_accuracy(self):
        """Test that timer measurements are reasonably accurate"""
        import time

        from shap_enhanced.tools.timer import Timer

        # Test short duration
        with Timer(verbose=False) as timer:
            time.sleep(0.05)

        self.assertGreater(timer.elapsed, 0.04)
        self.assertLess(timer.elapsed, 0.1)

        # Test very short duration
        with Timer(verbose=False) as timer:
            pass  # Minimal operation

        self.assertGreater(timer.elapsed, 0)
        self.assertLess(timer.elapsed, 0.01)

    def test_datasets_performance_characteristics(self):
        """Test performance characteristics of dataset generation"""
        from shap_enhanced.tools.datasets import (
            generate_synthetic_tabular,
        )
        from shap_enhanced.tools.timer import Timer

        # Test tabular generation scales reasonably
        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            with Timer(verbose=False) as timer:
                X, y, w = generate_synthetic_tabular(n_samples=size, n_features=10)
            times.append(timer.elapsed)

        # Should scale roughly linearly or better
        self.assertLess(times[1] / times[0], 10)  # Not more than 10x slower for 5x data
        self.assertLess(
            times[2] / times[0], 20
        )  # Not more than 20x slower for 10x data

    def test_memory_usage_datasets(self):
        """Test memory characteristics of dataset generation"""
        from shap_enhanced.tools.datasets import generate_synthetic_seqregression

        # Generate reasonably large dataset
        X, y = generate_synthetic_seqregression(
            n_samples=1000, seq_len=50, n_features=20
        )

        # Check memory usage is reasonable
        expected_size = 1000 * 50 * 20 * 8  # rough bytes for float64
        actual_size = X.nbytes + y.nbytes

        self.assertLess(actual_size, expected_size * 2)  # Within 2x of expected


class TestToolsCompatibility(unittest.TestCase):
    def test_numpy_compatibility(self):
        """Test compatibility with different numpy operations"""
        from shap_enhanced.tools.datasets import generate_synthetic_tabular

        X, y, w = generate_synthetic_tabular()

        # Test various numpy operations
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue(np.isfinite(w).all())

        # Test statistical operations
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)

        self.assertEqual(len(mean_X), X.shape[1])
        self.assertEqual(len(std_X), X.shape[1])

    def test_data_type_consistency(self):
        """Test that generated data has consistent types"""
        from shap_enhanced.tools.datasets import (
            generate_synthetic_seqregression,
            generate_synthetic_tabular,
        )

        # Test tabular data types
        X_tab, y_tab, w_tab = generate_synthetic_tabular()
        self.assertEqual(X_tab.dtype, np.float64)
        self.assertEqual(y_tab.dtype, np.float64)
        self.assertEqual(w_tab.dtype, np.float64)

        # Test sequential data types
        X_seq, y_seq = generate_synthetic_seqregression()
        self.assertEqual(X_seq.dtype, np.float64)
        self.assertEqual(y_seq.dtype, np.float64)

    def test_random_state_independence(self):
        """Test that different calls don't interfere with each other"""
        from shap_enhanced.tools.datasets import generate_synthetic_tabular

        # Generate data with same seed multiple times
        results1 = []
        results2 = []

        for _ in range(3):
            X1, y1, w1 = generate_synthetic_tabular(random_seed=42)
            results1.append((X1, y1, w1))

        for _ in range(3):
            X2, y2, w2 = generate_synthetic_tabular(random_seed=42)
            results2.append((X2, y2, w2))

        # Same seed should give same results
        for i in range(3):
            np.testing.assert_array_equal(results1[i][0], results2[i][0])
            np.testing.assert_array_equal(results1[i][1], results2[i][1])
            np.testing.assert_array_equal(results1[i][2], results2[i][2])


if __name__ == "__main__":
    unittest.main()
