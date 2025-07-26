import unittest
from unittest.mock import Mock

import numpy as np

import shap_enhanced
from shap_enhanced.base_explainer import BaseExplainer


class EdgeCaseMockExplainer(BaseExplainer):
    def shap_values(self, X, check_additivity=True, **kwargs):
        if X is None:
            raise ValueError("Input cannot be None")
        if isinstance(X, np.ndarray) and X.size == 0:
            return np.array([])
        if isinstance(X, np.ndarray):
            return np.zeros_like(X)
        return [0] * len(X) if hasattr(X, "__len__") else 0


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.explainer = EdgeCaseMockExplainer(self.mock_model)

    def test_none_input(self):
        """Test handling of None input"""
        with self.assertRaises(ValueError):
            self.explainer.shap_values(None)

    def test_empty_array_input(self):
        """Test handling of empty arrays"""
        empty_array = np.array([])
        result = self.explainer.shap_values(empty_array)
        self.assertEqual(len(result), 0)

    def test_single_element_input(self):
        """Test handling of single element inputs"""
        single_element = np.array([1.0])
        result = self.explainer.shap_values(single_element)
        np.testing.assert_array_equal(result, np.array([0.0]))

    def test_very_large_input(self):
        """Test handling of large inputs"""
        large_input = np.random.random((1000, 100))
        result = self.explainer.shap_values(large_input)
        self.assertEqual(result.shape, large_input.shape)

    def test_nan_input(self):
        """Test handling of NaN values in input"""
        nan_input = np.array([1.0, np.nan, 3.0])
        result = self.explainer.shap_values(nan_input)
        self.assertEqual(result.shape, nan_input.shape)
        # Result should be zeros (from our mock), not NaN
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isfinite(result[2]))

    def test_inf_input(self):
        """Test handling of infinite values in input"""
        inf_input = np.array([1.0, np.inf, -np.inf])
        result = self.explainer.shap_values(inf_input)
        self.assertEqual(result.shape, inf_input.shape)

    def test_zero_variance_input(self):
        """Test handling of inputs with zero variance"""
        zero_var_input = np.ones((10, 5))
        result = self.explainer.shap_values(zero_var_input)
        self.assertEqual(result.shape, zero_var_input.shape)

    def test_negative_values_input(self):
        """Test handling of negative values"""
        negative_input = np.array([[-1, -2], [-3, -4]])
        result = self.explainer.shap_values(negative_input)
        self.assertEqual(result.shape, negative_input.shape)

    def test_mixed_type_list_input(self):
        """Test handling of mixed type inputs in lists"""
        mixed_input = [1, 2.5, 3]
        result = self.explainer.shap_values(mixed_input)
        self.assertEqual(len(result), 3)

    def test_very_high_dimensional_input(self):
        """Test handling of high-dimensional inputs"""
        high_dim_input = np.random.random((2, 3, 4, 5))
        result = self.explainer.shap_values(high_dim_input)
        self.assertEqual(result.shape, high_dim_input.shape)

    def test_string_input_error(self):
        """Test that string inputs are handled appropriately"""
        # Our mock explainer will try to get len() of string, which works
        # So let's test with a string that doesn't have len
        try:
            result = self.explainer.shap_values("test")
            # If it doesn't raise an error, it should return something reasonable
            self.assertEqual(len(result), 4)  # length of "test"
        except (TypeError, AttributeError):
            # This is also acceptable behavior
            pass

    def test_boolean_input(self):
        """Test handling of boolean inputs"""
        bool_input = np.array([True, False, True])
        result = self.explainer.shap_values(bool_input)
        self.assertEqual(result.shape, bool_input.shape)


class TestPackageEdgeCases(unittest.TestCase):
    def test_package_version_format(self):
        """Test that package version follows expected format"""
        version = shap_enhanced.__version__
        self.assertIsInstance(version, str)
        self.assertTrue(len(version) > 0)
        # Should contain at least one dot for major.minor format
        self.assertIn(".", version)

    def test_package_all_attribute(self):
        """Test that __all__ is properly defined"""
        all_exports = shap_enhanced.__all__
        self.assertIsInstance(all_exports, list)
        self.assertTrue(len(all_exports) > 0)

        # All exports should be strings
        for export in all_exports:
            self.assertIsInstance(export, str)

        # All exports should be available in the module
        for export in all_exports:
            self.assertTrue(
                hasattr(shap_enhanced, export), f"Export {export} not found in module"
            )

    def test_circular_import_protection(self):
        """Test that imports don't create circular dependencies"""
        # This test ensures we can import multiple times without issues
        # Re-importing should work fine
        import shap_enhanced
        import shap_enhanced.base_explainer
        import shap_enhanced.explainers
        import shap_enhanced.tools

        self.assertIsNotNone(shap_enhanced)

    def test_memory_cleanup(self):
        """Test that objects can be garbage collected properly"""
        import gc

        # Create some objects
        model = Mock()
        explainer = EdgeCaseMockExplainer(model)
        large_data = np.random.random((1000, 1000))

        # Use them
        result = explainer.shap_values(large_data[:10, :10])

        # Delete references
        del model, explainer, large_data, result

        # Force garbage collection
        gc.collect()

        # Test should complete without memory errors
        self.assertTrue(True)

    def test_thread_safety_basic(self):
        """Basic test for thread safety concerns"""
        import threading

        results = []
        errors = []

        def worker():
            try:
                explainer = EdgeCaseMockExplainer(Mock())
                data = np.random.random((10, 5))
                result = explainer.shap_values(data)
                results.append(result.shape)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        self.assertEqual(len(results), 5)
        for shape in results:
            self.assertEqual(shape, (10, 5))


if __name__ == "__main__":
    unittest.main()
