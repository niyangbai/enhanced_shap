import unittest
from unittest.mock import Mock

import numpy as np

from shap_enhanced.base_explainer import BaseExplainer


class ComplexMockExplainer(BaseExplainer):
    def __init__(self, model, background=None, fail_on_check=False):
        super().__init__(model, background)
        self.fail_on_check = fail_on_check
        self._expected_value = None

    def shap_values(self, X, check_additivity=True, **kwargs):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            result = np.random.random(X.shape) * 0.1
        elif isinstance(X, list):
            result = [
                (
                    np.random.random(len(x)) * 0.1
                    if isinstance(x, list | np.ndarray)
                    else 0.1
                )
                for x in X
            ]
        else:
            result = np.array([0.1])

        if check_additivity and self.fail_on_check:
            raise ValueError("Additivity check failed")

        return result


class TestBaseExplainerAdvanced(unittest.TestCase):
    def test_abstract_class_cannot_instantiate(self):
        # Test that BaseExplainer cannot be instantiated directly
        with self.assertRaises(TypeError):
            BaseExplainer(Mock())

    def test_subclass_must_implement_shap_values(self):
        # Test that subclass without shap_values implementation fails
        class IncompleteExplainer(BaseExplainer):
            pass

        with self.assertRaises(TypeError):
            IncompleteExplainer(Mock())

    def test_different_input_types(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        # Test numpy array
        X_array = np.array([[1, 2, 3], [4, 5, 6]])
        result_array = explainer.shap_values(X_array)
        self.assertEqual(result_array.shape, X_array.shape)

        # Test 1D array
        X_1d = np.array([1, 2, 3])
        result_1d = explainer.shap_values(X_1d)
        self.assertEqual(result_1d.shape, (1, 3))

        # Test list
        X_list = [1, 2, 3]
        result_list = explainer.shap_values(X_list)
        self.assertEqual(len(result_list), 3)

    def test_kwargs_passing(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        # Test that kwargs are passed through
        result = explainer.shap_values(
            np.array([1, 2]), custom_param=True, another_param="test"
        )
        self.assertIsNotNone(result)

    def test_check_additivity_parameter(self):
        model = Mock()
        explainer = ComplexMockExplainer(model, fail_on_check=True)

        # Test that check_additivity=False prevents error
        result = explainer.shap_values(np.array([1, 2]), check_additivity=False)
        self.assertIsNotNone(result)

        # Test that check_additivity=True triggers error in our mock
        with self.assertRaises(ValueError):
            explainer.shap_values(np.array([1, 2]), check_additivity=True)

    def test_explain_method_kwargs(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        X = np.array([[1, 2], [3, 4]])
        result = explainer.explain(X, custom_kwarg="value")
        self.assertEqual(result.shape, X.shape)

    def test_call_method_kwargs(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        X = np.array([[1, 2], [3, 4]])
        result = explainer(X, another_kwarg=42)
        self.assertEqual(result.shape, X.shape)

    def test_expected_value_setter(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        # Test setting and getting expected value
        explainer._expected_value = 1.5
        self.assertEqual(explainer.expected_value, 1.5)

        # Test different types
        explainer._expected_value = np.array([1, 2, 3])
        np.testing.assert_array_equal(explainer.expected_value, np.array([1, 2, 3]))

    def test_background_data_types(self):
        model = Mock()

        # Test with numpy array background
        bg_array = np.array([[1, 2], [3, 4]])
        explainer = ComplexMockExplainer(model, bg_array)
        np.testing.assert_array_equal(explainer.background, bg_array)

        # Test with list background
        bg_list = [1, 2, 3]
        explainer2 = ComplexMockExplainer(model, bg_list)
        self.assertEqual(explainer2.background, bg_list)

        # Test with None background
        explainer3 = ComplexMockExplainer(model, None)
        self.assertIsNone(explainer3.background)

    def test_model_attribute_access(self):
        model = Mock()
        model.some_attribute = "test_value"
        model.predict = Mock(return_value=np.array([1, 2, 3]))

        explainer = ComplexMockExplainer(model)

        # Test that model is accessible and functional
        self.assertEqual(explainer.model.some_attribute, "test_value")
        result = explainer.model.predict([1, 2, 3])
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_multiple_calls_consistency(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        X = np.array([[1, 2], [3, 4]])

        # Multiple calls should work consistently
        result1 = explainer.shap_values(X)
        result2 = explainer.explain(X)
        result3 = explainer(X)

        # All should have same shape
        self.assertEqual(result1.shape, result2.shape)
        self.assertEqual(result2.shape, result3.shape)

    def test_edge_case_empty_input(self):
        model = Mock()
        explainer = ComplexMockExplainer(model)

        # Test empty array
        X_empty = np.array([]).reshape(0, 2)
        result = explainer.shap_values(X_empty)
        self.assertEqual(result.shape, X_empty.shape)


if __name__ == "__main__":
    unittest.main()
