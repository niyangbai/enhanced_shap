import unittest
from unittest.mock import Mock

import numpy as np

from shap_enhanced.base_explainer import BaseExplainer


class MockExplainer(BaseExplainer):
    def shap_values(self, X, check_additivity=True, **kwargs):
        # Simple mock implementation that returns zeros
        if isinstance(X, np.ndarray):
            return np.zeros_like(X)
        return [0] * len(X)


class TestBaseExplainer(unittest.TestCase):
    def test_base_explainer_init(self):
        model = Mock()
        background = np.array([1, 2, 3])
        explainer = MockExplainer(model, background)

        self.assertIs(explainer.model, model)
        self.assertTrue(np.array_equal(explainer.background, background))

    def test_base_explainer_explain(self):
        model = Mock()
        explainer = MockExplainer(model)
        X = np.array([[1, 2], [3, 4]])

        result = explainer.explain(X)
        expected = np.zeros_like(X)

        self.assertTrue(np.array_equal(result, expected))

    def test_base_explainer_call(self):
        model = Mock()
        explainer = MockExplainer(model)
        X = np.array([[1, 2], [3, 4]])

        result = explainer(X)
        expected = np.zeros_like(X)

        self.assertTrue(np.array_equal(result, expected))

    def test_expected_value_default(self):
        model = Mock()
        explainer = MockExplainer(model)

        self.assertIsNone(explainer.expected_value)

    def test_expected_value_with_attribute(self):
        model = Mock()
        explainer = MockExplainer(model)
        explainer._expected_value = 0.5

        self.assertEqual(explainer.expected_value, 0.5)


if __name__ == "__main__":
    unittest.main()
