import unittest
import shap_enhanced
from shap_enhanced import BaseExplainer, explainers, tools


class TestInit(unittest.TestCase):
    def test_package_import(self):
        # Test that the main package can be imported
        self.assertIsNotNone(shap_enhanced)

    def test_version_exists(self):
        # Test that version is defined
        self.assertTrue(hasattr(shap_enhanced, '__version__'))
        self.assertEqual(shap_enhanced.__version__, "0.0.1")

    def test_base_explainer_import(self):
        # Test that BaseExplainer can be imported from main package
        self.assertIsNotNone(BaseExplainer)

    def test_explainers_import(self):
        # Test that explainers module can be imported
        self.assertIsNotNone(explainers)

    def test_tools_import(self):
        # Test that tools module can be imported
        self.assertIsNotNone(tools)

    def test_all_exports(self):
        # Test that __all__ contains expected exports
        expected = ["explainers", "tools", "BaseExplainer"]
        self.assertEqual(shap_enhanced.__all__, expected)


if __name__ == '__main__':
    unittest.main()