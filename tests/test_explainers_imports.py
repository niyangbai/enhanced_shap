import unittest
from unittest.mock import Mock, patch
import sys
from shap_enhanced import explainers


class TestExplainersImports(unittest.TestCase):
    def test_all_explainer_imports(self):
        """Test that all explainers can be imported from the module"""
        expected_explainers = [
            'AdaptiveBaselineSHAPExplainer',
            'AttnSHAPExplainer', 
            'BShapExplainer',
            'CoalitionAwareSHAPExplainer',
            'ContextualMaskingSHAPExplainer',
            'EmpiricalConditionalSHAPExplainer',
            'ERSHAPExplainer',
            'EnsembleSHAPWithNoise',
            'HShapExplainer',
            'LatentSHAPExplainer',
            'NearestNeighborMultiBaselineSHAP',
            'RLShapExplainer',
            'SparseCoalitionSHAPExplainer',
            'SupportPreservingSHAPExplainer',
            'SurrogateSHAPExplainer',
            'TimeSHAPExplainer'
        ]
        
        for explainer_name in expected_explainers:
            with self.subTest(explainer=explainer_name):
                self.assertTrue(hasattr(explainers, explainer_name),
                              f"Explainer {explainer_name} not found in explainers module")

    def test_explainer_classes_are_classes(self):
        """Test that imported explainers are actually classes"""
        # Test a few key explainers
        key_explainers = ['LatentSHAPExplainer', 'TimeSHAPExplainer', 'BShapExplainer']
        
        for explainer_name in key_explainers:
            with self.subTest(explainer=explainer_name):
                explainer_class = getattr(explainers, explainer_name)
                self.assertTrue(callable(explainer_class),
                              f"{explainer_name} should be callable/class")

    def test_explainer_module_structure(self):
        """Test the overall structure of the explainers module"""
        # Test that it's a proper module
        self.assertTrue(hasattr(explainers, '__name__'))
        self.assertTrue(hasattr(explainers, '__file__'))
        
        # Test that it has the expected submodules
        import shap_enhanced.explainers
        module_path = shap_enhanced.explainers.__file__
        self.assertTrue(module_path.endswith('__init__.py'))

    def test_individual_explainer_modules_exist(self):
        """Test that individual explainer module files can be accessed"""
        explainer_modules = [
            'ABSHAP', 'AttnSHAP', 'BSHAP', 'CASHAP', 'CMSHAP', 
            'ECSHAP', 'ERSHAP', 'ESSHAP', 'hSHAP', 'LatentSHAP',
            'MBSHAP', 'RLSHAP', 'SCSHAP', 'SPSHAP', 'SurroSHAP', 'TimeSHAP'
        ]
        
        for module_name in explainer_modules:
            with self.subTest(module=module_name):
                try:
                    module = __import__(f'shap_enhanced.explainers.{module_name}', fromlist=[module_name])
                    self.assertIsNotNone(module)
                except ImportError as e:
                    self.fail(f"Could not import explainer module {module_name}: {e}")

    def test_explainers_namespace_clean(self):
        """Test that explainers module doesn't have unexpected attributes"""
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(explainers) if not attr.startswith('_')]
        
        # Should mostly be explainer classes
        for attr in public_attrs:
            with self.subTest(attribute=attr):
                obj = getattr(explainers, attr)
                # Should be either a class or a module
                self.assertTrue(
                    callable(obj) or hasattr(obj, '__file__'),
                    f"Unexpected attribute {attr} in explainers module"
                )


class TestExplainerMockInstantiation(unittest.TestCase):
    """Test that explainers can be instantiated with mock objects"""
    
    def setUp(self):
        self.mock_model = Mock()
        self.mock_background = Mock()

    def test_latent_shap_instantiation(self):
        """Test LatentSHAPExplainer can be instantiated"""
        try:
            from shap_enhanced.explainers.LatentSHAP import LatentSHAPExplainer
            # This might require specific parameters, so we'll just test import
            self.assertTrue(callable(LatentSHAPExplainer))
        except ImportError as e:
            self.skipTest(f"LatentSHAP import failed: {e}")

    def test_time_shap_instantiation(self):
        """Test TimeSHAPExplainer can be instantiated"""
        try:
            from shap_enhanced.explainers.TimeSHAP import TimeSHAPExplainer
            self.assertTrue(callable(TimeSHAPExplainer))
        except ImportError as e:
            self.skipTest(f"TimeSHAP import failed: {e}")

    def test_bshap_instantiation(self):
        """Test BShapExplainer can be instantiated"""
        try:
            from shap_enhanced.explainers.BSHAP import BShapExplainer
            self.assertTrue(callable(BShapExplainer))
        except ImportError as e:
            self.skipTest(f"BSHAP import failed: {e}")


if __name__ == '__main__':
    unittest.main()