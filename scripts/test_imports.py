#!/usr/bin/env python3

import sys
import os

# Add the project root to Python path so we can import shap_enhanced
sys.path.insert(0, '/home/niyang/repos/enhanced_shap/src')

def test_imports():
    """Test that all refactored explainers can be imported without errors"""
    print("🧪 Testing imports of refactored explainers...")
    
    success_count = 0
    total_count = 0
    
    explainers_to_import = [
        ("SCSHAP", "shap_enhanced.explainers.SCSHAP", "SparseCoalitionSHAPExplainer"),
        ("ECSHAP", "shap_enhanced.explainers.ECSHAP", "EmpiricalConditionalSHAPExplainer"),
        ("ERSHAP", "shap_enhanced.explainers.ERSHAP", "ERSHAPExplainer"),
        ("ESSHAP", "shap_enhanced.explainers.ESSHAP", "EnsembleSHAPWithNoise"),
        ("hSHAP", "shap_enhanced.explainers.hSHAP", "HShapExplainer"),
        ("LatentSHAP", "shap_enhanced.explainers.LatentSHAP", "LatentSHAPExplainer"),
        ("MBSHAP", "shap_enhanced.explainers.MBSHAP", "NearestNeighborMultiBaselineSHAP"),
        ("RLSHAP", "shap_enhanced.explainers.RLSHAP", "RLShapExplainer"),
        ("SurroSHAP", "shap_enhanced.explainers.SurroSHAP", "SurrogateSHAPExplainer"),
        ("TimeSHAP", "shap_enhanced.explainers.TimeSHAP", "TimeSHAPExplainer"),
        ("CMSHAP", "shap_enhanced.explainers.CMSHAP", "ContextualMaskingSHAPExplainer"),
        ("CASHAP", "shap_enhanced.explainers.CASHAP", "CoalitionAwareSHAPExplainer"),
        ("AttnSHAP", "shap_enhanced.explainers.AttnSHAP", "AttnSHAPExplainer"),
        ("BSHAP", "shap_enhanced.explainers.BSHAP", "BShapExplainer"),
        ("ABSHAP", "shap_enhanced.explainers.ABSHAP", "AdaptiveBaselineSHAPExplainer"),
        ("SPSHAP", "shap_enhanced.explainers.SPSHAP", "SupportPreservingSHAPExplainer"),
    ]
    
    for name, module_path, class_name in explainers_to_import:
        total_count += 1
        try:
            # Import the module
            module = __import__(module_path, fromlist=[class_name])
            
            # Get the class
            explainer_class = getattr(module, class_name)
            
            print(f"  ✅ {name}: {module_path}.{class_name}")
            success_count += 1
            
        except ImportError as e:
            print(f"  ❌ {name}: ImportError - {e}")
        except AttributeError as e:
            print(f"  ❌ {name}: AttributeError - {e}")
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__} - {e}")
    
    print(f"\n📊 Import Results: {success_count}/{total_count} explainers imported successfully")
    
    # Test algorithm imports
    print("\n🔧 Testing algorithm module imports...")
    algorithm_modules = [
        ("coalition_sampling", "shap_enhanced.algorithms.coalition_sampling"),
        ("masking", "shap_enhanced.algorithms.masking"),
        ("model_evaluation", "shap_enhanced.algorithms.model_evaluation"),
        ("normalization", "shap_enhanced.algorithms.normalization"),
        ("data_processing", "shap_enhanced.algorithms.data_processing"),
        ("attention", "shap_enhanced.algorithms.attention"),
    ]
    
    algo_success = 0
    for name, module_path in algorithm_modules:
        try:
            __import__(module_path)
            print(f"  ✅ {name}")
            algo_success += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print(f"\n📊 Algorithm Results: {algo_success}/{len(algorithm_modules)} algorithm modules imported successfully")
    
    return success_count, total_count, algo_success, len(algorithm_modules)

if __name__ == "__main__":
    explainer_success, explainer_total, algo_success, algo_total = test_imports()
    
    print(f"\n🎯 Overall Results:")
    print(f"   Explainers: {explainer_success}/{explainer_total} ({explainer_success/explainer_total*100:.1f}%)")
    print(f"   Algorithms: {algo_success}/{algo_total} ({algo_success/algo_total*100:.1f}%)")
    
    if explainer_success == explainer_total and algo_success == algo_total:
        print("🎉 All imports successful! Refactoring appears to be working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some imports failed. See details above.")
        sys.exit(1)