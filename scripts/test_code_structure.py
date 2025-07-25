#!/usr/bin/env python3

import sys
import os
import ast
import importlib.util

# Add the project root to Python path so we can import shap_enhanced
sys.path.insert(0, '/home/niyang/repos/enhanced_shap/src')

def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_code_structure():
    """Test the structural integrity of refactored code"""
    print("🔍 Testing code structure and syntax...")
    
    # Test explainer files
    explainer_files = [
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/SCSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/ECSHAP.py", 
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/ERSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/ESSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/hSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/LatentSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/MBSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/RLSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/SurroSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/TimeSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/CMSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/CASHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/AttnSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/BSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/ABSHAP.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/explainers/SPSHAP.py",
    ]
    
    # Test algorithm files
    algorithm_files = [
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/coalition_sampling.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/masking.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/model_evaluation.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/normalization.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/data_processing.py",
        "/home/niyang/repos/enhanced_shap/src/shap_enhanced/algorithms/attention.py",
    ]
    
    print("\n📝 Testing explainer file syntax...")
    explainer_success = 0
    for file_path in explainer_files:
        file_name = os.path.basename(file_path)
        is_valid, error = validate_python_syntax(file_path)
        if is_valid:
            print(f"  ✅ {file_name}")
            explainer_success += 1
        else:
            print(f"  ❌ {file_name}: {error}")
    
    print("\n🔧 Testing algorithm file syntax...")
    algorithm_success = 0
    for file_path in algorithm_files:
        file_name = os.path.basename(file_path)
        is_valid, error = validate_python_syntax(file_path)
        if is_valid:
            print(f"  ✅ {file_name}")
            algorithm_success += 1
        else:
            print(f"  ❌ {file_name}: {error}")
    
    print(f"\n📊 Syntax Validation Results:")
    print(f"   Explainers: {explainer_success}/{len(explainer_files)} ({explainer_success/len(explainer_files)*100:.1f}%)")
    print(f"   Algorithms: {algorithm_success}/{len(algorithm_files)} ({algorithm_success/len(algorithm_files)*100:.1f}%)")
    
    # Test for common refactoring patterns
    print("\n🔎 Testing for common refactoring patterns...")
    
    patterns_found = 0
    total_patterns = 0
    
    pattern_checks = [
        ("process_inputs usage", "process_inputs"),
        ("ModelEvaluator usage", "ModelEvaluator"),
        ("AdditivityNormalizer usage", "AdditivityNormalizer"),
        ("BaseMasker inheritance", "BaseMasker"),
        ("BaseExplainer inheritance", "BaseExplainer"),
    ]
    
    for pattern_name, pattern in pattern_checks:
        total_patterns += 1
        found_count = 0
        
        for file_path in explainer_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                if pattern in content:
                    found_count += 1
            except:
                continue
        
        if found_count > 0:
            print(f"  ✅ {pattern_name}: found in {found_count} files")
            patterns_found += 1
        else:
            print(f"  ⚠️  {pattern_name}: not found in any files")
    
    print(f"\n📊 Pattern Analysis Results:")
    print(f"   Common patterns: {patterns_found}/{total_patterns} patterns found")
    
    # Check if files exist and are non-empty
    print(f"\n📁 File existence check...")
    all_files = explainer_files + algorithm_files
    existing_files = 0
    
    for file_path in all_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            existing_files += 1
        else:
            print(f"  ❌ Missing or empty: {os.path.basename(file_path)}")
    
    print(f"   Files: {existing_files}/{len(all_files)} files exist and are non-empty")
    
    return explainer_success, len(explainer_files), algorithm_success, len(algorithm_files)

if __name__ == "__main__":
    explainer_success, explainer_total, algo_success, algo_total = test_code_structure()
    
    print(f"\n🎯 Final Results:")
    print(f"   Explainer syntax: {explainer_success}/{explainer_total} ({explainer_success/explainer_total*100:.1f}%)")
    print(f"   Algorithm syntax: {algo_success}/{algo_total} ({algo_success/algo_total*100:.1f}%)")
    
    if explainer_success == explainer_total and algo_success == algo_total:
        print("🎉 All code structure tests passed! Refactoring appears to be syntactically correct.")
        print("💡 Note: Full functional testing requires PyTorch installation.")
        sys.exit(0)
    else:
        print("⚠️  Some syntax errors found. See details above.")
        sys.exit(1)