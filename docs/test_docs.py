#!/usr/bin/env python3
"""
Test script to validate Sphinx documentation build.
"""
import os
import sys
import subprocess
from pathlib import Path

def test_sphinx_build():
    """Test that Sphinx can build the documentation without critical errors."""
    docs_dir = Path(__file__).parent
    build_dir = docs_dir / "build" / "html"
    
    print("Testing Sphinx documentation build...")
    
    # Change to docs directory
    os.chdir(docs_dir)
    
    # Clean previous build
    if build_dir.exists():
        subprocess.run(["make", "clean"], check=True)
    
    # Build documentation
    result = subprocess.run(["make", "html"], capture_output=True, text=True)
    
    # Check build result
    if result.returncode != 0:
        print("❌ Documentation build FAILED")
        print("STDERR:", result.stderr)
        return False
    
    # Check that main files exist
    required_files = [
        "index.html",
        "shap_enhanced_api.html", 
        "genindex.html",
        "py-modindex.html"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (build_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check for excessive warnings (more than 50 is concerning)
    warning_count = result.stderr.count("WARNING:")
    print(f"📝 Build completed with {warning_count} warnings")
    
    if warning_count > 50:
        print("⚠️  High number of warnings - consider reviewing documentation")
    
    print("✅ Documentation build SUCCESS")
    print(f"📄 Documentation available at: {build_dir / 'index.html'}")
    
    return True

if __name__ == "__main__":
    success = test_sphinx_build()
    sys.exit(0 if success else 1)