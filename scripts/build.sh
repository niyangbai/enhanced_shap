#!/bin/bash

# Enhanced SHAP - Build Script
# =============================
# Builds Python package (wheel and sdist)

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

build_package() {
    log_info "Building Python package (wheel and sdist)..."
    
    # Check dependencies
    check_dependencies "python3"
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info/
    
    # Install build dependencies if needed
    if ! python3 -c "import build" 2>/dev/null; then
        log_info "Installing build dependencies..."
        pip install build || log_error "Failed to install build dependencies"
    fi
    
    # Build package
    python3 -m build || log_error "Failed to build package"
    
    # Verify build artifacts
    if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist/)" ]]; then
        log_error "No build artifacts found in dist/"
    fi
    
    log_success "Package built successfully"
    log_info "Build artifacts:"
    ls -la dist/
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    build_package
fi