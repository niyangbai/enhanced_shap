#!/bin/bash

# Enhanced SHAP - Documentation Script
# =====================================
# Builds and manages project documentation

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

build_docs() {
    log_info "Building documentation..."
    
    # Check dependencies
    check_dependencies "sphinx-build"
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
    
    # Change to docs directory and build
    cd docs
    sphinx-build -b html source build/html || log_error "Documentation build failed"
    cd ..
    
    log_success "Documentation built successfully"
    log_info "Documentation available at: docs/build/html/index.html"
}

generate_api_docs() {
    log_info "Generating API documentation..."
    
    # Check dependencies
    check_dependencies "sphinx-autogen"
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
    
    # Generate API docs
    cd docs
    sphinx-autogen source/shap_enhanced_api.rst || log_error "API documentation generation failed"
    cd ..
    
    log_success "API documentation generated"
}

clean_docs() {
    log_info "Cleaning documentation build..."
    
    if [[ -d "docs/build" ]]; then
        rm -rf docs/build
        log_success "Documentation build cleaned"
    else
        log_info "No documentation build found to clean"
    fi
}

serve_docs() {
    local port="${1:-8000}"
    
    if [[ ! -d "docs/build/html" ]]; then
        log_info "Documentation not found. Building first..."
        build_docs
    fi
    
    log_info "Serving documentation on http://localhost:$port"
    cd docs/build/html
    python3 -m http.server "$port"
}

show_help() {
    echo "Usage: scripts/docs.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build     - Build HTML documentation"
    echo "  api       - Generate API documentation"
    echo "  clean     - Clean documentation build"
    echo "  serve     - Serve documentation locally (default port: 8000)"
    echo "  all       - Generate API docs and build HTML"
    echo ""
    echo "Examples:"
    echo "  scripts/docs.sh build"
    echo "  scripts/docs.sh serve 8080"
    echo "  scripts/docs.sh all"
}

# Parse arguments
case "${1:-build}" in
    build)
        build_docs
        ;;
    api)
        generate_api_docs
        ;;
    clean)
        clean_docs
        ;;
    serve)
        serve_docs "${2:-8000}"
        ;;
    all)
        generate_api_docs
        build_docs
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        log_error "Unknown command: $1. Use --help for usage information."
        ;;
esac