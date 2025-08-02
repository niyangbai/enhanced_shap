#!/bin/bash

# Enhanced SHAP - Test Script
# ============================
# Runs comprehensive test suite

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

run_tests() {
    local test_type="${1:-all}"
    
    log_info "Running tests ($test_type)..."
    
    # Check dependencies
    check_dependencies "python3"
    
    # Set PYTHONPATH
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
    
    case "$test_type" in
        "unit")
            log_info "Running unit tests..."
            python -m unittest discover tests/ -v
            ;;
        "coverage")
            log_info "Running tests with coverage..."
            python -m pytest --cov=src --cov-report=html --cov-report=term
            ;;
        "quick")
            log_info "Running quick tests..."
            python -m unittest discover tests/ -v --quiet
            ;;
        "specific")
            if [[ -n "${2:-}" ]]; then
                log_info "Running specific test: $2"
                python -m unittest "$2" -v
            else
                log_error "No specific test provided"
            fi
            ;;
        "all"|*)
            log_info "Running all tests..."
            python -m unittest discover tests/ -v
            ;;
    esac
    
    log_success "Tests completed successfully"
}

show_help() {
    echo "Usage: scripts/test.sh [TYPE] [SPECIFIC_TEST]"
    echo ""
    echo "Test types:"
    echo "  all       - Run all tests (default)"
    echo "  unit      - Run unit tests only"
    echo "  coverage  - Run tests with coverage report"
    echo "  quick     - Run tests quietly"
    echo "  specific  - Run specific test (requires SPECIFIC_TEST)"
    echo ""
    echo "Examples:"
    echo "  scripts/test.sh"
    echo "  scripts/test.sh coverage"
    echo "  scripts/test.sh specific tests.test_explainers.TestExplainers.test_latent_shap"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        run_tests "$@"
        ;;
esac