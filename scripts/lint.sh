#!/bin/bash

# Enhanced SHAP - Code Quality Script
# ====================================
# Runs linting, formatting, and type checking

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

run_black() {
    log_info "Running Black formatter..."
    check_dependencies "black"
    black src tests || log_error "Black formatting failed"
    log_success "Black formatting completed"
}

run_ruff() {
    local fix_mode="${1:-check}"
    
    log_info "Running Ruff linter..."
    check_dependencies "ruff"
    
    if [[ "$fix_mode" == "fix" ]]; then
        ruff check src tests --fix || log_error "Ruff linting failed"
    else
        ruff check src tests || log_error "Ruff linting failed"
    fi
    
    log_success "Ruff linting completed"
}

run_mypy() {
    log_info "Running MyPy type checker..."
    check_dependencies "mypy"
    mypy src || log_error "MyPy type checking failed"
    log_success "MyPy type checking completed"
}

run_all_checks() {
    local fix_mode="${1:-check}"
    
    log_info "Running all code quality checks..."
    
    run_black
    run_ruff "$fix_mode"
    run_mypy
    
    log_success "All code quality checks completed"
}

show_help() {
    echo "Usage: scripts/lint.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  all       - Run all checks (black, ruff, mypy)"
    echo "  black     - Run Black formatter"
    echo "  ruff      - Run Ruff linter"
    echo "  mypy      - Run MyPy type checker"
    echo ""
    echo "Options:"
    echo "  --fix     - Auto-fix issues where possible (for ruff)"
    echo ""
    echo "Examples:"
    echo "  scripts/lint.sh all"
    echo "  scripts/lint.sh all --fix"
    echo "  scripts/lint.sh ruff --fix"
}

# Parse arguments
case "${1:-all}" in
    black)
        run_black
        ;;
    ruff)
        if [[ "${2:-}" == "--fix" ]]; then
            run_ruff "fix"
        else
            run_ruff "check"
        fi
        ;;
    mypy)
        run_mypy
        ;;
    all)
        if [[ "${2:-}" == "--fix" ]]; then
            run_all_checks "fix"
        else
            run_all_checks "check"
        fi
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        log_error "Unknown command: $1. Use --help for usage information."
        ;;
esac