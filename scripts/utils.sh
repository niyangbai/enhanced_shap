#!/bin/bash

# Enhanced SHAP - Shared Utilities
# =================================
# Common functions and utilities used across scripts

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if required dependencies are installed
check_dependencies() {
    local deps=("$@")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency '$dep' not found. Please install it first."
        fi
    done
}

# Extract version from _version.py
extract_version() {
    if [[ ! -f "src/shap_enhanced/_version.py" ]]; then
        log_error "_version.py not found. Are you in the correct directory?"
    fi
    
    local version
    version=$(grep -Po '(?<=^__version__ = ")[^"]+' src/shap_enhanced/_version.py 2>/dev/null) || {
        log_error "Could not extract version from _version.py"
    }
    
    if [[ -z "$version" ]]; then
        log_error "Version string is empty"
    fi
    
    echo "$version"
}

# Check git working directory status
check_git_status() {
    if [[ -n "$(git status --porcelain)" ]]; then
        log_warning "Working directory is not clean. Uncommitted changes detected:"
        git status --short
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Aborted by user"
        fi
    fi
}

# Check if git tag already exists
check_git_tag() {
    local tag="$1"
    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_error "Tag '$tag' already exists. Please update the version."
    fi
}