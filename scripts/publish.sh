#!/bin/bash

# Enhanced SHAP Package Publishing Script
# ========================================
# 
# This script automates the release process for the Enhanced SHAP package.
# It packages the code and publishes to PyPI.
#
# Usage:
#   scripts/publish.sh [--prod] [--help]
#
# Options:
#   --prod         Upload to production PyPI (default: TestPyPI)
#   --help         Show this help message

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
PROD_PYPI=false

# Functions
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

show_help() {
    head -n 15 "$0" | tail -n +2 | sed 's/^# //'
    exit 0
}

check_dependencies() {
    local deps=("python3" "git" "gh" "twine")

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Required dependency '$dep' not found. Please install it first."
        fi
    done
}

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

check_git_tag() {
    local tag="$1"
    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_error "Tag '$tag' already exists. Please update the version in pyproject.toml"
    fi
}


build_package() {
    log_info "Building Python package (wheel and sdist)..."
    
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

upload_package() {
    local repository="$1"
    local repo_name="$2"
    
    log_info "Uploading to $repo_name..."
    
    # Check if twine is available
    if ! command -v twine &> /dev/null; then
        log_error "twine not found. Install with: pip install twine"
    fi
    
    # Upload package
    if [[ "$repository" == "pypi" ]]; then
        twine upload dist/* || log_error "Failed to upload to PyPI"
    else
        twine upload --repository testpypi dist/* || log_error "Failed to upload to TestPyPI"
    fi
    
    log_success "Package uploaded to $repo_name"
}

create_git_tag() {
    local tag="$1"
    local version="$2"
    
    log_info "Creating and pushing git tag: $tag"
    
    # Create annotated tag
    git tag -a "$tag" -m "Release version $version" || log_error "Failed to create git tag"
    
    # Push tag to origin
    git push origin "$tag" || log_error "Failed to push git tag"
    
    log_success "Git tag created and pushed"
}

create_github_release() {
    local tag="$1"
    
    log_info "Creating GitHub release for $tag..."
    
    # Check if gh CLI is authenticated
    if ! gh auth status &>/dev/null; then
        log_error "GitHub CLI not authenticated. Run: gh auth login"
    fi
    
    # Create release with changelog notes
    local release_notes="Release $tag"
    if [[ -f "CHANGELOG.md" ]]; then
        release_notes="$release_notes

See [CHANGELOG.md](https://github.com/$(gh repo view --json owner,name -q '.owner.login + \"/\" + .name')/blob/main/CHANGELOG.md) for details."
    fi
    
    gh release create "$tag" \
        --title "$tag" \
        --notes "$release_notes" \
        dist/* || log_error "Failed to create GitHub release"
    
    log_success "GitHub release created"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            PROD_PYPI=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done


# Main execution
main() {
    log_info "Starting Enhanced SHAP publishing process..."
    
    # Pre-flight checks
    check_dependencies
    check_git_status
    
    # Extract version
    local version
    version=$(extract_version)
    local tag="v$version"
    
    log_info "Detected version: $version"
    log_info "Git tag: $tag"
    
    # Check if tag already exists
    check_git_tag "$tag"
    
    # Build and publish package
    build_package
    
    if [[ "$PROD_PYPI" == true ]]; then
        log_warning "Publishing to PRODUCTION PyPI!"
        read -p "Are you sure? This cannot be undone. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Aborted by user"
        fi
        upload_package "pypi" "PyPI"
    else
        upload_package "testpypi" "TestPyPI"
        log_info "To publish to production PyPI, use: scripts/publish.sh --prod"
    fi
    
    # Create git tag and GitHub release
    create_git_tag "$tag" "$version"
    create_github_release "$tag"
    
    # Summary
    log_success "Publishing process completed successfully!"
    
    echo
    log_info "Summary:"
    if [[ "$PROD_PYPI" == true ]]; then
        echo "  📦 Package: Published to PyPI"
    else
        echo "  📦 Package: Published to TestPyPI"
    fi
    echo "  🏷️  Git Tag: $tag created and pushed"
    echo "  🚀 GitHub Release: Created with build artifacts"
}

# Run main function
main "$@"