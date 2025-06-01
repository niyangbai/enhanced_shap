#!/bin/bash

set -e

# Extract version from pyproject.toml
VERSION=$(grep -Po '(?<=^version = ")[^"]+' pyproject.toml)
TAG="v$VERSION"

echo "Detected version: $VERSION"

# Build Sphinx docs
echo "Building Sphinx HTML documentation..."
make -C docs html

# Deploy docs to GitHub Pages
echo "Deploying docs to gh-pages branch with ghp-import..."
ghp-import -n -p docs/build/html

# Build your package
echo "Building Python package (wheel and sdist)..."
rm -rf dist/
python3 -m build

# Upload to TestPyPI
echo "Uploading to TestPyPI..."
twine upload --repository testpypi dist/*

# Upload to PyPI (Uncomment for production!)
# echo "Uploading to PyPI..."
# twine upload dist/*

# Git Tag & Push
echo "Tagging release: $TAG"
git tag $TAG
git push origin $TAG

# GitHub Release
echo "Creating GitHub release for $TAG..."
gh release create $TAG \
  --title "$TAG" \
  --notes "Release $TAG. See CHANGELOG.md for details." \
  dist/*

echo "All done! Docs live, package published, tag and GitHub release created."