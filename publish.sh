#!/bin/bash

set -e

echo "Building Sphinx HTML documentation..."
make -C docs html

echo "Deploying docs to gh-pages branch with ghp-import..."
ghp-import -n -p docs/build/html

echo "Building Python package (wheel and sdist)..."
rm -rf dist/
python3 -m build

echo "Uploading to Test PyPI..."
twine upload --repository testpypi dist/*

# echo "Uploading to PyPI..."
# twine upload dist/*

echo "All done! Docs are live and package published."
