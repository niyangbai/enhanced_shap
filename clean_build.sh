#!/bin/bash

echo "Cleaning shared objects and build artifacts..."

# Remove all .so files recursively
find . -type f -name "*.so" -exec rm -v {} \;

# Remove build-related directories
for dir in build dist __pycache__ *.egg-info; do
    find . -type d -name "$dir" -exec rm -rv {} +
done

echo "Done."