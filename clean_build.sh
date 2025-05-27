#!/bin/bash

echo "Cleaning shared objects, build artifacts, and Python cache files in src/..."

# Remove all .so files recursively in src/
find src -type f -name "*.so" -exec rm -v {} \;

# Remove all .pyc files recursively in src/
find src -type f -name "*.pyc" -exec rm -v {} \;

# Remove build-related directories and __pycache__ only in src/
for dir in build dist __pycache__ *.egg-info; do
    find src -type d -name "$dir" -exec rm -rv {} +
done
done

echo "Done."