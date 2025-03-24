#!/bin/bash

# Get the path to the "src" directory (adjust if needed)
SRC_PATH="$(pwd)/src"

# Get the site-packages directory of the currently activated environment
SITE_PACKAGES_PATH=$(python -c "import site; print(site.getsitepackages()[0])")

# Create a .pth file pointing to your src directory
echo "$SRC_PATH" > "$SITE_PACKAGES_PATH/eshap.pth"

echo "Added '$SRC_PATH' to PYTHONPATH via '$SITE_PACKAGES_PATH/eshap.pth'"
