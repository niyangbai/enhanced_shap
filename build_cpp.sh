#!/bin/bash

set -e  # Exit immediately if a command fails
set -o pipefail

# --- Configuration ---
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
SRC_DIR="$PROJECT_ROOT/src"
BUILD_DIR="$SRC_DIR/build"

# --- Utility Functions ---
clean_build() {
  echo "Cleaning previous build..."
  rm -rf "$BUILD_DIR"
  find "$SRC_DIR" -name "*.so" -type f -delete
  echo "Cleaned build directory and old .so files."
}

configure_and_build() {
  echo "Configuring and building project..."

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active Conda environment detected."
    exit 1
  fi

  # Detect Python version
  PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

  # Set CMake path to Conda's PyTorch
  TORCH_CMAKE_PATH="$CONDA_PREFIX/lib/python${PYTHON_VERSION}/site-packages/torch/share/cmake/Torch"

  if [ ! -d "$TORCH_CMAKE_PATH" ]; then
    echo "Error: Could not find Torch CMake config at $TORCH_CMAKE_PATH"
    exit 1
  fi

  echo "Using Torch CMake config from: $TORCH_CMAKE_PATH"

  # Run CMake
  cmake ../ \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_BUILD_TYPE=Release

  # Build
  cmake --build . --config Release --parallel
  echo "Build finished."
}

show_usage() {
  echo "Usage: $0 {clean|build|rebuild}"
  echo "  clean   - Clean previous builds and .so files"
  echo "  build   - Configure and build the project"
  echo "  rebuild - Clean and build freshly"
}

# --- Main ---
case "$1" in
  clean)
    clean_build
    ;;
  build|"")
    configure_and_build
    ;;
  rebuild)
    clean_build
    configure_and_build
    ;;
  *)
    show_usage
    ;;
esac
