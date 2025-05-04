#!/bin/bash

set -e
set -o pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
SRC_DIR="$PROJECT_ROOT/src"
BUILD_DIR="$SRC_DIR/build"
VENV_DIR="$PROJECT_ROOT/.venv"

# --- Clean build ---
clean_build() {
  echo "[Clean] Removing previous builds..."
  rm -rf "$BUILD_DIR"
  find "$SRC_DIR" -name "*.so" -delete
  echo "[Clean] Done."
}

# --- Configure and build ---
configure_and_build() {
  echo "[Build] Configuring and building..."

  if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Python virtual environment not found at $VENV_DIR"
    exit 1
  fi

  source "$VENV_DIR/bin/activate"

  # Get CMake config paths from installed pip packages
  TORCH_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))")
  PYBIND11_CMAKE_DIR=$(python -m pybind11 --cmakedir)

  if [ ! -d "$TORCH_DIR" ]; then
    echo "Error: Torch CMake config not found at $TORCH_DIR"
    exit 1
  fi

  if [ ! -f "$PYBIND11_CMAKE_DIR/pybind11Config.cmake" ]; then
    echo "Error: pybind11Config.cmake not found at $PYBIND11_CMAKE_DIR"
    exit 1
  fi

  export TORCH_DIR

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake ../ \
    -DCMAKE_PREFIX_PATH="$TORCH_DIR;$PYBIND11_CMAKE_DIR" \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build . --config Release --parallel
  echo "[Build] Done."
}

# --- Help ---
show_usage() {
  echo "Usage: $0 {clean|build|rebuild}"
}

# --- CLI dispatcher ---
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
