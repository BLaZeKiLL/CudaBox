#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]:-$0}")/")"
source "$SCRIPT_DIR/../env.sh"

CLEAN=0
BUILD_TYPE="RelWithDebInfo"
while [ $# -gt 0 ]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --build-type=*)
      BUILD_TYPE="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: build.sh [--clean] [--build-type <type>]"
      echo "  --clean              Delete build and dist folders before building"
      echo "  --build-type <type>  CMake build type (default: RelWithDebInfo)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
if [ "$CLEAN" -eq 1 ]; then
  bold_status "CLEANING BUILD AND DIST FOLDERS" "green"
  rm -rf "$REPO_ROOT/build" "$REPO_ROOT/dist"
fi

VENV_PATH="$HOME/uv_venv/cudabox"
if [ ! -d "$VENV_PATH" ]; then
  bold_status "CREATING UV VENV AT $VENV_PATH" "green"
  uv venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

bold_status "INSTALLING BUILD DEPENDENCIES" "green"
uv pip install "scikit-build-core>=0.11" wheel "torch>=2.7.0" triton numpy pre-commit pytest

# Install pre-commit git hooks if a config exists and hooks aren't installed yet.
# Skip in CI (GitHub Actions sets CI=true) — no git hooks needed there.
if [ -z "${CI:-}" ] && \
   [ -f "$REPO_ROOT/.pre-commit-config.yaml" ] && \
   [ ! -f "$REPO_ROOT/.git/hooks/pre-commit" ]; then
  bold_status "INSTALLING PRE-COMMIT HOOKS" "green"
  (cd "$REPO_ROOT" && pre-commit install)
fi

bold_status "BUILDING CUDABOX (build-type: $BUILD_TYPE)" "green"
uv build --wheel -Cbuild-dir=build . --verbose --color=always \
  --no-build-isolation --config-settings=cmake.build-type="$BUILD_TYPE"
bold_status "BUILD COMPLETE" "green"
ls dist

bold_status "INSTALLING CUDABOX" "green"
uv pip install ./dist/cudabox*.whl --force-reinstall
bold_status "INSTALL COMPLETE" "green"

bold_status "LISTING CUBINS IN INSTALLED .SO FILES" "green"
INSTALL_DIR="$(python -c 'import cudabox, os; print(os.path.dirname(cudabox.__file__))')"
# Pick nvcc from CUDA_HOME if set, else from PATH.
NVCC_BIN="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
CUOBJDUMP="$(dirname "$NVCC_BIN")/cuobjdump"
if [ -x "$CUOBJDUMP" ]; then
  for so in "$INSTALL_DIR"/*.so; do
    [ -e "$so" ] || continue
    echo ":::: $so ::::"
    "$CUOBJDUMP" --list-elf "$so" 2>/dev/null | sed 's/^/    /'
  done
else
  echo "cuobjdump not found at $CUOBJDUMP; skipping cubin listing." >&2
fi

bold_status "VERIFYING CUDABOX IMPORT + LISTING REGISTERED TORCH OPS" "green"
python - <<'PYEOF'
import importlib
import sys

import torch

mod = importlib.import_module("cudabox")
print(f"Imported cudabox from: {mod.__file__}")

ns = "cudabox"
all_ops = torch._C._dispatch_get_all_op_names()
ns_ops = sorted(name for name in all_ops if name.startswith(f"{ns}::"))

if not ns_ops:
    print(f"ERROR: no torch ops registered under namespace '{ns}'", file=sys.stderr)
    sys.exit(1)

print(f"Registered torch ops under torch.ops.{ns} ({len(ns_ops)}):")
for op in ns_ops:
    print(f"  - torch.ops.{op.replace('::', '.')}")
PYEOF

# If this script was sourced, the activation persists in the caller's shell.
# Otherwise, print the command to activate manually.
(return 0 2>/dev/null) || \
  bold_status "TO ACTIVATE VENV: source $VENV_PATH/bin/activate" "yellow"
