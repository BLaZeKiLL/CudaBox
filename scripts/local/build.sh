#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]:-$0}")/")"
source "$SCRIPT_DIR/../env.sh"

bold_status "BUILDING CUDABOX" "green"
uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation
bold_status "BUILD COMPLETE" "green"
ls dist

bold_status "INSTALLING CUDABOX" "green"
pip install ./dist/cudabox*.whl --force-reinstall
bold_status "INSTALL COMPLETE" "green"
