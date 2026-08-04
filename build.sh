#!/bin/bash

set -e

SUDO=sudo
if ! command -v $SUDO; then
	echo no sudo on this system, proceeding as current user
	SUDO=""
fi

if ! command -v uv 2>&1 >/dev/null; then
    pip install uv
fi

source .env
# the version of pytorch we are using requires 3.11 or lower.
# The cloud builder restores a cached .venv but not the uv-managed interpreter it
# was created from, and its fallback to a partial cache key can hand back a .venv
# built for other requirements, so the restored environment is only reusable if
# its interpreter still resolves and is still 3.11. `uv venv` refuses to overwrite
# an existing environment without --clear. `uv pip install` then reconciles deps
# (fast against the restored uv cache).
if ! $PYTHON -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))" >/dev/null 2>&1; then
    uv venv --clear --python 3.11
fi
source .venv/bin/activate
uv pip install -r requirements.txt
# The linux/arm64 torch wheel bundles LLVM's libomp (the x86_64 one bundles
# libgomp, which is unaffected). Inside the cgroupv2 containers the cloud builder
# uses, libomp's CPU topology discovery aborts with "OMP: Error #13: Assertion
# failure at kmp_affinity.cpp" (llvm/llvm-project#137136). PyInstaller imports
# torch in a subprocess to enumerate its submodules, so that abort kills the
# build. Disabling affinity makes libomp skip topology discovery altogether.
KMP_AFFINITY=disabled $PYTHON -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data "./src/models/checkpoints:checkpoints"  main.py
tar -czvf dist/archive.tar.gz dist/main meta.json
