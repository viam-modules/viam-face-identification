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
# The cloud builder restores a cached .venv, and `uv venv` errors out if one
# already exists, so only create it when missing. `uv pip install` still runs
# to reconcile deps (fast against the restored uv cache).
if [ ! -d .venv ]; then
    uv venv --python 3.11
fi
source .venv/bin/activate
uv pip install -r requirements.txt
$PYTHON -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data "./src/models/checkpoints:checkpoints"  main.py
tar -czvf dist/archive.tar.gz dist/main meta.json first_run.sh
