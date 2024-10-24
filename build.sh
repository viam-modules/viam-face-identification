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
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
$PYTHON -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data "./src/models/checkpoints:checkpoints"  main.py
tar -czvf dist/archive.tar.gz dist/main