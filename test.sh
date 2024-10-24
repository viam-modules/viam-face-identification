#!/usr/bin/env bash

if ! command -v uv 2>&1 >/dev/null; then
		pip install uv
	fi
	uv venv --python 3.11
    source .venv/bin/activate
    uv pip install -r requirements.txt

	python3 -m pytest tests/*