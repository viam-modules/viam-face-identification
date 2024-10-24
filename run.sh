#!/usr/bin/env bash
cd `dirname $0`

if [ -f .installed ]
  then
    uv venv --python 3.10
    source .venv/bin/activate
  else
    apt-get install python3-pip
    if ! command -v uv 2>&1 >/dev/null; then
      pip install uv
    fi
    
    # the version of pytorch we are using requires 3.10 or lower
    uv venv --python 3.10
    source .venv/bin/activate
    uv pip install -r requirements.txt
    if [ $? -eq 0 ]
      then
        touch .installed
    fi
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec uv run python -m main $@

