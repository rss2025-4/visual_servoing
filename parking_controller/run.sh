#!/bin/bash

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

eval "$(direnv export bash)"

exec python $SCRIPT_DIR/run.py
