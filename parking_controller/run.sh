#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(cd "$SCRIPT_DIR" && direnv export bash)"

exec python $SCRIPT_DIR/run.py
