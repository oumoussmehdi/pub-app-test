#!/usr/bin/env bash
# Linux/macOS launcher: creates a venv on first run, syncs deps, starts the app.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
exec .venv/bin/python -m streamlit run app.py
