#!/usr/bin/env bash
# Starts the Streamlit UI. Address, port and BACKEND_URL come from .env (read by frontend_app.py).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python frontend_app.py
