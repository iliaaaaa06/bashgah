#!/usr/bin/env bash
# Starts the Streamlit UI. Backend address and port come from .env (BACKEND_URL / FRONTEND_PORT).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a
cd frontend
exec ../.venv/bin/streamlit run app.py --server.port "${FRONTEND_PORT:-8501}" --server.address "${FRONTEND_HOST:-0.0.0.0}"
