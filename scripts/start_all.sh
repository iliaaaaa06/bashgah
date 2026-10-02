#!/usr/bin/env bash
# Checks the model server, then starts the backend (:8000) and the Streamlit UI (:8501).
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a

.venv/bin/python scripts/check_connection.py || { echo; echo "اتصال به سرور مدل برقرار نیست؛ ابتدا .env را اصلاح کنید."; exit 1; }

echo; echo "راه‌اندازی بک‌اند…"
.venv/bin/python -m app.main &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 60); do
  curl -s "http://127.0.0.1:${APP_PORT:-8000}/health" >/dev/null && break
  sleep 1
done

echo "رابط کاربری: http://localhost:${FRONTEND_PORT:-8501}"
./scripts/start_frontend.sh
