#!/bin/bash
set -euo pipefail

echo "Starting Text Morph Application..."

# Configuration (override via env vars)
: "${BACKEND_HOST:=0.0.0.0}"
: "${BACKEND_PORT:=5000}"
: "${PORT:=8080}"                # Streamlit port
: "${UVICORN_RELOAD:=false}"     # Set true only for local dev, not production

UVICORN_MODULE="Backend.main:app"  # FastAPI app lives in Backend/main.py

if [ "${UVICORN_RELOAD}" = "true" ]; then
    RELOAD_FLAG="--reload"
else
    RELOAD_FLAG=""
fi

echo "Starting FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT} (reload=${UVICORN_RELOAD})"
uvicorn "${UVICORN_MODULE}" --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" ${RELOAD_FLAG} &
BACKEND_PID=$!

# Simple wait loop for health (max ~30s)
ATTEMPTS=0
until curl -sf "http://localhost:${BACKEND_PORT}/ping" >/dev/null 2>&1 || [ $ATTEMPTS -ge 30 ]; do
    ATTEMPTS=$((ATTEMPTS+1))
    sleep 1
done || true

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend process exited unexpectedly" >&2
    exit 1
fi

export BACKEND_URL="${BACKEND_URL:-http://localhost:${BACKEND_PORT}}"
echo "Starting Streamlit frontend on port ${PORT} (BACKEND_URL=${BACKEND_URL})"
exec streamlit run Frontend/app.py \
    --server.port "${PORT}" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
