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

# Start backend and capture PID
uvicorn "${UVICORN_MODULE}" \
    --host "${BACKEND_HOST}" \
    --port "${BACKEND_PORT}" \
    ${RELOAD_FLAG} \
    > >(while read line; do echo "[FASTAPI] $line"; done) \
    2> >(while read line; do echo "[FASTAPI][ERR] $line" >&2; done) &
BACKEND_PID=$!   # <-- store backend PID here

# Wait for health check
ATTEMPTS=0
until curl -sf "http://localhost:${BACKEND_PORT}/ping" >/dev/null 2>&1 || [ $ATTEMPTS -ge 30 ]; do
    ATTEMPTS=$((ATTEMPTS+1))
    sleep 1
done || true

# Verify backend is still alive
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend process exited unexpectedly" >&2
    exit 1
fi

# Set backend URL for Streamlit
export BACKEND_URL="${BACKEND_URL:-http://localhost:${BACKEND_PORT}}"

echo "Starting Streamlit frontend on port ${PORT} (BACKEND_URL=${BACKEND_URL})"

exec streamlit run Frontend/app.py \
    --server.port "${PORT}" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
