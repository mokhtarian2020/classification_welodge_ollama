#!/bin/sh
# Wait for Ollama to be reachable before starting the API.
# Prevents "Name or service not known" errors during stack restarts.
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MAX_WAIT=120
elapsed=0

echo "[startup] Waiting for Ollama at ${OLLAMA_HOST}..."
while ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "[startup] ERROR: Ollama not reachable after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "[startup] Ollama ready (${elapsed}s), starting API..."
exec uvicorn app:app --host 0.0.0.0 --port 8003
