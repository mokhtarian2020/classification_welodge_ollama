#!/bin/sh
# Pull the LLM model into the shared Ollama volume if it is not already present.
# This runs once as an init container (via the ollama-init service in docker-compose).
set -e

echo "[init] Waiting for Ollama to be ready..."
until curl -sf http://ollama:11434/api/tags > /dev/null; do
    sleep 2
done

MODEL="qwen2.5:7b"
echo "[init] Checking if ${MODEL} is already pulled..."

if ollama list 2>/dev/null | grep -q "${MODEL}"; then
    echo "[init] ${MODEL} already present — skipping pull."
else
    echo "[init] Pulling ${MODEL} — this may take a few minutes on first boot..."
    ollama pull "${MODEL}"
    echo "[init] Pull complete."
fi
