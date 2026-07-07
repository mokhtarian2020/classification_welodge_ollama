#!/bin/bash
# Monitors Ollama + classification API and restarts the stack if unhealthy.
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/classification}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/watchdog.log"
MAX_LOG_BYTES=5242880  # 5 MB

mkdir -p "$LOG_DIR"

rotate_log() {
    if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE")" -ge "$MAX_LOG_BYTES" ]; then
        mv "$LOG_FILE" "$LOG_FILE.1"
    fi
}

log() {
    rotate_log
    echo "$(date -Is) $*" | tee -a "$LOG_FILE"
}

container_health() {
    local name="$1"
    docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$name" 2>/dev/null || echo "missing"
}

container_running() {
    docker inspect --format='{{.State.Running}}' "$1" 2>/dev/null || echo "false"
}

check_ollama_model() {
    docker exec ollama ollama show qwen2.5:3b >/dev/null 2>&1
}

check_api_health() {
    curl -sf --connect-timeout 5 --max-time 10 "http://127.0.0.1/health" >/dev/null 2>&1
}

check_predict_smoke() {
    local token
    token=$(grep -E '^API_BEARER_TOKEN=' "$APP_DIR/.env" 2>/dev/null | cut -d= -f2-)
    if [ -z "$token" ]; then
        log "WARN: API_BEARER_TOKEN not found in .env, skipping predict smoke test"
        return 0
    fi

    local response http_code
    response=$(curl -s --connect-timeout 5 --max-time 30 \
        -w "\nHTTP_CODE:%{http_code}" \
        -X POST "http://127.0.0.1/predict" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${token}" \
        -d '{"input":"test watchdog"}')

    http_code=$(echo "$response" | tail -1 | cut -d: -f2)
    if [ "$http_code" = "200" ]; then
        return 0
    fi

    log "ISSUE: predict smoke test returned HTTP $http_code"
    return 1
}

restart_stack() {
    log "ACTION: restarting full stack (docker compose up -d)"
    cd "$APP_DIR"
    docker compose -f "$COMPOSE_FILE" up -d
}

restart_service() {
    local service="$1"
    log "ACTION: restarting service '$service'"
    cd "$APP_DIR"
    docker compose -f "$COMPOSE_FILE" restart "$service"
}

main() {
    local issues=0

    if ! docker info >/dev/null 2>&1; then
        log "ERROR: Docker daemon not reachable"
        exit 1
    fi

    local ollama_running api_running ollama_health api_health
    ollama_running=$(container_running ollama)
    api_running=$(container_running classification-api)
    ollama_health=$(container_health ollama)
    api_health=$(container_health classification-api)

    if [ "$ollama_running" != "true" ]; then
        log "ISSUE: ollama container not running"
        issues=$((issues + 1))
    elif [ "$ollama_health" != "healthy" ]; then
        log "ISSUE: ollama health=$ollama_health"
        issues=$((issues + 1))
    elif ! check_ollama_model; then
        log "ISSUE: qwen2.5:3b model not available in ollama"
        issues=$((issues + 1))
    fi

    if [ "$api_running" != "true" ]; then
        log "ISSUE: classification-api container not running"
        issues=$((issues + 1))
    elif [ "$api_health" != "healthy" ]; then
        log "ISSUE: classification-api health=$api_health"
        issues=$((issues + 1))
    elif ! check_api_health; then
        log "ISSUE: HTTP health check failed on http://127.0.0.1/health"
        issues=$((issues + 1))
    elif ! check_predict_smoke; then
        log "ISSUE: predict smoke test failed"
        issues=$((issues + 1))
    fi

    if [ "$issues" -eq 0 ]; then
        log "OK: ollama=healthy api=healthy model=qwen2.5:3b http=ok predict=ok"
        exit 0
    fi

    log "WARN: detected $issues issue(s), attempting recovery"

    if [ "$ollama_running" != "true" ] || [ "$ollama_health" != "healthy" ] || ! check_ollama_model; then
        restart_stack
        sleep 30
    elif [ "$api_running" != "true" ] || [ "$api_health" != "healthy" ] || ! check_api_health || ! check_predict_smoke; then
        restart_service app
        sleep 15
    fi

    if check_ollama_model && check_api_health && check_predict_smoke; then
        log "RECOVERY: successful"
        exit 0
    fi

    log "RECOVERY: failed, forcing full stack restart"
    restart_stack
    sleep 45

    if check_ollama_model && check_api_health && check_predict_smoke; then
        log "RECOVERY: successful after full restart"
        exit 0
    fi

    log "RECOVERY: failed — manual intervention required"
    exit 1
}

main "$@"
