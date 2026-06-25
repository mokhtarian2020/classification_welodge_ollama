#!/bin/bash
# Copy monitoring config to VM and activate it.
set -euo pipefail

VM_HOST="${VM_HOST:-ceia-gesan}"
APP_DIR="${APP_DIR:-classification}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Copying files to $VM_HOST:~/classification/"
ssh -F ~/.ssh/config "$VM_HOST" "mkdir -p ~/$APP_DIR/scripts ~/$APP_DIR/logs"
scp -F ~/.ssh/config \
    "$REPO_ROOT/docker-compose.prod.yml" \
    "$VM_HOST:~/$APP_DIR/"
scp -F ~/.ssh/config \
    "$REPO_ROOT/scripts/watchdog.sh" \
    "$REPO_ROOT/scripts/install_monitoring.sh" \
    "$REPO_ROOT/scripts/classification-stack.service" \
    "$REPO_ROOT/scripts/classification-watchdog.service" \
    "$REPO_ROOT/scripts/classification-watchdog.timer" \
    "$VM_HOST:~/$APP_DIR/scripts/"

echo "==> Applying compose + monitoring on VM..."
ssh -F ~/.ssh/config "$VM_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd ~/classification
docker compose -f docker-compose.prod.yml up -d
bash scripts/install_monitoring.sh
docker compose -f docker-compose.prod.yml ps
REMOTE

echo ""
echo "==> Done. Monitor with:"
echo "  ssh $VM_HOST 'tail -f ~/classification/logs/watchdog.log'"
echo "  ssh $VM_HOST 'systemctl status classification-watchdog.timer'"
