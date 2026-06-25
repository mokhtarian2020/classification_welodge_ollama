#!/bin/bash
# Install systemd auto-start + watchdog timer on the VM.
# Run on the VM: bash scripts/install_monitoring.sh
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/classification}"
SCRIPT_DIR="$APP_DIR/scripts"

chmod +x "$SCRIPT_DIR/watchdog.sh"

echo "==> Installing systemd units..."
sudo cp "$SCRIPT_DIR/classification-stack.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/classification-watchdog.service" /etc/systemd/system/
sudo cp "$SCRIPT_DIR/classification-watchdog.timer" /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable classification-stack.service
sudo systemctl enable classification-watchdog.timer
sudo systemctl start classification-watchdog.timer

echo "==> Watchdog timer status:"
systemctl list-timers classification-watchdog.timer --no-pager

echo ""
echo "==> Running initial watchdog check..."
"$SCRIPT_DIR/watchdog.sh" || true

echo ""
echo "Done. Stack auto-starts on boot; watchdog runs every 5 minutes."
echo "Logs: $APP_DIR/logs/watchdog.log"
