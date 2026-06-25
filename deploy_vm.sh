#!/bin/bash
# Run this ON THE VM to set up and start the application.
# First-time setup AND updates both use this script.
set -e

APP_DIR="$HOME/classification"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"

# ── 1. Install Docker if not present ─────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "==> Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in."
fi

# ── 2. Create app directory ───────────────────────────────────────────────────
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# ── 3. Download docker-compose.prod.yml from GitHub (or copy manually) ───────
echo "==> Downloading docker-compose.prod.yml..."
curl -fsSL "https://raw.githubusercontent.com/amirmokhtarian/classification_welodge_ollama/main/docker-compose.prod.yml" \
    -o "$COMPOSE_FILE"

# ── 4. Create .env file if it doesn't exist ───────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    echo "==> Creating .env file..."
    read -rsp "Enter API_BEARER_TOKEN: " token
    echo ""
    echo "API_BEARER_TOKEN=${token}" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo ".env created."
fi

# ── 5. Create feedback.csv if it doesn't exist (bind-mounted into container) ─
if [ ! -f "feedback.csv" ]; then
    echo "Testo,Predizione_Modello,Etichetta_Corretta" > feedback.csv
    echo "feedback.csv created."
fi

# ── 6. Pull latest images and start ──────────────────────────────────────────
echo "==> Pulling latest images..."
docker compose -f "$COMPOSE_FILE" pull

echo "==> Starting services..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "==> Status:"
docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "Done! API is running at http://$(hostname -I | awk '{print $1}'):8003"
