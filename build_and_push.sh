#!/bin/bash
# Build the app image and push it to Docker Hub.
# Run this from your local machine whenever you want to deploy a new version.
set -e

IMAGE="amirmokhtarian/classification-welodge"
TAG="${1:-latest}"

echo "==> Building image: ${IMAGE}:${TAG}"
docker build -t "${IMAGE}:${TAG}" -f dockerfile .

# Also tag as latest if a specific version was given
if [ "${TAG}" != "latest" ]; then
    docker tag "${IMAGE}:${TAG}" "${IMAGE}:latest"
fi

echo "==> Logging in to Docker Hub..."
docker login

echo "==> Pushing ${IMAGE}:${TAG}"
docker push "${IMAGE}:${TAG}"

if [ "${TAG}" != "latest" ]; then
    echo "==> Pushing ${IMAGE}:latest"
    docker push "${IMAGE}:latest"
fi

echo ""
echo "Done! Deploy on the VM with:"
echo "  ssh ceia-gesan 'cd ~/classification && docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d'"
