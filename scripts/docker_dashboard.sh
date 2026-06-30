#!/usr/bin/env bash
# Build et relance le conteneur Docker du dashboard Streamlit.
# Usage (depuis la racine du projet) :
#   bash scripts/docker_dashboard.sh           → port 8501 (défaut)
#   bash scripts/docker_dashboard.sh 8502      → port personnalisé

set -euo pipefail

PORT=${1:-8501}
IMAGE=getaround-dashboard
CONTAINER=getaround-dashboard-test

# Aller à la racine du projet (là où se trouve le Dockerfile)
cd "$(dirname "$0")/.."

echo "── Stop / remove conteneur existant ──"
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "── Build image (context = racine du projet) ──"
docker build -t "$IMAGE" -f dashboard/Dockerfile .

echo "── Lancement sur le port $PORT ──"
# --network host : le container peut atteindre l'API sur http://localhost:8000
docker run -d --name "$CONTAINER" --network host \
    -e PORT="$PORT" \
    "$IMAGE"

echo "── Logs de démarrage ──"
sleep 4
docker logs "$CONTAINER" 2>&1 | tail -6

echo ""
echo "Dashboard disponible sur http://localhost:${PORT}"
