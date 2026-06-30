#!/usr/bin/env bash
# Build et relance le conteneur Docker de l'API FastAPI.
# Usage (depuis la racine du projet) :
#   bash scripts/docker_api.sh              → port 8000, MODEL_ENV=prod (via api/.env)
#   bash scripts/docker_api.sh 8001         → port personnalisé
#   bash scripts/docker_api.sh 8000 test    → mode test (pkl local, pas de MLflow)

set -euo pipefail

PORT=${1:-8000}
MODE=${2:-prod}   # prod | staging | test (pkl local)
IMAGE=getaround-api
CONTAINER=getaround-api-test

cd "$(dirname "$0")/.."

echo "── Stop / remove conteneur existant ──"
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "── Build image (context = racine du projet) ──"
docker build -t "$IMAGE" -f api/Dockerfile .

echo "── Lancement sur le port $PORT (mode=$MODE) ──"
if [ "$MODE" = "test" ]; then
    docker run -d --name "$CONTAINER" -p "${PORT}:8000" \
        -e MODEL_ENV=test \
        -e MODEL_PATH=models/getaround_pricing_model.pkl \
        "$IMAGE"
else
    docker run -d --name "$CONTAINER" -p "${PORT}:8000" \
        --env-file api/.env \
        "$IMAGE"
fi

echo "── Logs de démarrage ──"
sleep 6
docker logs "$CONTAINER" 2>&1 | tail -8

echo ""
echo "API disponible sur http://localhost:${PORT}"
echo "  GET  http://localhost:${PORT}/"
echo "  POST http://localhost:${PORT}/predict"
echo "  POST http://localhost:${PORT}/predict/batch"
echo "  Docs http://localhost:${PORT}/docs"
