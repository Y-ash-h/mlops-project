#!/usr/bin/env bash
set -e

# Start local infra from project root
cd "$(dirname "$0")"

echo "Starting local infra..."
docker compose up -d

echo "Done. Containers:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"

echo "MinIO UI: http://localhost:9000 (minio/minio123)"
# MLflow is intentionally bound to 5001 to avoid macOS system port conflicts
echo "MLflow UI: http://localhost:5001"
