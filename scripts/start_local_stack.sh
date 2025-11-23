#!/usr/bin/env bash
set -e
echo "Starting local infra..."
docker compose up -d
echo "Done. Containers:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
echo "MinIO UI: http://localhost:9000 (minio/minio123)"
echo "MLflow UI: http://localhost:5000"
