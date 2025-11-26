#!/bin/bash

# Simple DAG Status Checker
# Run this to see current pipeline status

AIRFLOW_CONTAINER="mlops_project_clean-airflow-1"
DAG_NAME="mlops_full_pipeline"

echo "🔍 Checking MLOps Pipeline Status..."
echo "=================================="
echo ""

# Get latest run
echo "📊 Latest DAG Runs:"
docker exec "$AIRFLOW_CONTAINER" airflow dags list-runs -d "$DAG_NAME" | head -6

echo ""
echo "=================================="
echo ""

# Get task states for latest run
echo "📋 Task Status (latest run):"
docker exec "$AIRFLOW_CONTAINER" bash -c "
  airflow tasks states-for-dag-run $DAG_NAME \$(airflow dags list-runs -d $DAG_NAME --output=json 2>/dev/null | head -1 | grep -o '\"run_id\":\"[^\"]*' | cut -d'\"' -f4) 2>/dev/null || echo 'No runs found'
"

echo ""
echo "=================================="
echo ""
echo "💡 Quick Actions:"
echo "  • View in Airflow UI: http://localhost:8099"
echo "  • View in MLflow UI: http://localhost:5050"
echo "  • Trigger new run: docker exec $AIRFLOW_CONTAINER airflow dags trigger $DAG_NAME"
