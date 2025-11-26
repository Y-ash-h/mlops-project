# 🚀 MLOps End-to-End Production Pipeline

A complete, production-ready MLOps pipeline featuring automated data ingestion, preprocessing, model training, monitoring, and serving. Built with Apache Airflow, MLflow, FastAPI, and Docker.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a complete MLOps pipeline that demonstrates industry best practices for machine learning operations, including:

- **Automated ML Workflows**: Orchestrated with Apache Airflow
- **Experiment Tracking**: MLflow for model versioning and metrics
- **Model Serving**: FastAPI-based REST API
- **Data Quality**: Automated validation and drift detection
- **Containerization**: Full Docker-based deployment
- **Multi-Modal Support**: Tabular, text, image, and audio data

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Apache Airflow                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Ingest  │→│Preprocess│→│  Train   │→│ Promote  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│       ↓            ↓            ↓            ↓              │
└───────┼────────────┼────────────┼────────────┼─────────────┘
        ↓            ↓            ↓            ↓
┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   MinIO   │ │PostgreSQL│ │  MLflow  │ │ FastAPI  │
│  Storage  │ │    DB    │ │ Registry │ │ Serving  │
└───────────┘ └──────────┘ └──────────┘ └──────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Airflow** | 8099 | Workflow orchestration UI |
| **MLflow** | 5050 | Experiment tracking & model registry |
| **FastAPI** | 8000 | Model serving REST API |
| **MinIO** | 9000-9001 | S3-compatible object storage |
| **PostgreSQL** | 5432 | Metadata database |
| **Redis** | 6379 | Caching & message broker |

## ✨ Features

### 🔄 Automated ML Pipeline
- Data ingestion from multiple sources
- Automated data validation and quality checks
- Feature engineering and preprocessing
- Model training with hyperparameter optimization
- Model evaluation and validation
- Automated model promotion based on metrics

### 📊 Experiment Tracking
- Complete model lineage tracking with MLflow
- Metrics, parameters, and artifacts logging
- Model versioning and registry
- Parent-child run relationships

### 🎯 Data Quality & Monitoring
- Data drift detection using Evidently AI
- Automated alerts for significant drift
- Model performance monitoring
- HTML drift reports generation

### 🔌 Model Serving
- RESTful API with FastAPI
- Automatic model loading from MLflow registry
- Health check endpoints
- Interactive API documentation (Swagger UI)

### 🔔 Alerting & Notifications
- Slack notifications for pipeline events (optional)
- Validation failure alerts
- Drift detection alerts
- Model promotion notifications

## 🛠️ Prerequisites

- **Docker** (v20.10+) & **Docker Compose** (v2.0+)
- **Git**
- **8GB+ RAM** recommended
- **10GB+ free disk space**

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Y-ash-h/mlops-project.git
cd mlops-project
```

### 2. Start All Services

```bash
# Start all Docker containers
docker-compose up -d

# Wait 30 seconds for services to initialize
sleep 30

# Verify all services are running
docker-compose ps
```

Expected output - all services should show "Up":
```
NAME                                  STATUS
mlops_project_clean-airflow-1         Up
mlops_project_clean-mlflow-1          Up
mlops_project_clean-postgres-1        Up
mlops_project_clean-redis-1           Up
mlops_project_clean-minio-1           Up
mlops_project_clean-model-serving-1   Up
```

### 3. Prepare Sample Data

The project includes sample tabular data. To use it:

```bash
# Copy sample data to the pipeline input directory
cp mlops-pipeline/data/raw/sample_data.csv data/raw/data.csv

# Verify meta.json exists (specifies data type)
cat data/raw/meta.json
```

### 4. Run the Pipeline

#### Option A: Using the Automated Script (Recommended)

```bash
bash run_pipeline.sh
```

This script will:
- ✅ Check all Docker services
- ✅ Validate data files
- ✅ Trigger the ML pipeline
- ✅ Monitor execution progress
- ✅ Display results and reports

#### Option B: Manual Trigger

```bash
# Trigger the DAG
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline

# Monitor progress
docker exec mlops_project_clean-airflow-1 airflow dags list-runs -d mlops_full_pipeline
```

### 5. Access the UIs

Open your browser and visit:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8099 | admin / admin |
| **MLflow** | http://localhost:5050 | N/A |
| **API Docs** | http://localhost:8000/docs | N/A |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |

## 📁 Project Structure

```
mlops-project/
├── airflow/                    # Airflow configuration
│   ├── dags/                  # DAG definitions
│   │   ├── mlops_dag.py      # Main ML pipeline DAG
│   │   └── ml_training_dag.py
│   └── airflow_home/          # Airflow home directory
├── mlops-pipeline/            # Core ML pipeline code
│   ├── src/                   # Source code
│   │   ├── data_ingestion/   # Data loading modules
│   │   ├── preprocessing/    # Data preprocessing
│   │   ├── model_training/   # Training scripts
│   │   │   ├── train_tabular.py
│   │   │   ├── train_text.py
│   │   │   ├── train_image.py
│   │   │   └── train_audio.py
│   │   ├── model_registry/   # Model promotion logic
│   │   ├── model_monitoring/ # Drift detection
│   │   ├── explainability/   # Model interpretability
│   │   ├── alerts/           # Notification system
│   │   └── utils/            # Utility functions
│   ├── data/                  # Data storage
│   │   ├── raw/              # Raw input data
│   │   ├── clean/            # Cleaned data
│   │   ├── features/         # Engineered features
│   │   ├── train.csv         # Training split
│   │   ├── validation.csv    # Validation split
│   │   └── monitoring/       # Drift reports
│   └── config/               # Configuration files
├── model-serving/            # FastAPI serving application
│   ├── app.py               # API implementation
│   └── Dockerfile           # Serving container
├── docker-compose.yml       # Services orchestration
├── run_pipeline.sh         # Automated pipeline runner
└── README.md               # This file
```

## 🔧 Pipeline Components

### 1. Data Ingestion (`ingest_data`)

Ingests data from the raw directory and validates basic structure.

**Output**: `mlops-pipeline/data/ingested/data.csv`

### 2. Data Validation (`validate_data`)

Checks data quality, validates schema, and identifies issues.

**Output**: Validation report logs

### 3. Data Preprocessing (`preprocess_data`)

Cleans data, handles missing values, engineers features, and creates train/validation splits.

**Outputs**:
- `mlops-pipeline/data/clean/data.csv`
- `mlops-pipeline/data/features/data.csv`
- `mlops-pipeline/data/train.csv`
- `mlops-pipeline/data/validation.csv`

### 4. Data Type Detection (`detect_data_type`)

Automatically detects whether data is tabular, text, image, or audio. Uses `meta.json` if present for explicit type specification.

### 5. Data Drift Monitoring (`monitor_data_drift`)

Compares training data vs. new data to detect distribution changes using Evidently AI.

**Output**: `mlops-pipeline/data/monitoring/drift_report.html`

### 6. Model Training (`train_model`)

Routes to appropriate trainer based on data type:

#### Tabular Data Training
- Trains **LightGBM** and **XGBoost** models
- Compares performance and selects best
- Logs metrics: `accuracy`, `best_accuracy`
- Registers model in MLflow registry

**Experiment**: `tabular_model`

### 7. Model Explainability (`explain_model`)

Generates SHAP explanations for tabular models (conditional step).

### 8. Model Promotion (`promote_model`)

Compares new model vs. production model and promotes if metrics improve. This is a non-critical step - pipeline continues even if it fails.

### 9. Model Card Generation (`generate_model_card`)

Creates HTML documentation for the trained model.

**Output**: `mlops-pipeline/data/model_card.html`

## ⚙️ Configuration

### Environment Variables

Create `.env` file (optional):

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=tabular_model

# Model Serving
MODEL_NAME=tabular_model
MODEL_STAGE=Production

# Slack Notifications (optional)
SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Data Type Configuration

Create `data/raw/meta.json` to explicitly specify data type:

```json
{
  "type": "tabular",
  "description": "Sample dataset for classification"
}
```

Supported types: `tabular`, `text`, `image`, `audio`

## 📖 Usage

### Running the Complete Pipeline

```bash
# Full pipeline with monitoring
bash run_pipeline.sh

# Trigger without monitoring
bash run_pipeline.sh --no-monitor
```

### Manual DAG Management

```bash
# List all DAGs
docker exec mlops_project_clean-airflow-1 airflow dags list

# Trigger specific DAG
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline

# Check DAG runs
docker exec mlops_project_clean-airflow-1 airflow dags list-runs -d mlops_full_pipeline

# View task states for a specific run
docker exec mlops_project_clean-airflow-1 airflow tasks states-for-dag-run \
  mlops_full_pipeline <RUN_ID>
```

### Using Your Own Data

1. **Prepare your data**:
   ```bash
   # Place your CSV in the raw directory
   cp your_data.csv data/raw/data.csv
   ```

2. **Create meta.json**:
   ```bash
   echo '{"type": "tabular", "description": "My dataset"}' > data/raw/meta.json
   ```

3. **Run the pipeline**:
   ```bash
   bash run_pipeline.sh
   ```

### Expected Data Format

For tabular data, your CSV should:
- Have a header row with column names
- Include feature columns and a target column
- The last column is assumed to be the target
- Support numerical or categorical features

Example:
```csv
feature1,feature2,feature3,target
1.5,2.3,4.1,0
2.1,3.4,5.2,1
...
```

## 🔌 API Documentation

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_name": "tabular_model",
  "model_stage": "Production",
  "mlflow_uri": "http://mlflow:5050"
}
```

### Make Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"feature1": 1.5, "feature2": 2.3},
      {"feature1": 3.2, "feature2": 1.8}
    ]
  }'
```

Response:
```json
{
  "predictions": [0, 1],
  "model_name": "tabular_model",
  "model_stage": "Production"
}
```

### Interactive API Docs

Visit **http://localhost:8000/docs** for Swagger UI with interactive testing.

## 📊 Monitoring & Observability

### Airflow Monitoring

1. **DAG View**: http://localhost:8099
   - View pipeline status and history
   - Check task logs
   - Trigger manual runs
   - View task dependencies (Graph view)

2. **Task Logs**:
   ```bash
   # View logs for specific task
   docker logs mlops_project_clean-airflow-1 --tail 100
   ```

### MLflow Tracking

1. **Experiments View**: http://localhost:5050
   - Compare model runs
   - View metrics and parameters
   - Download model artifacts
   - Inspect model versions

2. **Model Registry**:
   - Track model versions
   - View production models
   - Check model lineage

### Drift Reports

Open the drift report after pipeline completion:

```bash
open mlops-pipeline/data/monitoring/drift_report.html
```

### Model Cards

View auto-generated model documentation:

```bash
open mlops-pipeline/data/model_card.html
```

## 🐛 Troubleshooting

### Services Not Starting

```bash
# Check service logs
docker-compose logs -f

# Check specific service
docker logs mlops_project_clean-airflow-1 --tail 100

# Restart all services
docker-compose down
docker-compose up -d
```

### Pipeline Failures

1. **Check Airflow UI** (http://localhost:8099)
   - Click on the failed task
   - View logs for error details

2. **Common Issues**:

   **Data not found**:
   ```bash
   # Ensure data.csv exists
   ls -la data/raw/data.csv
   
   # Copy sample data if missing
   cp mlops-pipeline/data/raw/sample_data.csv data/raw/data.csv
   ```

   **Model promotion fails**:
   - This is expected if no production model exists yet
   - Pipeline continues anyway (non-critical step)

   **Port conflicts**:
   ```bash
   # Check what's using the port
   lsof -i :8099
   
   # Change port in docker-compose.yml if needed
   ```

### Cleaning Up

```bash
# Stop all services
docker-compose down

# Remove all data (clean slate)
docker-compose down -v

# Remove generated data files
rm -rf mlops-pipeline/data/clean/*
rm -rf mlops-pipeline/data/features/*
rm -rf mlops-pipeline/data/monitoring/*
```

### Reset Everything

```bash
# Complete reset
docker-compose down -v --remove-orphans
docker system prune -a --volumes
docker-compose up -d --build
```

## 🔧 Advanced Configuration

### Custom Model Training

To add your own model:

1. Create a new trainer in `mlops-pipeline/src/model_training/`
2. Update `model_router.py` to route to your trainer
3. Rebuild containers: `docker-compose up -d --build`

### Adding Notifications

Configure Slack webhooks:

```bash
export SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK"
```

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

- **Yash Vardhanjain** - [@Y-ash-h](https://github.com/Y-ash-h)

## 🙏 Acknowledgments

- Apache Airflow community
- MLflow project
- FastAPI framework
- Evidently AI for drift detection

---

**Built with ❤️ for the MLOps community**
