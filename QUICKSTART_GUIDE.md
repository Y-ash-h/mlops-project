# 🚀 MLOps Pipeline Quick Start Guide

This guide will walk you through running the complete MLOps pipeline step-by-step using Airflow and MLflow UIs.

## 📋 Table of Contents
1. [Pre-requisites](#pre-requisites)
2. [Understanding the Pipeline](#understanding-the-pipeline)
3. [Step-by-Step Execution](#step-by-step-execution)
4. [Using Airflow UI](#using-airflow-ui)
5. [Using MLflow UI](#using-mlflow-ui)
6. [Understanding the Results](#understanding-the-results)

---

## ✅ Pre-requisites

**All services are already running!** You should have:
- ✅ Docker & Docker Compose installed
- ✅ Services running (Airflow, MLflow, MinIO, PostgreSQL, Redis)
- ✅ Sample dataset available at `data/raw/data.csv`

**Service URLs:**
- **Airflow UI**: http://localhost:8099
- **MLflow UI**: http://localhost:5050
- **MinIO Console**: http://localhost:9001
- **Model Serving API**: http://localhost:8000

**Default Credentials:**
- **Airflow**: username: `admin`, password: `admin`
- **MinIO**: username: `minio`, password: `minio123`

---

## 🔄 Understanding the Pipeline

Our MLOps pipeline has **10 stages**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Flow                       │
└─────────────────────────────────────────────────────────────────┘

1. 📊 Detect Data Type      → Identifies if data is tabular/image/text
2. 📥 Ingest Data           → Loads raw data into pipeline
3. ✅ Validate Data         → Checks data quality & schema
4. 🔧 Preprocess Data       → Cleans, engineers features, splits data
5. 🎯 Train Model           → Trains ML model (Random Forest)
6. 📈 Monitor Data Drift    → Detects distribution shifts
7. 🚨 Drift Alert           → Sends alerts if drift detected
8. 🔍 Explain Model         → Generates SHAP explanations (tabular only)
9. 🏆 Promote Model         → Compares & promotes best model
10. 📄 Generate Model Card  → Creates documentation
```

**Dataset Overview:**
- **Features**: `feature1`, `feature2` (numeric)
- **Target**: `target` (binary: 0 or 1)
- **Size**: 25 samples (20 for training, 5 for validation)
- **Task**: Binary classification using Random Forest

---

## 🎯 Step-by-Step Execution

### Step 1: Access Airflow UI

1. Open your browser and navigate to: **http://localhost:8099**
2. Login with:
   - **Username**: `admin`
   - **Password**: `admin`

3. You'll see the Airflow home page with a list of DAGs (Directed Acyclic Graphs)

![Airflow Login](https://via.placeholder.com/800x400?text=Airflow+Dashboard)

---

### Step 2: Locate the MLOps DAG

1. In the DAG list, find **`mlops_full_pipeline`**
2. You'll see:
   - **Toggle switch** (left side) - enables/disables the DAG
   - **DAG name** - click to view details
   - **Last run status** - shows success/failure
   - **Actions** - play button to trigger runs

**DAG Status Indicators:**
- 🟢 Green = Success
- 🔴 Red = Failed
- 🟡 Yellow = Running
- ⚪ Gray = Not started

---

### Step 3: Understand the DAG Structure

1. Click on the **`mlops_full_pipeline`** DAG name
2. You'll see the **Graph View** showing all tasks and dependencies

**Graph View shows:**
```
detect_data_type → ingest_data → validate_data → preprocess_data → train_model
                                                                          ↓
                                                                  monitor_data_drift
                                                                          ↓
                                                                   drift_alert_check
                                                                          ↓
                                                                  check_explainability
                                                                    /              \
                                                            explain_model      skip_explain
                                                                    \              /
                                                                      promote_model
                                                                           ↓
                                                                  generate_model_card
```

**Switch between views:**
- **Graph** - Visual flow diagram
- **Grid** - Timeline of runs
- **Calendar** - Run history by date
- **Task Duration** - Performance metrics
- **Code** - View the DAG source code

---

### Step 4: Trigger the Pipeline

1. Click the **Play button (▶️)** on the top right
2. Select **"Trigger DAG"**
3. Click **"Trigger"** in the confirmation dialog

**What happens next:**
- Airflow creates a new DAG run
- Tasks execute in sequence based on dependencies
- You'll be redirected to the Grid view

---

### Step 5: Monitor Pipeline Execution

**Watch the Grid View:**
1. Each row = one DAG run
2. Each column = one task
3. Colors indicate status:
   - ⚪ Queued
   - 🟡 Running
   - 🟢 Success
   - 🔴 Failed
   - 🟠 Upstream failed

**Click on any task square to:**
- View logs
- See task details
- Mark success/failed manually
- Clear task instance

---

### Step 6: View Task Logs

**To see what's happening inside each task:**

1. Click on any **task square** in the Grid view
2. Click **"Log"** button in the popup
3. You'll see real-time execution logs

**Important logs to check:**

**a) detect_data_type:**
```log
Detected data type: tabular
```

**b) ingest_data:**
```log
Ingested files: ['data.csv']
```

**c) validate_data:**
```log
✅ Data validation passed
```

**d) preprocess_data:**
```log
Preprocessing result: ok
- Cleaned data saved to: data/clean/data.csv
- Engineered features: feature1, feature2, feature_1_x_feature_2
- Saved to: data/features/data.csv
- Train/validation split completed
```

**e) train_model:**
```log
Training result: success
Model trained with RMSE: 0.245
Features used: ['feature1', 'feature2', 'feature_1_x_feature_2']
MLflow run ID: abc123...
```

**f) monitor_data_drift:**
```log
Data drift report generated: data/monitoring/drift_report.html
```

**g) explain_model:**
```log
SHAP explanation generated
Artifacts: shap_summary.png, shap_values.csv, explanation.html
```

**h) promote_model:**
```log
Comparing models...
New model RMSE: 0.245 < Production RMSE: 0.312
✅ Model promoted to Production stage!
```

---

### Step 7: Access MLflow UI

While the pipeline is running or after completion:

1. Open new tab: **http://localhost:5050**
2. You'll see the MLflow landing page

**MLflow has two main sections:**
- **Experiments** - Track training runs
- **Models** - Manage registered models

---

### Step 8: Explore MLflow Experiments

1. Click **"Experiments"** in left sidebar
2. Find **"demo_experiment"** (or click "All Experiments")
3. You'll see a table of all training runs

**Run Table Columns:**
- **Start Time** - When training began
- **Run Name** - e.g., "RandomForest_v1"
- **Status** - FINISHED, RUNNING, FAILED
- **Metrics** - RMSE, accuracy, etc.
- **Parameters** - Hyperparameters used
- **Version** - MLflow version

**Click on a run to see:**
- Full parameters list
- All metrics logged
- Artifacts (model files, plots, etc.)
- Tags and metadata

---

### Step 9: Compare Multiple Runs

**If you trigger the DAG multiple times:**

1. In Experiments view, select **multiple runs** (checkboxes)
2. Click **"Compare"** button
3. View side-by-side comparison:
   - **Parallel Coordinates Plot** - Visual parameter/metric comparison
   - **Scatter Plot** - Correlations
   - **Contour Plot** - Parameter spaces
   - **Box Plot** - Metric distributions

**Filter and sort:**
- Click column headers to sort
- Use search bar to filter by metrics
- Add/remove columns as needed

---

### Step 10: View Model Artifacts

1. Click on any **run name** to open run details
2. Scroll to **"Artifacts"** section
3. You'll see:

```
artifacts/
├── model/                          # Trained model files
│   ├── MLmodel
│   ├── conda.yaml
│   ├── model.pkl
│   ├── python_env.yaml
│   └── requirements.txt
├── shap_summary.png                # SHAP feature importance plot
├── shap_values.csv                 # SHAP values data
├── explanation.html                # Interactive SHAP dashboard
└── training_features.txt           # List of features used
```

**Download artifacts:**
- Click on any file to preview
- Click download icon to save locally

---

### Step 11: Explore Model Registry

1. Click **"Models"** in MLflow left sidebar
2. Find **"tabular_model"** (our registered model name)
3. Click on it to see:

**Model Versions:**
```
Version | Stage      | Created          | Run ID
--------|------------|------------------|--------
1       | Production | 2025-11-24 10:30 | abc123
2       | Staging    | 2025-11-24 11:45 | def456
3       | None       | 2025-11-24 12:15 | ghi789
```

**Stages:**
- **Production** - Currently serving in production
- **Staging** - Testing before production
- **Archived** - Retired versions
- **None** - Not yet promoted

**Click on a version to:**
- View source run details
- Change stage (promote/demote)
- Add description
- Add tags

---

### Step 12: View Data Drift Report

1. After pipeline completes, check the drift report:
   ```bash
   open drift_report.html
   ```
   (or navigate to `data/monitoring/drift_report.html`)

2. The **Evidently Report** shows:
   - **Dataset Drift** - Overall distribution changes
   - **Feature Drift** - Per-feature drift scores
   - **Target Drift** - Label distribution changes
   - **Drift Score** - Quantitative metric (0-1)

**Interpreting drift:**
- 🟢 **No drift** (score < 0.1) - Model safe to use
- 🟡 **Mild drift** (0.1-0.3) - Monitor closely
- 🔴 **Significant drift** (> 0.3) - Retrain recommended

---

### Step 13: View Model Explanation

The SHAP summary plot shows feature importance:

**From Airflow:**
1. Go to task **`explain_model`** logs
2. Note the artifact path
3. Access via MLflow (Step 10)

**From MLflow:**
1. Open latest run → Artifacts → `shap_summary.png`

**SHAP Plot Interpretation:**
- **Y-axis**: Features ranked by importance
- **X-axis**: SHAP value (impact on prediction)
- **Colors**: 
  - 🔴 Red = High feature value
  - 🔵 Blue = Low feature value
- **Width**: Concentration of impacts

**Example:**
```
feature_1_x_feature_2  |||||||||||||||||||||| (most important)
feature2               ||||||||||||
feature1               ||||||
```

---

### Step 14: View Model Card

After pipeline completion:

1. Check `data/model_card.html`
   ```bash
   open /Users/yashvardhanjain/Downloads/mlops_project_clean/data/model_card.html
   ```

2. The Model Card contains:
   - **Model Overview** - Type, algorithm, purpose
   - **Performance Metrics** - RMSE, accuracy, etc.
   - **Training Details** - Dataset size, features, hyperparameters
   - **Model Explanation** - SHAP plot embedded
   - **Deployment Info** - Version, stage, timestamp
   - **Intended Use** - Recommended applications
   - **Limitations** - Known issues and constraints

---

## 🎓 Understanding the Results

### What Just Happened?

1. **Data was processed:**
   - Raw CSV → Cleaned data → Feature engineering → Train/validation split

2. **Model was trained:**
   - Random Forest with default parameters
   - Logged to MLflow with metrics and artifacts

3. **Quality checks performed:**
   - Data validation passed
   - Drift monitoring completed
   - Model explanation generated

4. **Model was promoted:**
   - Compared against existing production model
   - Promoted if performance improved

5. **Documentation created:**
   - Model card generated
   - Artifacts saved for reproducibility

---

### Key Metrics to Monitor

| Metric | Location | Good Value | Action Required |
|--------|----------|------------|-----------------|
| **RMSE** | MLflow Runs | < 0.3 | Tune hyperparameters if higher |
| **Drift Score** | drift_report.html | < 0.1 | Retrain if > 0.3 |
| **Feature Importance** | SHAP plots | Balanced | Investigate if one feature dominates |
| **Task Duration** | Airflow Graph | < 5 min total | Optimize slow tasks |

---

## 🔄 Running the Pipeline Again

**To run with new data:**

1. Replace `data/raw/data.csv` with new dataset
   ```bash
   cp your_new_data.csv /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/data.csv
   ```

2. In Airflow UI, click **"Trigger DAG"** again

3. New run will:
   - Process new data
   - Train new model version
   - Compare against previous best
   - Promote if better

**To modify pipeline behavior:**

Edit environment variables in `docker-compose.yml`:
```yaml
environment:
  - MLFLOW_EXPERIMENT_NAME=demo_experiment  # Change experiment
  - LABEL_COLUMN=target                      # Change target column
  - MODEL_REGISTRY_NAME=tabular_model        # Change model name
  - PROMOTE_ON_IMPROVEMENT=true              # Auto-promote if better
```

Then restart services:
```bash
docker-compose restart airflow
```

---

## 🐛 Troubleshooting

### Pipeline Failed?

1. **Check task logs:**
   - Airflow → Task → Log button
   - Look for error messages in red

2. **Common issues:**

   **a) File not found:**
   ```
   FileNotFoundError: data/raw/data.csv
   ```
   **Fix**: Ensure data.csv exists in data/raw/

   **b) Column missing:**
   ```
   ValueError: Label column 'target' not found
   ```
   **Fix**: Verify CSV has 'target' column

   **c) MLflow connection error:**
   ```
   ConnectionError: http://mlflow:5000
   ```
   **Fix**: Check MLflow service is running:
   ```bash
   docker-compose ps mlflow
   ```

3. **Clear failed task:**
   - Click failed task → "Clear" → "Confirm"
   - Re-run from that point

### MLflow Not Loading?

1. Check service status:
   ```bash
   docker-compose ps mlflow
   ```

2. View MLflow logs:
   ```bash
   docker-compose logs mlflow
   ```

3. Restart MLflow:
   ```bash
   docker-compose restart mlflow
   ```

### No Model Promoted?

**Reasons:**
- New model didn't improve metrics
- Validation set too small
- Promotion disabled in config

**Check promotion logs:**
- Airflow → promote_model task → Logs
- Look for comparison metrics

---

## 🎯 Next Steps

### Enhance the Pipeline:

1. **Add more features:**
   - Edit `src/data_preprocessing/feature_engineering.py`
   - Add polynomial features, interactions, etc.

2. **Try different models:**
   - Edit `training/train.py`
   - Replace RandomForest with XGBoost, LightGBM, etc.

3. **Tune hyperparameters:**
   - Add hyperparameter search in training
   - Log different configurations to MLflow

4. **Add more monitoring:**
   - Model performance over time
   - Feature quality checks
   - Prediction distribution monitoring

5. **Deploy the model:**
   - Use the model-serving service at port 8000
   - Send prediction requests via API

### Test the Model Serving API:

```bash
# Health check
curl http://localhost:8000/health

# Make prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 2.5, "feature2": 3.1}'
```

---

## 📚 Additional Resources

**Airflow Documentation:**
- Concepts: https://airflow.apache.org/docs/apache-airflow/stable/concepts/
- DAGs: https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html

**MLflow Documentation:**
- Tracking: https://mlflow.org/docs/latest/tracking.html
- Models: https://mlflow.org/docs/latest/models.html

**Project Files:**
- `README.md` - Project overview and checkpoints
- `SETUP_REPO.md` - Initial setup instructions
- `PRODUCTION_DEPLOYMENT.md` - Production deployment guide
- `SYSTEM_VERIFICATION.md` - System verification steps

---

## 🎉 Success Criteria

You've successfully run the pipeline when:

✅ All 10 tasks show **green** in Airflow Grid view
✅ MLflow shows new run with metrics logged
✅ Model registered in MLflow Model Registry
✅ `drift_report.html` generated and accessible
✅ `model_card.html` created with model info
✅ SHAP plots visible in MLflow artifacts

**Congratulations!** You've completed a full MLOps pipeline run! 🚀

---

## 💡 Tips for Learning

1. **Experiment freely**: Try triggering the DAG multiple times with different data
2. **Read the logs**: Understanding logs helps debug issues faster
3. **Compare runs**: Use MLflow's comparison feature to see what works best
4. **Modify gradually**: Change one thing at a time and observe the impact
5. **Document findings**: Keep notes on what configurations work best

Happy MLOps! 🎈
