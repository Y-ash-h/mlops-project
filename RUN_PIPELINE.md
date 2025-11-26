# 🚀 Run the MLOps Pipeline NOW - Quick Commands

## ✅ Everything is Ready!

Your services are running and data is prepared. Follow these simple steps:

---

## 📊 Step 1: Open the Interfaces

I've already opened these for you in VS Code's Simple Browser:

1. **Airflow UI**: http://localhost:8099
   - Username: `admin`
   - Password: `admin`

2. **MLflow UI**: http://localhost:5050

**Or open in your regular browser:**
```bash
open http://localhost:8099  # Airflow
open http://localhost:5050  # MLflow
```

---

## 🎯 Step 2: Trigger the Pipeline in Airflow

### Via Airflow UI (Recommended for learning):

1. **Login to Airflow** (http://localhost:8099)
   - Username: `admin`
   - Password: `admin`

2. **Find the DAG**:
   - Look for `mlops_full_pipeline` in the list
   - It should be at the top of the page

3. **Trigger the DAG**:
   - Click the **Play button (▶️)** on the right side
   - Select "Trigger DAG"
   - Click "Trigger" to confirm

4. **Watch it Run**:
   - The page will refresh showing the Grid view
   - Tasks will turn from gray → yellow → green
   - Total runtime: ~2-3 minutes

### Via Command Line (Alternative):

```bash
# Trigger the DAG directly
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline
```

---

## 👀 Step 3: Monitor Progress

### In Airflow:

**View Real-time Progress:**
1. Stay on the **Grid view** (default after triggering)
2. Watch task colors change:
   - ⚪ Gray = Not started
   - 🟡 Yellow = Running  
   - 🟢 Green = Success
   - 🔴 Red = Failed

**Expected Timeline:**
```
0:00 - detect_data_type      (5 seconds)
0:05 - ingest_data           (10 seconds)
0:15 - validate_data         (5 seconds)
0:20 - preprocess_data       (15 seconds)
0:35 - train_model           (45 seconds)
1:20 - monitor_data_drift    (20 seconds)
1:40 - drift_alert_check     (5 seconds)
1:45 - check_explainability  (instant)
1:45 - explain_model         (25 seconds)
2:10 - promote_model         (10 seconds)
2:20 - generate_model_card   (10 seconds)
✅ COMPLETE (2:30 total)
```

**View Task Logs:**
1. Click any task square
2. Click "Log" button
3. See real-time output

### In MLflow:

**Watch Training Runs:**
1. Go to http://localhost:5050
2. Click "Experiments" → "demo_experiment"
3. Refresh to see new run appear
4. Click run name to see details

---

## 📈 Step 4: View Results

### A) Check Airflow DAG Status

```bash
# View DAG run status
docker exec mlops_project_clean-airflow-1 airflow dags list-runs -d mlops_full_pipeline

# View task instance status
docker exec mlops_project_clean-airflow-1 airflow tasks list mlops_full_pipeline
```

### B) Check MLflow Experiment

**In MLflow UI:**
1. Click "Experiments" (left sidebar)
2. Click "demo_experiment"
3. See your training run with:
   - Metrics: RMSE, training time
   - Parameters: model hyperparameters
   - Artifacts: model files, SHAP plots

**Via Command Line:**
```bash
# List recent MLflow runs
docker exec mlops_project_clean-mlflow-1 ls -lht /mlflow-artifacts/1/ | head -5
```

### C) Check Generated Artifacts

```bash
# View all generated files
ls -lR /Users/yashvardhanjain/Downloads/mlops_project_clean/data/

# Expected output:
# data/raw/data.csv                    ✅ Input data
# data/clean/data.csv                  ✅ Cleaned data
# data/features/data.csv               ✅ Feature-engineered data
# data/train.csv                       ✅ Training split
# data/validation.csv                  ✅ Validation split
# data/monitoring/drift_report.html    ✅ Drift report
# data/model_card.html                 ✅ Model documentation
```

### D) Open Generated Reports

```bash
# Open drift report
open /Users/yashvardhanjain/Downloads/mlops_project_clean/data/monitoring/drift_report.html

# Open model card
open /Users/yashvardhanjain/Downloads/mlops_project_clean/data/model_card.html
```

Or copy to current directory:
```bash
cd /Users/yashvardhanjain/Downloads/mlops_project_clean
cp data/monitoring/drift_report.html ./drift_report.html
cp data/model_card.html ./model_card.html
open drift_report.html
open model_card.html
```

---

## 🎓 Understanding What You See

### In Airflow Grid View:

```
Tasks (left to right):
┌────────────────────────────────────────────────────────────┐
│ [🟢] [🟢] [🟢] [🟢] [🟢] [🟢] [🟢] [🟢] [🟢] [🟢] [🟢]     │
│  1    2    3    4    5    6    7    8    9   10   11      │
└────────────────────────────────────────────────────────────┘
```

1. detect_data_type - Identified data as "tabular"
2. ingest_data - Loaded data.csv
3. validate_data - Passed schema checks
4. preprocess_data - Cleaned + engineered features
5. train_model - Trained Random Forest
6. monitor_data_drift - Generated drift report
7. drift_alert_check - Checked drift thresholds
8. check_explainability - Branch decision (tabular → explain)
9. explain_model - Generated SHAP plots
10. promote_model - Compared and promoted model
11. generate_model_card - Created documentation

### In MLflow Experiments:

```
Run: RandomForest_v1
├── Metrics
│   ├── rmse: 0.245
│   └── train_time: 1.23s
├── Parameters
│   ├── n_estimators: 100
│   ├── max_depth: None
│   └── random_state: 42
└── Artifacts
    ├── model/
    ├── shap_summary.png
    ├── shap_values.csv
    └── training_features.txt
```

### In Model Registry:

```
Model: tabular_model
└── Version 1
    ├── Stage: Production
    ├── Source: run abc123...
    └── Metrics: rmse=0.245
```

---

## 🔍 Detailed Inspection Commands

### View Specific Task Logs:

```bash
# All task logs
docker exec mlops_project_clean-airflow-1 airflow tasks logs mlops_full_pipeline detect_data_type $(date +%Y-%m-%d)

docker exec mlops_project_clean-airflow-1 airflow tasks logs mlops_full_pipeline ingest_data $(date +%Y-%m-%d)

docker exec mlops_project_clean-airflow-1 airflow tasks logs mlops_full_pipeline train_model $(date +%Y-%m-%d)

docker exec mlops_project_clean-airflow-1 airflow tasks logs mlops_full_pipeline promote_model $(date +%Y-%m-%d)
```

### Check Data Pipeline Outputs:

```bash
# View cleaned data
head -10 /Users/yashvardhanjain/Downloads/mlops_project_clean/data/clean/data.csv

# View feature-engineered data (with interaction term)
head -10 /Users/yashvardhanjain/Downloads/mlops_project_clean/data/features/data.csv

# Count training vs validation samples
wc -l /Users/yashvardhanjain/Downloads/mlops_project_clean/data/train.csv
wc -l /Users/yashvardhanjain/Downloads/mlops_project_clean/data/validation.csv
```

### Check MLflow Artifacts:

```bash
# List all MLflow runs
ls -lt /Users/yashvardhanjain/Downloads/mlops_project_clean/mlflow-data/1/

# View a specific run's artifacts (replace RUN_ID)
ls -lR /Users/yashvardhanjain/Downloads/mlops_project_clean/mlflow-data/1/RUN_ID/artifacts/
```

---

## 🐛 If Something Goes Wrong

### Pipeline Failed?

**1. Check which task failed:**
- In Airflow Grid view, look for red square
- Click it → "Log" button → Read error message

**2. Common issues:**

#### Issue: "File not found"
```bash
# Ensure data exists
ls -lh /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/data.csv

# If missing, recreate:
cp /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/sample_data.csv \
   /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/data.csv
```

#### Issue: "MLflow connection error"
```bash
# Check MLflow is running
docker-compose ps mlflow

# Restart if needed
docker-compose restart mlflow

# Wait 10 seconds then retry
```

#### Issue: "Permission denied"
```bash
# Fix permissions
chmod -R 755 /Users/yashvardhanjain/Downloads/mlops_project_clean/data/
```

**3. Clear and retry:**
```bash
# Clear failed task and re-run
# In Airflow UI: Click failed task → "Clear" → "Confirm"

# Or via command line:
docker exec mlops_project_clean-airflow-1 \
  airflow tasks clear mlops_full_pipeline -y
```

### Can't Access Airflow?

```bash
# Check if Airflow is running
docker-compose ps airflow

# View Airflow logs
docker-compose logs airflow | tail -50

# Restart Airflow
docker-compose restart airflow
```

### MLflow Not Loading?

```bash
# Check MLflow status
docker-compose ps mlflow

# View logs
docker-compose logs mlflow | tail -50

# Restart
docker-compose restart mlflow
```

---

## 🔄 Run Again with Different Data

### Modify the dataset:

```bash
# Edit the CSV
nano /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/data.csv

# Or replace with new data
cp your_new_data.csv /Users/yashvardhanjain/Downloads/mlops_project_clean/data/raw/data.csv
```

### Clean previous run data:

```bash
# Remove processed data (keeps raw data)
rm -rf /Users/yashvardhanjain/Downloads/mlops_project_clean/data/clean/*
rm -rf /Users/yashvardhanjain/Downloads/mlops_project_clean/data/features/*
rm -f /Users/yashvardhanjain/Downloads/mlops_project_clean/data/train.csv
rm -f /Users/yashvardhanjain/Downloads/mlops_project_clean/data/validation.csv
```

### Trigger new run:

```bash
# Via command line
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline

# Or use Airflow UI (click Play button again)
```

---

## 📊 Sample Output Examples

### Expected Airflow Log Output:

**detect_data_type:**
```
[2025-11-24 10:30:15] INFO - Detecting data type...
[2025-11-24 10:30:15] INFO - Found CSV files in: /opt/airflow/mlops-pipeline/data/raw
[2025-11-24 10:30:15] INFO - Detected data type: tabular
```

**train_model:**
```
[2025-11-24 10:31:00] INFO - Loading training data...
[2025-11-24 10:31:00] INFO - Loaded 20 training samples
[2025-11-24 10:31:00] INFO - Features: ['feature1', 'feature2', 'feature_1_x_feature_2']
[2025-11-24 10:31:00] INFO - Training Random Forest...
[2025-11-24 10:31:02] INFO - Training complete
[2025-11-24 10:31:02] INFO - RMSE: 0.245
[2025-11-24 10:31:02] INFO - Logging to MLflow...
[2025-11-24 10:31:03] INFO - Run ID: abc123def456
```

**promote_model:**
```
[2025-11-24 10:32:00] INFO - Checking for existing production model...
[2025-11-24 10:32:00] INFO - Current production RMSE: 0.312
[2025-11-24 10:32:00] INFO - New model RMSE: 0.245
[2025-11-24 10:32:00] INFO - ✅ New model is better! Promoting...
[2025-11-24 10:32:01] INFO - Model registered as version 2
[2025-11-24 10:32:01] INFO - Transitioned to Production stage
```

---

## 🎯 Success Checklist

After pipeline completes, verify:

- [ ] All 11 tasks are **green** in Airflow Grid view
- [ ] New run appears in MLflow "demo_experiment"
- [ ] Model shows in MLflow Model Registry as "tabular_model"
- [ ] Files exist:
  - [ ] `data/clean/data.csv`
  - [ ] `data/features/data.csv`
  - [ ] `data/train.csv`
  - [ ] `data/validation.csv`
  - [ ] `data/monitoring/drift_report.html`
  - [ ] `data/model_card.html`
- [ ] Can open drift_report.html in browser
- [ ] Can open model_card.html in browser
- [ ] SHAP plots visible in MLflow artifacts

---

## 🎉 You're Done!

**Your MLOps pipeline is running successfully!**

**What you've accomplished:**
✅ Automated data ingestion and validation
✅ Feature engineering pipeline
✅ Model training with MLflow tracking
✅ Data drift monitoring with Evidently
✅ Model explainability with SHAP
✅ Automated model promotion
✅ Model documentation generation

**Next steps:**
1. Explore the QUICKSTART_GUIDE.md for detailed explanations
2. Try modifying the data and running again
3. Experiment with different model parameters
4. Add your own features to the pipeline

🚀 Happy ML Engineering!
