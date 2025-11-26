# 🎯 SIMPLE 3-STEP GUIDE TO RUN YOUR MLOPS PIPELINE

## ✅ CURRENT STATUS

**Services Running:** ✓ All services are UP
**Data Ready:** ✓ 25 samples in data.csv
**Pipeline:** 🟡 TRIGGERED - Running now!

---

## 🚀 WHAT TO DO NOW

### Option 1: Watch in Airflow UI (RECOMMENDED)

1. **Open Airflow**: http://localhost:8099
   - Login: `admin` / `admin`

2. **Find your DAG**: Click `mlops_full_pipeline`

3. **Watch the Grid View**: 
   - Tasks will change from gray → yellow → green
   - Takes ~2-3 minutes total
   - ✅ All green = SUCCESS!

4. **Click any task** to see logs and understand what it's doing

### Option 2: Watch in Terminal

```bash
# Check if running
docker exec mlops_project_clean-airflow-1 airflow dags list-runs -d mlops_full_pipeline | head -5

# See task status
docker exec mlops_project_clean-airflow-1 airflow tasks list mlops_full_pipeline -t
```

---

## 📊 VIEW RESULTS (After completion)

### Check Files Generated:

```bash
# All output files
ls -lh data/clean/
ls -lh data/features/
ls -lh data/*.csv
ls -lh data/monitoring/
ls -lh data/*.html
```

### Open Reports:

```bash
# Drift report (data quality)
open data/monitoring/drift_report.html

# Model card (documentation)
open data/model_card.html
```

### View in MLflow:

1. **Open MLflow**: http://localhost:5050
2. Click **"Experiments"** → **"demo_experiment"**
3. See your training run with:
   - Metrics (RMSE)
   - Artifacts (model files, SHAP plots)
   - Parameters

4. Click **"Models"** → **"tabular_model"**
   - See registered model versions
   - Check which is in "Production"

---

## 🎓 UNDERSTANDING THE PIPELINE

### What's Happening Right Now:

**The pipeline is processing your data through 11 steps:**

```
1. 📊 detect_data_type       → "This is tabular data"
2. 📥 ingest_data            → Load data.csv
3. ✅ validate_data          → Check schema & quality
4. 🔧 preprocess_data        → Clean + add features
                                Creates: feature_1_x_feature_2
5. 🎯 train_model            → Train Random Forest
                                Log to MLflow
6. 📈 monitor_data_drift     → Check for data shifts
7. 🚨 drift_alert_check      → Alert if drift detected
8. 🔀 check_explainability   → "Tabular? → explain it"
9. 🔍 explain_model          → Generate SHAP plots
                                Shows feature importance
10. 🏆 promote_model         → Compare RMSE scores
                                Promote if better
11. 📄 generate_model_card   → Create documentation
```

### Your Simple Dataset:

**Input:** `data/raw/data.csv`
```
feature1, feature2 → target
   1.5,     2.3    →   0
   2.1,     3.4    →   1
   ...25 total rows
```

**After Feature Engineering:**
```
feature1, feature2, feature_1_x_feature_2 → target
   1.5,     2.3,          3.45           →   0
   2.1,     3.4,          7.14           →   1
```

**After Split:**
- Training: 20 samples → train model
- Validation: 5 samples → evaluate performance

**Model:** Random Forest Classifier
- Predicts: 0 or 1
- Evaluates: Using RMSE metric
- Logs: Everything to MLflow

---

## 🎯 SUCCESS INDICATORS

### In Airflow (http://localhost:8099):

✅ **ALL TASKS GREEN** = Pipeline succeeded!

```
[🟢] detect_data_type
[🟢] ingest_data
[🟢] validate_data
[🟢] preprocess_data
[🟢] train_model
[🟢] monitor_data_drift
[🟢] drift_alert_check
[🟢] check_explainability
[🟢] explain_model
[🟢] promote_model
[🟢] generate_model_card
```

### Files Created:

```bash
✅ data/clean/data.csv                    # Cleaned data
✅ data/features/data.csv                 # With feature_1_x_feature_2
✅ data/train.csv                         # 20 samples
✅ data/validation.csv                    # 5 samples
✅ data/monitoring/drift_report.html      # Drift analysis
✅ data/model_card.html                   # Model docs
```

### In MLflow (http://localhost:5050):

```
✅ New run in "demo_experiment"
✅ Metrics logged (RMSE)
✅ Model artifacts saved
✅ SHAP plots generated
✅ Model "tabular_model" registered
✅ Version promoted to "Production"
```

---

## 📸 WHAT YOU'LL SEE

### Airflow Grid View Example:

```
Run Date: 2025-11-24 09:19:18

detect → ingest → validate → preprocess → train → monitor → alert → explain → promote → card
[🟢]     [🟢]      [🟢]        [🟢]        [🟢]     [🟢]      [🟢]     [🟢]       [🟢]       [🟢]

Status: SUCCESS
Duration: 2m 34s
```

### MLflow Experiment View Example:

```
demo_experiment

Run Name: RandomForest_v1
Start Time: 2025-11-24 09:19:45
Status: FINISHED

Metrics:
  rmse: 0.245

Parameters:
  n_estimators: 100
  max_depth: None
  random_state: 42

Artifacts:
  📁 model/
  📊 shap_summary.png
  📄 shap_values.csv
  📝 training_features.txt
```

### SHAP Explanation Plot:

Shows which features matter most:

```
Feature Importance (SHAP values)

feature_1_x_feature_2  ████████████████████  (most important)
feature2               ███████████
feature1               ████████

→ Interaction term is most predictive!
```

---

## 🔄 TRY IT YOURSELF

### Experiment 1: Add More Data

```bash
# Edit the CSV
nano data/raw/data.csv

# Add more rows, then trigger again
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline
```

### Experiment 2: Compare Model Versions

1. Trigger pipeline multiple times
2. In MLflow, select multiple runs
3. Click "Compare" button
4. See which performed best!

### Experiment 3: Modify Features

Edit: `mlops-pipeline/src/data_preprocessing/feature_engineering.py`

Add new interaction terms:
```python
# Add this in feature engineering
df['feature1_squared'] = df['feature1'] ** 2
df['feature2_squared'] = df['feature2'] ** 2
```

Then trigger pipeline again!

---

## 🎓 LEARNING POINTS

### Key Concepts You're Seeing:

1. **ML Pipeline Automation**: 
   - Data flows automatically through all stages
   - No manual intervention needed

2. **Experiment Tracking (MLflow)**:
   - Every run is logged
   - Can compare performance over time
   - Reproducibility built-in

3. **Model Registry**:
   - Models are versioned
   - Promotion logic (Staging → Production)
   - Easy rollback if needed

4. **Data Monitoring**:
   - Detect when data distribution changes
   - Alert before model degrades

5. **Explainability**:
   - Understand WHY model makes predictions
   - SHAP shows feature contributions

6. **Documentation**:
   - Model cards auto-generated
   - Includes metrics, features, intended use

---

## 💡 QUICK REFERENCE

### Important URLs:
- **Airflow**: http://localhost:8099 (admin/admin)
- **MLflow**: http://localhost:5050
- **API**: http://localhost:8000

### Key Directories:
- **Input**: `data/raw/data.csv`
- **Outputs**: `data/clean/`, `data/features/`
- **Splits**: `data/train.csv`, `data/validation.csv`
- **Reports**: `data/monitoring/`, `data/*.html`

### Useful Commands:
```bash
# Trigger pipeline
docker exec mlops_project_clean-airflow-1 airflow dags trigger mlops_full_pipeline

# Check status
docker exec mlops_project_clean-airflow-1 airflow dags list-runs -d mlops_full_pipeline

# View logs
docker-compose logs airflow | tail -50

# Restart services
docker-compose restart
```

---

## ✅ CHECKLIST

After pipeline completes:

- [ ] All Airflow tasks show green
- [ ] New MLflow run visible in demo_experiment
- [ ] Model "tabular_model" in Model Registry
- [ ] drift_report.html exists and opens
- [ ] model_card.html exists and opens
- [ ] Can see SHAP plots in MLflow artifacts

---

## 🎉 YOU'RE DONE!

**You've successfully run a complete MLOps pipeline!**

**What you accomplished:**
✅ Automated data preprocessing
✅ ML model training with tracking
✅ Data quality monitoring
✅ Model explainability
✅ Automated model promotion
✅ Documentation generation

**Next steps:**
1. Explore the Airflow and MLflow UIs
2. Read the detailed `QUICKSTART_GUIDE.md`
3. Try modifying the data or features
4. Compare multiple runs

**Need help?** Check `QUICKSTART_GUIDE.md` for detailed explanations!

---

🚀 **Happy ML Engineering!**
