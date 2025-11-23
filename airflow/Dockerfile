FROM apache/airflow:2.9.3

# Install OS dependencies needed by ML libraries as root
USER root
RUN apt-get update && apt-get install -y libgomp1 git && rm -rf /var/lib/apt/lists/*

# Avoid pip-as-root complaints (kept for safety)
ENV PIP_ROOT_USER_ACTION=ignore

# Switch to airflow user BEFORE pip installs (required by Airflow base image)
USER airflow

# Install Python dependencies as airflow user (persistent in image)
# NOTE: if this full set causes long builds or errors, start with the minimal set below.
RUN pip install --no-cache-dir \
    "mlflow>=3.6.0" \
    "evidently==0.4.34" \
    scikit-learn \
    pandas \
    boto3 \
    xgboost \
    lightgbm \
    shap \
    torch \
    torchvision \
    pillow \
    scipy \
    soundfile \
    mlflow>=2.0 \
    numpy>=1.20 \
    matplotlib>=3.3 \
    shap>=0.41 \
    jinja2 \
    requests

# If you truly need heavy libs like torch/xgboost/lightgbm, add them later once core works.
# ENV GIT_PYTHON_REFRESH=quiet optionally, but keep GIT silence if desired
ENV GIT_PYTHON_REFRESH=quiet

# Keep running as airflow user at runtime
USER airflow
