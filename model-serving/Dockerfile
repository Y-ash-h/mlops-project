FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    mlflow==2.10.0 \
    pandas==2.1.4 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    boto3==1.34.23

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
