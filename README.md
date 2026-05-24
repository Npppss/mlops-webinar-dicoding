# MLOps Demo Project

Complete MLOps starter project for customer churn prediction. The project demonstrates an end-to-end workflow: data ingestion, preprocessing, model training, model registry, FastAPI serving, Docker packaging, and AWS deployment guidance.

## Project Overview

Objective: predict customer churn using a reproducible ML pipeline and a deployable API.

Architecture: Data Pipeline -> Training -> Model Registry -> FastAPI -> Docker -> AWS

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python scripts/full_pipeline.py
```

This will:
- Ingest synthetic churn data into `data/raw/raw_data.csv`
- Preprocess and validate data into `data/processed/processed_data.csv`
- Train a logistic regression model with MLflow tracking
- Save `model/churn_model.pkl`
- Register a versioned model under `model_registry/churn_model/`

### 3. Start API Locally

```bash
python -m uvicorn api.app:app --reload
```

Visit: http://localhost:8000/docs

### 4. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure": 12,
    "monthly_charges": 65.5,
    "num_products": 2,
    "support_calls": 3
  }'
```

Example response:

```json
{
  "prediction": 0,
  "probability_churn": 0.23,
  "model_version": "v1.0"
}
```

## Project Structure

```text
mlops-project/
|-- data/
|   |-- raw/
|   `-- processed/
|-- data_pipeline/
|   |-- ingest.py
|   `-- preprocess.py
|-- training/
|   `-- train.py
|-- model/
|   |-- churn_model.pkl
|   `-- model_registry.py
|-- api/
|   |-- app.py
|   |-- schemas.py
|   `-- utils.py
|-- docker/
|   `-- Dockerfile
|-- config/
|   `-- config.yaml
|-- scripts/
|   |-- full_pipeline.py
|   `-- deploy_guide.sh
|-- logs/
|-- requirements.txt
`-- README.md
```

## Key MLOps Concepts

### Data Pipeline

- `data_pipeline/ingest.py`: simulates data collection.
- `data_pipeline/preprocess.py`: validates schema, cleans data, engineers features, and writes statistics.

### Model Training

- `training/train.py`: trains a scikit-learn logistic regression model.
- MLflow logs parameters, metrics, and the trained model when installed.

### Model Registry

- `model/model_registry.py`: saves versioned model artifacts and metadata to `model_registry/`.

### API Serving

- `api/app.py`: FastAPI app with validation, health checks, batch predictions, and metrics.
- `api/schemas.py`: Pydantic request/response models.
- `api/utils.py`: shared inference feature engineering.

### Containerization

```bash
docker build -t churn-api:v1 -f docker/Dockerfile .
docker run -p 8000:8000 churn-api:v1
```

## Monitoring and Logging

View logs:

```bash
tail -f logs/api.log
tail -f logs/training.log
tail -f logs/ingest.log
tail -f logs/preprocess.log
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlflow
```

Visit: http://localhost:5000

Health check:

```bash
curl http://localhost:8000/health
```

## AWS Deployment

Use the deployment guide script as a reference:

```bash
bash scripts/deploy_guide.sh
```

The script builds the Docker image, pushes it to ECR, and writes a `Dockerrun.aws.json` file for Elastic Beanstalk.

## Useful Commands

Run only ingestion:

```bash
python data_pipeline/ingest.py --rows 1000 --seed 42
```

Run only preprocessing:

```bash
python data_pipeline/preprocess.py
```

Run only training:

```bash
python training/train.py
```

Run compatibility ETL wrapper:

```bash
python data_pipeline/etl.py --rows 1000 --seed 42
```

Run compatibility retrain wrapper:

```bash
python scripts/retrain.py --rows 1400
```

## Best Practices Demonstrated

- Reproducibility through seeds and configuration.
- Automated pipeline from data to model registry.
- Input validation and structured API responses.
- Model and data quality metadata.
- Local MLflow tracking.
- Dockerized API serving.
