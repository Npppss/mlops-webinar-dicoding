# Introduction to MLOps: From Data Pipeline to Model Deployment

Complete webinar demo project that covers:
- ETL pipeline
- Model training with versioning
- FastAPI inference API
- Docker containerization
- AWS ECR + Elastic Beanstalk deployment workflow

## Project Structure

```text
mlops-project/
│
├── data_pipeline/
│   └── etl.py
├── training/
│   └── train.py
├── model/
│   ├── model_v1.pkl
│   └── metadata.json
├── api/
│   ├── main.py
│   └── schema.py
├── scripts/
│   └── retrain.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## 1) Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

## 2) Run ETL + Training (Create First Model)

```bash
python data_pipeline/etl.py --rows 1200 --seed 42
python training/train.py
```

This generates:
- `processed_data.csv`
- `model/model_v1.pkl`
- `model/metadata.json`

## 3) Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 62000,
    "tenure": 4,
    "num_products": 3,
    "last_month_spend": 1300
  }'
```

## 4) Retrain (Auto-Increment Model Version)

```bash
python scripts/retrain.py --rows 1500
```

If `model_v1.pkl` exists, retraining creates `model_v2.pkl`, updates `metadata.json`, and API starts using the latest model.

## 5) Docker

Build image:

```bash
docker build -t mlops-webinar-demo:latest .
```

Run container:

```bash
docker run -p 8000:8000 mlops-webinar-demo:latest
```

## 6) AWS Deployment (ECR + Elastic Beanstalk Docker Platform)

### Prerequisites
- AWS CLI configured (`aws configure`)
- Elastic Beanstalk CLI installed (`pip install awsebcli`)
- Docker installed and running

### Variables

```bash
AWS_REGION=ap-southeast-1
AWS_ACCOUNT_ID=<your-account-id>
ECR_REPO=mlops-webinar-demo
IMAGE_TAG=v1
```

### Create ECR repository (one-time)

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
```

### Authenticate Docker to ECR

```bash
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

### Build, tag, push image

```bash
docker build -t $ECR_REPO:$IMAGE_TAG .
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
```

### Prepare Elastic Beanstalk app (Docker platform)

```bash
eb init -p docker mlops-webinar-app --region $AWS_REGION
eb create mlops-webinar-env
```

### Deploy updates

```bash
eb deploy
```

### Optional: open in browser

```bash
eb open
```

## Notes for Webinar Demo
- Start with ETL and show generated `processed_data.csv`.
- Train and show `model/metadata.json` for versioning.
- Call `/predict` and show logs for incoming request + prediction.
- Run retrain script and show new model version appears.
- Containerize and present AWS push/deploy commands.
