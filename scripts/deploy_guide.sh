#!/bin/bash

# AWS deployment guide for the churn prediction API.
# Prerequisites: AWS CLI, Docker, EB CLI, and configured AWS credentials.

set -e

echo "======================================"
echo "MLOps Project AWS Deployment Guide"
echo "======================================"

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REPO_NAME="${ECR_REPO_NAME:-churn-prediction-api}"
IMAGE_TAG="${IMAGE_TAG:-v1.0}"
BEANSTALK_APP="${BEANSTALK_APP:-churn-api}"
BEANSTALK_ENV="${BEANSTALK_ENV:-churn-api-prod}"

echo "AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "Region: ${AWS_REGION}"
echo "ECR Repo: ${ECR_REPO_NAME}"

echo
echo "[1/5] Building Docker image"
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" -f docker/Dockerfile .

echo
echo "[2/5] Creating ECR repository if needed"
aws ecr create-repository \
  --repository-name "${ECR_REPO_NAME}" \
  --region "${AWS_REGION}" \
  || echo "Repository already exists"

echo
echo "[3/5] Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS \
  --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo
echo "[4/5] Pushing image to ECR"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

echo
echo "[5/5] Preparing Elastic Beanstalk Dockerrun.aws.json"
cat > Dockerrun.aws.json << EOF
{
  "AWSEBDockerrunVersion": "1",
  "Image": {
    "Name": "${ECR_URI}",
    "Update": "true"
  },
  "Ports": [
    {
      "ContainerPort": 8000,
      "HostPort": 8000
    }
  ],
  "Logging": "/var/log/containers"
}
EOF

echo "Created Dockerrun.aws.json"
echo
echo "Manual deploy option:"
echo "1. Open AWS Elastic Beanstalk Console."
echo "2. Select application: ${BEANSTALK_APP}"
echo "3. Upload Dockerrun.aws.json as a new version."
echo "4. Deploy to environment: ${BEANSTALK_ENV}"
echo
echo "EB CLI option after initialization:"
echo "eb deploy ${BEANSTALK_ENV}"
echo
echo "Deployment guide complete."
