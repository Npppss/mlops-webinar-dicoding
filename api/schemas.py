"""Pydantic schemas for churn prediction API validation."""

from pydantic import BaseModel, Field


class ChurnPredictionInput(BaseModel):
    """Input schema for churn prediction."""

    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure: int = Field(..., ge=1, le=72, description="Months with company")
    monthly_charges: float = Field(..., gt=0, le=200, description="Monthly charges")
    num_products: int = Field(..., ge=1, le=4, description="Number of products")
    support_calls: int = Field(..., ge=0, le=10, description="Support calls made")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "tenure": 12,
                "monthly_charges": 65.5,
                "num_products": 2,
                "support_calls": 3,
            }
        }
    }


class ChurnPredictionOutput(BaseModel):
    """Output schema for churn prediction."""

    prediction: int = Field(description="Prediction: 1=churn, 0=retain")
    probability_churn: float = Field(description="Probability of churn from 0 to 1")
    model_version: str = Field(description="Model version used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "probability_churn": 0.23,
                "model_version": "v1.0",
            }
        }
    }


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether a model is loaded")
    model_version: str = Field(description="Current model version")
