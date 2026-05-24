"""Compatibility aliases for the new API schemas.

Prefer importing from api.schemas.
"""

from api.schemas import ChurnPredictionInput, ChurnPredictionOutput, HealthCheck

PredictionRequest = ChurnPredictionInput
PredictionResponse = ChurnPredictionOutput
