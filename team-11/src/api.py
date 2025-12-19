"""FastAPI server for Fair Credit Score Prediction.

This module provides REST API endpoints for:
- Single and batch predictions
- Model explanations
- Fairness metrics
- Health checks

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    uvicorn src.api:app --reload  # Development mode
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from utils.serialization import to_python

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    from starlette.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, load_config
from src.model import CreditModel
from src.preprocessing import Preprocessor
from src.fairness import FairnessAnalyzer
from src.explainability import Explainer
from src.model_registry import ModelRegistry, ModelStatus, ModelMetadata
from src.utils import setup_logging, get_logger


# =============================================================================
# Prometheus Metrics (if available)
# =============================================================================
if PROMETHEUS_AVAILABLE:
    PREDICTION_COUNTER = Counter(
        'predictions_total',
        'Total number of predictions made',
        ['outcome', 'group']
    )
    PREDICTION_LATENCY = Histogram(
        'prediction_latency_seconds',
        'Prediction latency in seconds',
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )
    PREDICTION_ERRORS = Counter(
        'prediction_errors_total',
        'Total number of prediction errors',
        ['error_type']
    )
    FAIRNESS_DISPARATE_IMPACT = Gauge(
        'fairness_disparate_impact',
        'Current disparate impact ratio'
    )
    FAIRNESS_STATISTICAL_PARITY = Gauge(
        'fairness_statistical_parity_diff',
        'Current statistical parity difference'
    )
    FAIRNESS_EQUALIZED_ODDS = Gauge(
        'fairness_equalized_odds_diff',
        'Current equalized odds difference'
    )

    # Calibration metrics
    MODEL_CALIBRATION_BRIER_SCORE = Gauge(
        'model_calibration_brier_score',
        'Brier score for probability calibration (lower is better)'
    )
    MODEL_IS_CALIBRATED = Gauge(
        'model_is_calibrated',
        'Whether the model is calibrated (1=yes, 0=no)'
    )

    # Data quality metrics
    DATA_MISSING_VALUES = Gauge(
        'data_quality_missing_values_total',
        'Total number of missing values in recent predictions'
    )
    DATA_VALIDATION_ISSUES = Counter(
        'data_quality_validation_issues_total',
        'Total number of data validation issues',
        ['issue_type']
    )
    DATA_UNSEEN_CATEGORIES = Counter(
        'data_quality_unseen_categories_total',
        'Total number of unseen categorical values',
        ['column']
    )
    DATA_QUALITY_SCORE = Gauge(
        'data_quality_score',
        'Overall data quality score (0-100)'
    )

    # Model performance metrics
    MODEL_ACCURACY = Gauge(
        'model_accuracy',
        'Current model accuracy on recent predictions'
    )
    MODEL_FEATURE_DRIFT = Gauge(
        'model_feature_drift',
        'Feature drift score (0-1, higher = more drift)',
        ['feature']
    )
    CV_SCORE_MEAN = Gauge(
        'model_cv_score_mean',
        'Mean cross-validation score'
    )
    CV_SCORE_STD = Gauge(
        'model_cv_score_std',
        'Standard deviation of cross-validation scores'
    )

    # Audit logging metrics
    AUDIT_LOG_EVENTS = Counter(
        'audit_log_events_total',
        'Total number of audit log events',
        ['event_type', 'user']
    )
    SECURITY_EVENTS = Counter(
        'security_events_total',
        'Total number of security-related events',
        ['event_type']
    )
    API_AUTH_FAILURES = Counter(
        'api_auth_failures_total',
        'Total number of API authentication failures'
    )
    RATE_LIMIT_HITS = Counter(
        'api_rate_limit_hits_total',
        'Total number of rate limit hits',
        ['endpoint']
    )

    # A/B testing and model registry metrics
    AB_TEST_REQUESTS = Counter(
        'ab_test_requests_total',
        'Total number of A/B test requests',
        ['model_version']
    )
    MODEL_VERSION_PREDICTIONS = Counter(
        'model_version_predictions_total',
        'Total predictions by model version',
        ['version', 'outcome']
    )
    ACTIVE_MODELS = Gauge(
        'active_models_count',
        'Number of active model versions'
    )


# =============================================================================
# Audit Logging
# =============================================================================
import json
import logging
from datetime import datetime

# Create dedicated audit logger
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# Create file handler for audit logs
audit_log_path = Path("logs/audit.log")
audit_log_path.parent.mkdir(parents=True, exist_ok=True)
audit_handler = logging.FileHandler(audit_log_path)
audit_handler.setLevel(logging.INFO)

# Create formatter for audit logs
audit_formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}'
)
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)


def log_audit_event(event_type: str, details: dict, user: str = "anonymous"):
    """Log an audit event to the audit log and Prometheus.

    Args:
        event_type: Type of event (e.g., 'prediction', 'model_update', 'auth_failure')
        details: Dictionary containing event details
        user: User identifier (API key hash, IP address, or 'anonymous')
    """
    event_data = {
        "event_type": event_type,
        "user": user,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details
    }

    # Log to file
    audit_logger.info(json.dumps(event_data))

    # Update Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        AUDIT_LOG_EVENTS.labels(event_type=event_type, user=user).inc()


# =============================================================================
# Rate Limiting
# =============================================================================
if RATE_LIMITING_AVAILABLE:
    # Initialize rate limiter
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


# =============================================================================
# Request/Response Models
# =============================================================================
class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    Income: float = Field(..., ge=0, description="Annual income in dollars")
    Debt: float = Field(..., ge=0, description="Total debt in dollars")
    Loan_Amount: float = Field(..., ge=0, description="Requested loan amount")
    Loan_Term: int = Field(..., ge=1, le=360, description="Loan term in months")
    Num_Credit_Cards: int = Field(..., ge=0, le=50, description="Number of credit cards")
    Gender: str = Field(..., description="Gender (Male/Female)")
    Education: str = Field(..., description="Education level")
    Payment_History: str = Field(..., description="Payment history rating")
    Employment_Status: str = Field(..., description="Employment status")
    Residence_Type: str = Field(..., description="Type of residence")
    Marital_Status: str = Field(..., description="Marital status")

    class Config:
        json_schema_extra = {
            "example": {
                "Income": 75000,
                "Debt": 15000,
                "Loan_Amount": 25000,
                "Loan_Term": 36,
                "Num_Credit_Cards": 3,
                "Gender": "Female",
                "Education": "Bachelor's",
                "Payment_History": "Good",
                "Employment_Status": "Employed",
                "Residence_Type": "Rented",
                "Marital_Status": "Single"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    creditworthy: bool = Field(..., description="Whether applicant is creditworthy")
    probability: float = Field(..., ge=0, le=1, description="Probability of creditworthiness")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    cdi_score: int = Field(..., ge=0, le=4, description="Composite Disadvantage Index score")
    group: str = Field(..., description="Group classification (privileged/unprivileged)")
    explanation: dict[str, Any] | None = Field(None, description="Decision explanation")
    natural_language_explanation: list[str] | None = Field(None, description="Comprehensive explanation")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    records: list[PredictionRequest] = Field(..., max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    predictions: list[PredictionResponse]
    count: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class FairnessMetricsResponse(BaseModel):
    """Fairness metrics response schema."""
    thresholds: dict[str, float]
    targets: dict[str, float]
    cdi_config: dict[str, Any]


# =============================================================================
# Application State
# =============================================================================
class AppState:
    """Application state container."""
    config: Config | None = None
    model: CreditModel | None = None
    explainer: Explainer | None = None
    preprocessor: Preprocessor | None = None
    preprocessor_ready: bool = False
    logger: Any = None
    start_time: float = 0
    # Model registry for A/B testing
    registry: ModelRegistry | None = None
    ab_testing_enabled: bool = False


state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    state.start_time = time.time()
    
    config_path = os.getenv("CONFIG_PATH")
    state.config = load_config(config_path)
    state.logger = setup_logging(state.config.logging)
    
    model_path = os.getenv("MODEL_PATH", "models/credit_model.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
    
    if Path(model_path).exists():
        state.model = CreditModel(state.config)
        state.model.load(model_path)
        
        state.explainer = Explainer(state.config)
        state.explainer.setup_shap_explainer(state.model)
        
        state.logger.info(f"Model loaded from {model_path}")
    else:
        state.logger.warning(f"Model not found at {model_path}")
    

    config_path = os.getenv("CONFIG_PATH")
    state.config = load_config(config_path)
    state.logger = setup_logging(state.config.logging)

    model_path = os.getenv("MODEL_PATH", "models/credit_model.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")

    if Path(model_path).exists():
        state.model = CreditModel(state.config)
        state.model.load(model_path)

        state.explainer = Explainer(state.config)
        state.explainer.setup_shap_explainer(state.model)

        state.logger.info(f"Model loaded from {model_path}")

        # Set calibration and performance metrics
        if PROMETHEUS_AVAILABLE:
            is_calibrated = state.model._calibrated_model is not None
            MODEL_IS_CALIBRATED.set(1 if is_calibrated else 0)
            state.logger.info(f"Model calibration status: {is_calibrated}")

            # Set CV score metrics if available
            cv_scores = state.model.get_cv_scores()
            if cv_scores and isinstance(cv_scores, dict):
                # If cv_scores contains arrays of fold scores
                for metric, scores in cv_scores.items():
                    if isinstance(scores, (list, np.ndarray)) and len(scores) > 0:
                        CV_SCORE_MEAN.set(float(np.mean(scores)))
                        CV_SCORE_STD.set(float(np.std(scores)))
                        state.logger.info(f"CV {metric} - Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
                        break  # Use first metric found
    else:
        state.logger.warning(f"Model not found at {model_path}")

    state.preprocessor = Preprocessor(state.config)
    if Path(preprocessor_path).exists():
        state.preprocessor.load(preprocessor_path)
        state.preprocessor_ready = True
        state.logger.info(f"Preprocessor loaded from {preprocessor_path}")
    else:
        state.preprocessor_ready = False
        state.logger.warning(
            f"Preprocessor not found at {preprocessor_path}. "
            "Falling back to stateless request preprocessing."
        )
    
    yield
    

    # Initialize model registry for A/B testing
    state.ab_testing_enabled = os.getenv("AB_TESTING_ENABLED", "false").lower() == "true"
    if state.ab_testing_enabled:
        registry_path = Path(os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"))
        state.registry = ModelRegistry(registry_path)
        state.logger.info(f"Model registry initialized: {len(state.registry.models)} models registered")

        # Update active models count
        if PROMETHEUS_AVAILABLE:
            active_count = len([
                m for m in state.registry.models.values()
                if m.status == ModelStatus.ACTIVE
            ])
            ACTIVE_MODELS.set(active_count)
    else:
        state.logger.info("A/B testing disabled. Using single model mode.")

    yield

    
    yield
    
    # Shutdown
    state.logger.info("Shutting down API server")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Fair Credit Score Prediction API",
    description="""
    ## Equitable, Explainable Credit Scoring
    
    This API provides fair and transparent credit predictions with:
    
    - **Fairness-aware predictions** using CDI-based group thresholds
    - **Full explainability** via SHAP values
    - **Regulatory compliance** with fairness metrics
    
    ### Fairness Targets
    - Disparate Impact Ratio ≥ 0.80
    - Statistical Parity Difference < 0.05
    - Equalized Odds Difference < 0.10
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting setup
if RATE_LIMITING_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =============================================================================
# Security (Optional API Key)
# =============================================================================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)):
    """Verify API key if configured."""
    required_key = os.getenv("API_KEY")
    if required_key and api_key != required_key:
        # Log authentication failure
        log_audit_event(
            event_type="auth_failure",
            details={
                "reason": "invalid_or_missing_api_key",
                "provided_key": api_key[:8] + "..." if api_key else None
            }
        )

        if PROMETHEUS_AVAILABLE:
            API_AUTH_FAILURES.inc()
            SECURITY_EVENTS.labels(event_type="auth_failure").inc()

        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


# =============================================================================
# Helper Functions
# =============================================================================
def prepare_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run inference preprocessing using the persisted preprocessor when available."""
    if state.preprocessor and state.preprocessor_ready:
        return state.preprocessor.prepare_inference_features(df)

    # Fallback to stateless preprocessing (development mode)
    df = state.preprocessor.convert_numeric_columns(df)
    df = state.preprocessor.create_derived_features(df)
    df = state.preprocessor.calculate_cdi(df)
    df = state.preprocessor.encode_categorical(df)

    target_col = state.config.model.target_column
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    proxy_col = state.config.cdi.proxy_column
    proxy_series = df[proxy_col] if proxy_col in df.columns else None
    features = df.drop(columns=[proxy_col], errors="ignore")

    expected_features = state.model.get_feature_names() if state.model else []
    if expected_features:
        missing = [col for col in expected_features if col not in features.columns]
        for col in missing:
            features[col] = 0.0
        features = features[expected_features]

    if proxy_series is None:
        features[proxy_col] = 0.0
    else:
        features[proxy_col] = proxy_series

    return features.apply(pd.to_numeric, errors="coerce")


def preprocess_request(request: PredictionRequest) -> pd.DataFrame:
    """Convert request to preprocessed DataFrame."""
    df = pd.DataFrame([request.model_dump()])
    return prepare_features_dataframe(df)


def get_confidence_level(probability: float) -> str:
    """Determine confidence level from probability."""
    probability = float(probability)
    if probability > 0.8 or probability < 0.2:
        return "high"
    elif probability > 0.6 or probability < 0.4:
        return "medium"
    return "low"


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy" if state.model else "degraded",
        model_loaded=state.model is not None,
        version="0.1.0",
        uptime_seconds=time.time() - state.start_time,
    )

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Generate credit prediction",
)
async def predict(
    request: PredictionRequest,
    include_explanation: bool = True,
    _: str = Depends(verify_api_key),
):
    """Generate a single credit prediction with optional explanation.
    
    The prediction uses fairness-aware thresholds based on the applicant's
    CDI (Composite Disadvantage Index) score.
    """
    if state.model is None:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess
        df = preprocess_request(request)
        
        # Get CDI info
        cdi_score = int(df["CDI"].iloc[0])
        is_unprivileged = df[state.config.cdi.proxy_column].iloc[0] == 1
        
        # Get probability
        prob = float(state.model.predict_proba(df)[0])
        
        # Apply group-specific threshold
        if is_unprivileged:
            threshold = state.config.fairness.threshold_unprivileged
            group = "unprivileged"
        else:
            threshold = state.config.fairness.threshold_privileged
            group = "privileged"
        
        pred = int(prob >= threshold)
        
        # Record metrics
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNTER.labels(
                outcome="approved" if pred else "denied",
                group=group
            ).inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)
        
        # Audit log the prediction
        log_audit_event(
            event_type="prediction",
            details={
                "prediction": bool(pred),
                "probability": float(prob),
                "cdi_score": cdi_score,
                "group": group,
                "threshold_used": threshold,
                "include_explanation": include_explanation,
            }
        )
        
        response = PredictionResponse(
            creditworthy=bool(pred),
            probability=float(prob),
            confidence=get_confidence_level(prob),
            cdi_score=cdi_score,
            group=group,
        )
        
        # Add explanation if requested
        if include_explanation and state.explainer:
            explanation = state.explainer.explain_individual(df, 0, pred)
            response.explanation = {
                "decision": explanation["prediction_text"],
                "top_positive_factors": [
                    {"feature": f["feature"], "impact": float(f["strength"])}
                    for f in explanation["top_positive_factors"][:3]
                ],
                "top_negative_factors": [
                    {"feature": f["feature"], "impact": float(f["strength"])}
                    for f in explanation["top_negative_factors"][:3]
                ],
            }
            natural_language_explanation = state.explainer.generate_natural_language_explanation(explanation)
            response.natural_language_explanation = natural_language_explanation.split('\n')
        
        return to_python(response)
        
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_ERRORS.labels(error_type="processing_error").inc()
        state.logger.error(f"Prediction error: {e}")
        log_audit_event(
            event_type="prediction_error",
            details={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Generate batch predictions",
)
async def batch_predict(
    request: BatchPredictionRequest,
    _: str = Depends(verify_api_key),
):
    """Generate predictions for multiple applicants.
    
    Maximum 1000 records per request.
    
    This endpoint processes all records in a single batch for efficiency,
    rather than making individual predictions sequentially.
    """
    if state.model is None:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert all records to a single DataFrame for vectorized processing
        records_data = [record.model_dump() for record in request.records]
        df = pd.DataFrame(records_data)
        
        # Preprocess entire batch at once
        df = prepare_features_dataframe(df)
        
        # Extract CDI info for all records
        cdi_scores = df["CDI"].values.astype(int)
        proxy_col = state.config.cdi.proxy_column
        protected_values = df[proxy_col].values if proxy_col in df.columns else np.zeros(len(df))
        
        # Get probabilities for entire batch at once (vectorized!)
        probs = state.model.predict_proba(df)
        
        # Apply group-specific thresholds (vectorized!)
        threshold_priv = state.config.fairness.threshold_privileged
        threshold_unpriv = state.config.fairness.threshold_unprivileged
        
        thresholds = np.where(
            protected_values == 1,
            threshold_unpriv,
            threshold_priv
        )
        preds = (probs >= thresholds).astype(int)
        groups = np.where(protected_values == 1, "unprivileged", "privileged")
        
        # Build response objects
        predictions = []
        for i in range(len(df)):
            prob = float(probs[i])
            pred = int(preds[i])
            
            # Record metrics
            if PROMETHEUS_AVAILABLE:
                PREDICTION_COUNTER.labels(
                    outcome="approved" if pred else "denied",
                    group=groups[i]
                ).inc()
            
            predictions.append(PredictionResponse(
                creditworthy=bool(pred),
                probability=prob,
                confidence=get_confidence_level(prob),
                cdi_score=int(cdi_scores[i]),
                group=groups[i],
            ))
        
        # Record batch latency
        if PROMETHEUS_AVAILABLE:
            PREDICTION_LATENCY.observe(time.time() - start_time)
        

        # Audit log the batch prediction
        log_audit_event(
            event_type="batch_prediction",
            details={
                "count": len(predictions),
                "processing_time_seconds": time.time() - start_time,
                "num_approved": sum(1 for p in predictions if p.creditworthy),
                "num_denied": sum(1 for p in predictions if not p.creditworthy),
            }
        )

        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            processing_time_seconds=time.time() - start_time,
        )
        

        
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_ERRORS.labels(error_type="batch_processing_error").inc()
        state.logger.error(f"Batch prediction error: {e}")

        # Audit log the error
        log_audit_event(
            event_type="batch_prediction_error",
            details={"error": str(e), "num_records": len(request.records)}
        )

        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/fairness_metrics",
    response_model=FairnessMetricsResponse,
    tags=["Fairness"],
    summary="Get fairness configuration",
)
async def get_fairness_metrics():
    """Get current fairness thresholds, targets, and CDI configuration."""
    return FairnessMetricsResponse(
        thresholds={
            "privileged": state.config.fairness.threshold_privileged,
            "unprivileged": state.config.fairness.threshold_unprivileged,
        },
        targets={
            "disparate_impact_ratio": state.config.fairness.target_disparate_impact_ratio,
            "statistical_parity_diff": state.config.fairness.target_statistical_parity_diff,
            "equalized_odds_diff": state.config.fairness.target_equalized_odds_diff,
        },
        cdi_config={
            "factors": state.config.cdi.factors,
            "threshold": state.config.cdi.threshold,
        },
    )


@app.get("/model_info", tags=["Model"], summary="Get model metadata")
async def get_model_info():
    """Get information about the loaded model."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    

    
    metadata = state.model.get_training_metadata()
    return {
        "num_features": metadata.get("num_features"),
        "features": metadata.get("feature_names"),
        "used_fairness_weights": metadata.get("used_sample_weights"),
        "xgb_params": metadata.get("xgb_params"),
    }


@app.get("/model/metadata", tags=["Model"], summary="Get comprehensive model metadata")
async def get_model_metadata():
    """Get comprehensive metadata about the loaded model including version, CV scores, and training info."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    version_info = state.model.get_version()
    cv_scores = state.model.get_cv_scores()
    metadata = state.model.get_training_metadata()

    return {
        "version": version_info,
        "cv_scores": cv_scores,
        "training_metadata": metadata,
        "is_calibrated": state.model._calibrated_model is not None,
        "model_type": "XGBoost",
    }


@app.get("/model/calibration", tags=["Model"], summary="Get calibration curve data")
async def get_calibration_data():
    """Get calibration curve data for visualization.

    Returns the calibration curve data if the model is calibrated,
    otherwise returns an error.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if state.model._calibrated_model is None:
        raise HTTPException(
            status_code=400,
            detail="Model is not calibrated. Train with calibration enabled first."
        )

    # Note: This requires test data to compute the calibration curve
    # In production, you'd want to cache this or compute it during training
    return {
        "is_calibrated": True,
        "calibration_method": getattr(state.model._calibrated_model, "method", "unknown"),
        "message": "Model is calibrated. Use the calibrate CLI command to generate calibration curves."
    }


@app.get("/model/feature-importance", tags=["Model"], summary="Get feature importance")
async def get_feature_importance(importance_type: str = "gain"):
    """Get feature importance scores.

    Args:
        importance_type: Type of importance - 'weight', 'gain', or 'cover'
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if importance_type not in ["weight", "gain", "cover"]:
        raise HTTPException(
            status_code=400,
            detail="importance_type must be one of: weight, gain, cover"
        )

    try:
        all_importances = state.model.get_all_feature_importances()

        if importance_type not in all_importances:
            raise HTTPException(
                status_code=404,
                detail=f"Importance type '{importance_type}' not available"
            )

        importance_dict = all_importances[importance_type]

        # Convert to sorted list of dicts for easier consumption
        importance_list = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        return {
            "importance_type": importance_type,
            "features": importance_list,
            "total_features": len(importance_list),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fairness/visualization", tags=["Fairness"], summary="Get fairness metrics for visualization")
async def get_fairness_visualization():
    """Get fairness metrics formatted for visualization.

    Note: This endpoint returns the current fairness configuration and targets.
    To get actual fairness metrics on test data, use the visualize-fairness CLI command.
    """
    return {
        "thresholds": {
            "privileged": state.config.fairness.threshold_privileged,
            "unprivileged": state.config.fairness.threshold_unprivileged,
        },
        "targets": {
            "disparate_impact_ratio": state.config.fairness.target_disparate_impact_ratio,
            "statistical_parity_diff": state.config.fairness.target_statistical_parity_diff,
            "equalized_odds_diff": state.config.fairness.target_equalized_odds_diff,
        },
        "cdi_config": {
            "factors": state.config.cdi.factors,
            "threshold": state.config.cdi.threshold,
        },
        "message": "Use the visualize-fairness CLI command with test data to generate actual fairness plots."
    }


@app.post("/explain/waterfall", tags=["Explainability"], summary="Generate waterfall explanation")
async def generate_waterfall_explanation(
    request: PredictionRequest,
    _: str = Depends(verify_api_key),
):
    """Generate a waterfall plot explanation for a single prediction.

    Returns SHAP values and feature contributions in a format suitable for
    creating waterfall visualizations.
    """
    if state.model is None or state.explainer is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded")

    try:
        # Preprocess the request
        df = preprocess_request(request)

        # Generate prediction
        prob = float(state.model.predict_proba(df)[0])

        # Get SHAP explanation
        explanation = state.explainer.explain_individual(df, 0, int(prob >= 0.5))

        # Format for waterfall plot
        waterfall_data = {
            "prediction": {
                "probability": float(prob),
                "predicted_class": int(prob >= 0.5),
                "prediction_text": explanation.get("prediction_text", ""),
            },
            "base_value": float(explanation.get("base_value", 0)),
            "shap_values": [
                {
                    "feature": item["feature"],
                    "feature_value": item.get("value", ""),
                    "shap_value": item["shap_value"],
                    "impact": "positive" if item["shap_value"] > 0 else "negative",
                }
                for item in (explanation.get("top_positive_factors", []) +
                           explanation.get("top_negative_factors", []))
            ],
            "top_positive_factors": explanation.get("top_positive_factors", [])[:5],
            "top_negative_factors": explanation.get("top_negative_factors", [])[:5],
        }

        return waterfall_data

    except Exception as e:
        state.logger.error(f"Waterfall explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/compare", tags=["Explainability"], summary="Compare explanations across multiple records")
async def compare_predictions(
    records: list[PredictionRequest],
    _: str = Depends(verify_api_key),
):
    """Compare SHAP explanations across multiple predictions to identify common patterns.

    Maximum 10 records per request.
    """
    if state.model is None or state.explainer is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded")

    if len(records) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 records allowed for comparison"
        )

    try:
        # Process all records
        records_data = [record.model_dump() for record in records]
        df = pd.DataFrame(records_data)
        df = prepare_features_dataframe(df)

        # Generate predictions
        probs = state.model.predict_proba(df)
        preds = (probs >= 0.5).astype(int)

        # Generate explanations for all
        explanations = []
        for i in range(len(df)):
            explanation = state.explainer.explain_individual(df.iloc[i:i+1], 0, preds[i])
            explanations.append({
                "index": i,
                "probability": float(probs[i]),
                "prediction": int(preds[i]),
                "top_positive_factors": explanation.get("top_positive_factors", [])[:3],
                "top_negative_factors": explanation.get("top_negative_factors", [])[:3],
            })

        # Find common patterns
        all_positive_features = {}
        all_negative_features = {}

        for exp in explanations:
            for factor in exp["top_positive_factors"]:
                feature = factor["feature"]
                all_positive_features[feature] = all_positive_features.get(feature, 0) + 1
            for factor in exp["top_negative_factors"]:
                feature = factor["feature"]
                all_negative_features[feature] = all_negative_features.get(feature, 0) + 1

        common_positive = sorted(
            all_positive_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        common_negative = sorted(
            all_negative_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            "num_records": len(records),
            "individual_explanations": explanations,
            "common_patterns": {
                "most_common_positive_factors": [
                    {"feature": f, "count": c} for f, c in common_positive
                ],
                "most_common_negative_factors": [
                    {"feature": f, "count": c} for f, c in common_negative
                ],
            },
        }

    except Exception as e:
        state.logger.error(f"Comparison explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", tags=["Data"], summary="Validate input data quality")
async def validate_input_data(
    records: list[PredictionRequest],
    _: str = Depends(verify_api_key),
):
    """Validate data quality before making predictions.

    Checks for missing values, invalid ranges, and data quality issues.
    """
    try:
        # Convert to DataFrame
        records_data = [record.model_dump() for record in records]
        df = pd.DataFrame(records_data)

        # Run validation
        from src.preprocessing import Preprocessor
        preprocessor = Preprocessor(state.config)
        validation_report = preprocessor.validate_data(df)

        is_valid = len(validation_report.get("issues", [])) == 0

        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            missing_values = validation_report.get("missing_values", 0)
            DATA_MISSING_VALUES.set(missing_values)

            # Track validation issues
            for issue in validation_report.get("issues", []):
                issue_type = issue.split(":")[0] if ":" in issue else "unknown"
                DATA_VALIDATION_ISSUES.labels(issue_type=issue_type).inc()

            # Calculate quality score (100 - number of issues * 5)
            quality_score = max(0, 100 - len(validation_report.get("issues", [])) * 5)
            DATA_QUALITY_SCORE.set(quality_score)

        return {
            "is_valid": is_valid,
            "num_records": len(records),
            "validation_report": validation_report,
        }

    except Exception as e:
        state.logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type="text/plain")


# =============================================================================
# Model Registry & A/B Testing Endpoints
# =============================================================================
@app.get("/registry/models", tags=["Model Registry"], summary="List all registered models")
async def list_registry_models(
    status: Optional[str] = None,
    _: str = Depends(verify_api_key),
):
    """List all registered model versions.

    Args:
        status: Filter by status (active, inactive, testing, deprecated)

    Returns:
        List of registered models with metadata
    """
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(
            status_code=503,
            detail="A/B testing not enabled. Set AB_TESTING_ENABLED=true"
        )

    model_status = ModelStatus(status) if status else None
    models = state.registry.list_models(status=model_status)

    return {
        "total_models": len(models),
        "models": [m.to_dict() for m in models]
    }


@app.get("/registry/models/{version}", tags=["Model Registry"], summary="Get model metadata")
async def get_model_metadata_registry(
    version: str,
    _: str = Depends(verify_api_key),
):
    """Get metadata for a specific model version."""
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    try:
        metadata = state.registry.get_model_metadata(version)
        return metadata.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/registry/models/{version}/load", tags=["Model Registry"], summary="Load model version")
async def load_model_version(
    version: str,
    _: str = Depends(verify_api_key),
):
    """Load a model version into memory."""
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    try:
        state.registry.load_model(version, state.config)

        # Audit log
        log_audit_event(
            event_type="model_load",
            details={"version": version}
        )

        return {
            "message": f"Model version '{version}' loaded successfully",
            "version": version,
            "loaded_models": list(state.registry.loaded_models.keys())
        }
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/registry/models/{version}/unload", tags=["Model Registry"], summary="Unload model version")
async def unload_model_version(
    version: str,
    _: str = Depends(verify_api_key),
):
    """Unload a model version from memory."""
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    state.registry.unload_model(version)

    # Audit log
    log_audit_event(
        event_type="model_unload",
        details={"version": version}
    )

    return {
        "message": f"Model version '{version}' unloaded successfully",
        "version": version,
        "loaded_models": list(state.registry.loaded_models.keys())
    }


@app.get("/registry/ab-test/weights", tags=["A/B Testing"], summary="Get traffic weights")
async def get_traffic_weights(_: str = Depends(verify_api_key)):
    """Get current traffic weights for A/B testing."""
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    weights = {
        version: meta.traffic_weight
        for version, meta in state.registry.models.items()
        if meta.traffic_weight > 0
    }

    return {
        "weights": weights,
        "total_weight": sum(weights.values()),
        "loaded_models": list(state.registry.loaded_models.keys())
    }


@app.post("/registry/ab-test/weights", tags=["A/B Testing"], summary="Set traffic weights")
async def set_traffic_weights(
    weights: Dict[str, float],
    _: str = Depends(verify_api_key),
):
    """Set traffic weights for A/B testing.

    Args:
        weights: Dictionary mapping version to traffic weight (0.0 - 1.0)
                Weights must sum to 1.0

    Example:
        {"v1.0.0": 0.8, "v1.1.0": 0.2}
    """
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    try:
        state.registry.set_traffic_weights(weights)

        # Audit log
        log_audit_event(
            event_type="ab_test_weights_update",
            details={"weights": weights}
        )

        return {
            "message": "Traffic weights updated successfully",
            "weights": weights
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/registry/ab-test/compare/{version_a}/{version_b}", tags=["A/B Testing"], summary="Compare models")
async def compare_model_versions(
    version_a: str,
    version_b: str,
    _: str = Depends(verify_api_key),
):
    """Compare two model versions.

    Returns detailed comparison including metrics, status, and traffic weights.
    """
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    try:
        comparison = state.registry.compare_models(version_a, version_b)
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/registry/models", tags=["Model Registry"], summary="Register new model")
async def register_new_model(
    version: str,
    name: str,
    model_path: str,
    description: str = "",
    preprocessor_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    _: str = Depends(verify_api_key),
):
    """Register a new model version in the registry."""
    if not state.ab_testing_enabled or state.registry is None:
        raise HTTPException(status_code=503, detail="A/B testing not enabled")

    try:
        metadata = state.registry.register_model(
            version=version,
            name=name,
            model_path=model_path,
            description=description,
            preprocessor_path=preprocessor_path,
            metrics=metrics,
            tags=tags,
        )

        # Update active models count
        if PROMETHEUS_AVAILABLE:
            active_count = len([
                m for m in state.registry.models.values()
                if m.status == ModelStatus.ACTIVE
            ])
            ACTIVE_MODELS.set(active_count)

        # Audit log
        log_audit_event(
            event_type="model_register",
            details={
                "version": version,
                "name": name,
                "model_path": model_path
            }
        )

        return {
            "message": f"Model version '{version}' registered successfully",
            "metadata": metadata.to_dict()
        }
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
