# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import os
import json
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import PlainTextResponse
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Import functions from forecaster.py
from app.forecaster import (
    load_itsm_data, preprocess_data, load_model, forecast_with_model,
    run_full_training_pipeline, MODEL_DIR, detect_data_drift
)

app = FastAPI(title="ITSM Incident Forecasting API",
              description="API for forecasting ITSM incident volumes using various time series models.")

# --- Prometheus Metrics ---
# Counter: Total number of HTTP requests, labeled by method, endpoint, and status code.
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'status_code'])
# Histogram: Distribution of HTTP request durations, labeled by method and endpoint.
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency', ['method', 'endpoint'])
# Gauge: Total number of model retraining events.
MODEL_RETRAIN_COUNT = Gauge('model_retrain_total', 'Total Model Retraining Events')
# Gauge: Status of data drift detection (1 if drift detected, 0 if not).
DATA_DRIFT_DETECTED = Gauge('data_drift_detected', 'Data Drift Detection Status (1=drift, 0=no drift)')

# --- In-memory storage for processed data and model metadata ---
# In a real production application, this data would typically be loaded from a database
# or a more robust persistent storage solution (e.g., cloud storage, dedicated data store)
# to ensure consistency and availability across multiple instances.
processed_data_cache: Dict[str, pd.DataFrame] = {}
trained_models_metadata: Dict[str, Dict] = {}

# --- Pydantic Models for API Request/Response ---
class ForecastRequest(BaseModel):
    category: str
    granularity: str # 'quarterly' or 'annually'
    forecast_horizon: int # Number of periods to forecast

class ForecastResponse(BaseModel):
    category: str
    granularity: str
    forecast_periods: int
    predictions: Dict[str, float] # Date string to predicted value
    model_used: str

class RetrainResponse(BaseModel):
    message: str
    success: bool

class MetricsResponse(BaseModel):
    metrics: str

@app.on_event("startup")
async def startup_event():
    """
    Event that runs when the FastAPI application starts up.
    Loads initial data and attempts to load pre-trained models metadata.
    If no model metadata is found, it triggers an initial training process
    in the background.
    """
    print("Application startup: Loading data and models...")
    global processed_data_cache, trained_models_metadata

    # 1. Load initial raw ITSM data from CSV
    daily_incident_data = load_itsm_data(file_path="ITSM_data.csv")
    if daily_incident_data.empty:
        print("Warning: Initial data loading failed. Forecasting might not work.")
        return

    # 2. Preprocess data and cache it for quick access by API endpoints
    # This prepares the data for forecasting requests without re-loading/re-processing
    for granularity in ['quarterly', 'annually']:
        processed_data_by_category = preprocess_data(daily_incident_data, time_granularity=granularity)
        for category, df_agg in processed_data_by_category.items():
            key = f"{category}_{granularity}"
            processed_data_cache[key] = df_agg
    print("Initial data loaded and cached.")

    # 3. Attempt to load existing trained models metadata
    metadata_path = os.path.join(MODEL_DIR, "trained_models_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            trained_models_metadata = json.load(f)
        print("Loaded existing model metadata.")
    else:
        print("No existing model metadata found. Triggering initial model training in background.")
        # Trigger initial training if no models are found.
        # This is done in the background to avoid blocking startup.
        await retrain_models_background()

async def retrain_models_background():
    """
    Background task to run the full model training pipeline.
    This function is executed asynchronously, allowing the API to remain responsive.
    """
    global trained_models_metadata
    print("Background: Starting model retraining...")
    success = run_full_training_pipeline(file_path="ITSM_data.csv")
    if success:
        # Reload metadata after training is complete to reflect the new models
        metadata_path = os.path.join(MODEL_DIR, "trained_models_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                trained_models_metadata = json.load(f)
            print("Background: Model retraining completed and metadata reloaded.")
            MODEL_RETRAIN_COUNT.inc() # Increment Prometheus metric for retraining events
    else:
        print("Background: Model retraining failed.")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to record HTTP request latency and count for Prometheus.
    This runs for every incoming HTTP request.
    """
    start_time = time.time()
    response = await call_next(request) # Process the actual request
    process_time = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    status_code = response.status_code

    # Increment request counter with appropriate labels
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    # Observe request latency for the histogram
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)
    return response

@app.get("/health", summary="Health Check")
async def health_check():
    """
    Returns the health status of the API.
    A simple endpoint to check if the service is running.
    """
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse, summary="Get Incident Forecast")
async def get_forecast(request: ForecastRequest):
    """
    Provides a forecast for ITSM incident volumes for a given category and granularity.
    The API uses the best pre-trained model for the specified category and granularity.
    """
    category = request.category
    granularity = request.granularity
    forecast_horizon = request.forecast_horizon

    key = f"{category}_{granularity}"

    # Check if processed data for the requested category/granularity is available
    if key not in processed_data_cache:
        raise HTTPException(status_code=404, detail=f"Data for category '{category}' and granularity '{granularity}' not found. Please ensure data is loaded and processed.")

    df_agg = processed_data_cache[key]

    # Check if a trained model exists for the requested category/granularity
    if key not in trained_models_metadata:
        raise HTTPException(status_code=404, detail=f"Model for category '{category}' and granularity '{granularity}' not found. Please trigger retraining.")

    model_info = trained_models_metadata[key]
    model_name = model_info['model_name']

    # Load the model object from disk (Naive model does not have a saved object)
    model_object = None
    if model_name != 'Naive':
        model_object = load_model(category, granularity, model_name)
        if model_object is None:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}' for {category}_{granularity}. This might require a manual retraining.")

    # Perform forecasting using the loaded model
    predictions_series = forecast_with_model(model_object, model_name, df_agg, forecast_horizon, granularity)

    # Handle cases where forecasting might have failed or produced NaN values
    if predictions_series.isnull().all():
        raise HTTPException(status_code=500, detail=f"Forecasting failed for {category}_{granularity} with model {model_name}. Predictions are NaN.")

    # Convert predictions to a dictionary with date strings as keys
    predictions_dict = {str(k.date()): round(v, 2) for k, v in predictions_series.items()}

    return ForecastResponse(
        category=category,
        granularity=granularity,
        forecast_periods=forecast_horizon,
        predictions=predictions_dict,
        model_used=model_name
    )

@app.post("/retrain", response_model=RetrainResponse, summary="Trigger Model Retraining")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Triggers the full model retraining pipeline in the background.
    This endpoint immediately returns a response, and the retraining
    process runs asynchronously, preventing the API from blocking.
    """
    background_tasks.add_task(retrain_models_background)
    return RetrainResponse(message="Model retraining initiated in the background.", success=True)

@app.post("/detect_drift", summary="Detect Data Drift and Trigger Retrain")
async def detect_drift_and_retrain(background_tasks: BackgroundTasks, threshold_factor: float = 0.1):
    """
    Detects data drift for all categories and granularities by comparing recent data
    with historical data snapshots. If significant drift is detected for any model,
    it triggers a full model retraining in the background.
    Args:
        threshold_factor (float): The percentage change threshold for drift detection (e.g., 0.1 for 10%).
    Returns:
        dict: A message indicating whether drift was detected and if retraining was triggered.
    """
    global processed_data_cache, trained_models_metadata
    drift_detected_any = False

    print("Starting data drift detection...")
    # Reload the latest raw data to compare against the historical snapshots
    current_daily_incident_data = load_itsm_data(file_path="ITSM_data.csv")
    if current_daily_incident_data.empty:
        raise HTTPException(status_code=500, detail="Failed to load current data for drift detection. Please check ITSM_data.csv.")

    # Preprocess the current data for comparison
    current_processed_data_cache = {}
    for granularity in ['quarterly', 'annually']:
        current_processed_data_by_category = preprocess_data(current_daily_incident_data, time_granularity=granularity)
        for category, df_agg in current_processed_data_by_category.items():
            key = f"{category}_{granularity}"
            current_processed_data_cache[key] = df_agg

    # Iterate through each trained model's metadata to check for drift
    for key, model_meta in trained_models_metadata.items():
        if key in current_processed_data_cache:
            current_data_for_key = current_processed_data_cache[key]
            
            # Load historical data snapshot from metadata (stored as JSON string)
            historical_data_json = model_meta.get("data_snapshot")
            if historical_data_json:
                historical_data_df = pd.read_json(historical_data_json, orient='records', convert_dates=['ds'])
                
                # For comparison, use a recent subset of the current data
                # This is a simplified approach; more robust drift detection would compare distributions
                comparison_points = min(len(current_data_for_key), 4) # Compare last 4 periods
                if comparison_points > 0:
                    recent_current_data = current_data_for_key.tail(comparison_points)
                    
                    if detect_data_drift(recent_current_data, historical_data_df, threshold_factor):
                        print(f"Drift detected for {key}. Triggering retraining.")
                        drift_detected_any = True
                        DATA_DRIFT_DETECTED.set(1) # Set Prometheus metric to 1 (drift detected)
                        break # Trigger retraining once if drift is found in any model
                else:
                    print(f"Not enough recent data points for drift detection for {key}.")
            else:
                print(f"No historical data snapshot found for {key} in metadata. Cannot perform drift detection.")
        else:
            print(f"Current data not found for {key}. Skipping drift detection.")

    if drift_detected_any:
        background_tasks.add_task(retrain_models_background) # Trigger retraining in background
        return {"message": "Data drift detected for one or more models. Model retraining initiated in the background.", "drift_detected": True}
    else:
        DATA_DRIFT_DETECTED.set(0) # Set Prometheus metric to 0 (no drift)
        return {"message": "No significant data drift detected.", "drift_detected": False}


@app.get("/metrics", summary="Prometheus Metrics")
async def metrics():
    """
    Exposes Prometheus metrics for monitoring the application's performance and events.
    The metrics are exposed in a format that Prometheus can scrape.
    """
    return PlainTextResponse(generate_latest().decode('utf-8'), media_type="text/plain")

