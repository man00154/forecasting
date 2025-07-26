import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import joblib
import os
from prophet import Prophet
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Directory to save and load models. This directory will be created if it doesn't exist.
MODEL_DIR = "app/models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_itsm_data(file_path="ITSM_data.csv"):
    """
    Loads and initially processes the ITSM incident data from a CSV file.
    Args:
        file_path (str): Path to the ITSM CSV file.
    Returns:
        pd.DataFrame: A DataFrame with 'Date', 'Category', and 'Ticket_Count'
                      aggregated daily for each category.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())

        # Convert 'Close_Time' to datetime, coercing errors to NaT (Not a Time)
        df['Close_Time'] = pd.to_datetime(df['Close_Time'], errors='coerce', dayfirst=True)

        # Drop rows where 'Close_Time' is invalid (NaT)
        df.dropna(subset=['Close_Time'], inplace=True)
        print(f"After dropping rows with invalid 'Close_Time', shape: {df.shape}")

        # Extract date part from 'Close_Time'
        df['Date'] = df['Close_Time'].dt.date

        # Group by Date and Category to get daily incident counts
        daily_incident_counts = df.groupby(['Date', 'Category']).size().reset_index(name='Ticket_Count')
        return daily_incident_counts

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during data loading or initial processing: {e}")
        return pd.DataFrame()

def preprocess_data(df, time_granularity='quarterly'):
    """
    Preprocesses the data by aggregating ticket counts based on time granularity.
    Args:
        df (pd.DataFrame): Input DataFrame with 'Date', 'Category', 'Ticket_Count' (daily aggregated).
        time_granularity (str): 'quarterly' or 'annually'.
    Returns:
        dict: A dictionary where keys are categories and values are DataFrames
              aggregated by the specified time granularity.
    """
    processed_data = {}
    unique_categories = df['Category'].unique()

    for category in unique_categories:
        category_df = df[df['Category'] == category].copy()

        # Convert 'Date' to datetime and set as index for resampling
        category_df['Date'] = pd.to_datetime(category_df['Date'])
        category_df.set_index('Date', inplace=True)

        if time_granularity == 'quarterly':
            # Resample to quarterly start ('QS') and sum ticket counts
            agg_df = category_df['Ticket_Count'].resample('QS').sum().reset_index()
            agg_df.rename(columns={'Date': 'ds', 'Ticket_Count': 'y'}, inplace=True)
        elif time_granularity == 'annually':
            # Resample to annual start ('AS') and sum ticket counts
            agg_df = category_df['Ticket_Count'].resample('AS').sum().reset_index()
            agg_df.rename(columns={'Date': 'ds', 'Ticket_Count': 'y'}, inplace=True)
        else:
            raise ValueError("time_granularity must be 'quarterly' or 'annually'.")

        # Normalize 'ds' column to remove time component, keeping only date
        agg_df['ds'] = pd.to_datetime(agg_df['ds']).dt.normalize()
        processed_data[category] = agg_df
    return processed_data

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a time series model using MAE, RMSE, and MAPE.
    Args:
        y_true (pd.Series or np.array): Actual values.
        y_pred (pd.Series or np.array): Predicted values.
        model_name (str): Name of the model for printing.
    Returns:
        dict: A dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE calculation by replacing zeros with NaN
    y_true_no_zero = np.where(y_true != 0, y_true, np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_no_zero)) * 100

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def train_and_forecast_arima(train_df, test_df, order=(1,1,1), seasonal_order=(1,1,0,4)):
    """
    Trains and forecasts using SARIMA model.
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        order (tuple): (p,d,q) order of the ARIMA model.
        seasonal_order (tuple): (P,D,Q,s) seasonal order of the SARIMA model.
    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        train_series = train_df.set_index('ds')['y']

        # Infer frequency for the time series index
        try:
            freq = pd.infer_freq(train_series.index)
        except ValueError:
            freq = None

        # If frequency cannot be inferred, attempt to determine it based on time differences
        if freq is None and len(train_series) > 1:
            time_diff = train_series.index.to_series().diff().dropna()
            if (time_diff.dt.days == 91).all(): # Approximate for quarterly
                freq = 'QS'
            elif (time_diff.dt.days == 365).all(): # Approximate for annually
                freq = 'AS'

        # Set frequency if determined, otherwise skip model
        if freq:
            train_series.index.freq = freq
        else:
            print("SARIMA: Could not infer frequency. Skipping model.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        # Initialize and fit the ARIMA model
        model = ARIMA(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        forecast_steps = len(test_df)
        # Predict values for the test set
        predictions = model_fit.predict(start=len(train_df), end=len(train_df) + forecast_steps - 1)
        metrics = evaluate_model(test_df['y'], predictions, "SARIMA")
        return predictions, metrics
    except Exception as e:
        print(f"SARIMA training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def train_and_forecast_prophet(train_df, test_df, time_granularity):
    """
    Trains and forecasts using Facebook Prophet.
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        time_granularity (str): 'quarterly' or 'annually'.
    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
        )
        # Add quarterly seasonality if applicable
        if time_granularity == 'quarterly':
            model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)

        model.fit(train_df)
        # Create future dataframe for predictions
        future = model.make_future_dataframe(periods=len(test_df), freq='QS' if time_granularity == 'quarterly' else 'AS')
        forecast = model.predict(future)
        # Extract predictions for the test set
        predictions = forecast['yhat'].tail(len(test_df)).values
        metrics = evaluate_model(test_df['y'], predictions, "Prophet")
        return pd.Series(predictions, index=test_df['ds']), metrics
    except Exception as e:
        print(f"Prophet training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def train_and_forecast_ets(train_df, test_df, time_granularity):
    """
    Trains and forecasts using Exponential Smoothing (Holt-Winters).
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        time_granularity (str): 'quarterly' or 'annually'.
    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        train_series = train_df.set_index('ds')['y']

        # Infer frequency for the time series index
        try:
            freq = pd.infer_freq(train_series.index)
        except ValueError:
            freq = None

        # If frequency cannot be inferred, attempt to determine it based on time differences
        if freq is None and len(train_series) > 1:
            time_diff = train_series.index.to_series().diff().dropna()
            if (time_diff.dt.days == 91).all(): # Approximate for quarterly
                freq = 'QS'
            elif (time_diff.dt.days == 365).all(): # Approximate for annually
                freq = 'AS'

        # Set frequency if determined, otherwise skip model
        if freq:
            train_series.index.freq = freq
        else:
            print("ETS: Could not infer frequency. Skipping model.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        seasonal_periods = 4 if time_granularity == 'quarterly' else 1

        # Initialize and fit the Exponential Smoothing model
        model = ExponentialSmoothing(
            train_series,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated"
        )
        model_fit = model.fit()
        # Forecast values for the test set
        predictions = model_fit.forecast(len(test_df))
        metrics = evaluate_model(test_df['y'], predictions, "ETS")
        return predictions, metrics
    except Exception as e:
        print(f"ETS training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def create_time_features(df):
    """
    Creates time-based features for ML models.
    Args:
        df (pd.DataFrame): DataFrame with a 'ds' (datetime) column.
    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    df['year'] = df['ds'].dt.year
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    return df

def train_and_forecast_ml_model(train_df, test_df, model_type='RandomForest'):
    """
    Trains and forecasts using a machine learning regressor (Random Forest or XGBoost).
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
        model_type (str): 'RandomForest' or 'XGBoost'.
    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    try:
        # Create time-based features for training and test sets
        train_features = create_time_features(train_df.copy())
        test_features = create_time_features(test_df.copy())

        features = ['year', 'quarter', 'month', 'dayofyear', 'weekofyear']
        # Filter for features that actually exist in both dataframes
        existing_features = [f for f in features if f in train_features.columns and f in test_features.columns]
        if not existing_features:
            print(f"ML Model: No relevant time features found for training. Skipping {model_type}.")
            return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

        X_train = train_features[existing_features]
        y_train = train_features['y']
        X_test = test_features[existing_features]

        # Initialize the selected ML model
        if model_type == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model_name = "RandomForest"
        elif model_type == 'XGBoost':
            model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
            model_name = "XGBoost"
        else:
            raise ValueError("model_type must be 'RandomForest' or 'XGBoost'.")

        # Fit the model and make predictions
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = pd.Series(predictions, index=test_df['ds'])
        metrics = evaluate_model(test_df['y'], predictions, model_name)
        return predictions, metrics
    except Exception as e:
        print(f"{model_type} training failed: {e}")
        return pd.Series([np.nan] * len(test_df), index=test_df['ds']), {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}

def train_and_forecast_naive(train_df, test_df):
    """
    Trains and forecasts using a Naive (last value) model.
    Args:
        train_df (pd.DataFrame): Training data with 'ds' and 'y'.
        test_df (pd.DataFrame): Test data with 'ds' and 'y'.
    Returns:
        tuple: (pd.Series, dict) - Predicted values and evaluation metrics.
    """
    if not train_df.empty:
        # Naive forecast uses the last known value from the training data
        last_value = train_df['y'].iloc[-1]
        predictions = pd.Series([last_value] * len(test_df), index=test_df['ds'])
    else:
        predictions = pd.Series([np.nan] * len(test_df), index=test_df['ds'])
    metrics = evaluate_model(test_df['y'], predictions, "Naive")
    return predictions, metrics

def save_model(model, category, granularity, model_name):
    """
    Saves a trained model to disk using joblib.
    Args:
        model: The trained model object.
        category (str): The incident category.
        granularity (str): The time granularity ('quarterly' or 'annually').
        model_name (str): The name of the model (e.g., 'Prophet', 'SARIMA').
    """
    filename = os.path.join(MODEL_DIR, f"{category}_{granularity}_{model_name}.joblib")
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")

def load_model(category, granularity, model_name):
    """
    Loads a trained model from disk using joblib.
    Args:
        category (str): The incident category.
        granularity (str): The time granularity ('quarterly' or 'annually').
        model_name (str): The name of the model.
    Returns:
        The loaded model object, or None if not found.
    """
    filename = os.path.join(MODEL_DIR, f"{category}_{granularity}_{model_name}.joblib")
    if os.path.exists(filename):
        print(f"Loading model: {filename}")
        return joblib.load(filename)
    print(f"Model not found: {filename}")
    return None

def train_and_select_best_model(df_agg, category, granularity, test_size):
    """
    Trains multiple time series models, evaluates their performance, and selects
    the best performing one based on RMSE. The best model is then re-trained
    on the full dataset.
    Args:
        df_agg (pd.DataFrame): Aggregated data for a specific category and granularity.
        category (str): The incident category.
        granularity (str): The time granularity ('quarterly' or 'annually').
        test_size (int): Number of periods to use for the test set.
    Returns:
        tuple: (str, object) - The name of the best model and its trained object,
                               or (None, None) if no suitable model is found.
    """
    train_df = df_agg.iloc[:-test_size].copy()
    test_df = df_agg.iloc[-test_size:].copy()

    all_model_results = {}
    all_model_predictions = {}
    trained_models_for_selection = {} # Stores models fitted on train_df for evaluation

    sarima_order = (1,1,1)
    sarima_seasonal_order = (1,1,0,4)

    # --- Naive Model ---
    naive_predictions, naive_metrics = train_and_forecast_naive(train_df, test_df)
    all_model_results['Naive'] = naive_metrics
    all_model_predictions['Naive'] = naive_predictions
    # No actual model object for Naive, it's just a logic

    # --- SARIMA Model ---
    sarima_predictions, sarima_metrics = train_and_forecast_arima(train_df, test_df, order=sarima_order, seasonal_order=sarima_seasonal_order)
    all_model_results['SARIMA'] = sarima_metrics
    all_model_predictions['SARIMA'] = sarima_predictions
    # For SARIMA, we need to re-fit on full data for final forecast, so we save its parameters or the fitted model if possible
    try:
        train_series = train_df.set_index('ds')['y']
        freq = pd.infer_freq(train_series.index)
        if freq:
            train_series.index.freq = freq
            model = ARIMA(train_series, order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            trained_models_for_selection['SARIMA'] = model.fit() # Save fitted model for potential re-training on full data
    except Exception as e:
        print(f"Could not fit SARIMA model for selection: {e}")


    # --- Prophet Model ---
    prophet_model = Prophet(yearly_seasonality=True)
    if granularity == 'quarterly':
        prophet_model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
    prophet_model.fit(train_df)
    prophet_predictions, prophet_metrics = train_and_forecast_prophet(train_df, test_df, granularity)
    all_model_results['Prophet'] = prophet_metrics
    all_model_predictions['Prophet'] = prophet_predictions
    trained_models_for_selection['Prophet'] = prophet_model # Save Prophet model

    # --- ETS Model ---
    try:
        train_series = train_df.set_index('ds')['y']
        freq = pd.infer_freq(train_series.index)
        if freq:
            train_series.index.freq = freq
            seasonal_periods = 4 if granularity == 'quarterly' else 1
            ets_model = ExponentialSmoothing(train_series, seasonal_periods=seasonal_periods, initialization_method="estimated")
            ets_model_fit = ets_model.fit()
            ets_predictions = ets_model_fit.forecast(len(test_df))
            ets_metrics = evaluate_model(test_df['y'], ets_predictions, "ETS")
            all_model_results['ETS'] = ets_metrics
            all_model_predictions['ETS'] = ets_predictions
            trained_models_for_selection['ETS'] = ets_model_fit # Save fitted ETS model
        else:
            print("ETS: Could not infer frequency for training. Skipping model.")
            all_model_results['ETS'] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
            all_model_predictions['ETS'] = pd.Series([np.nan] * len(test_df), index=test_df['ds'])
    except Exception as e:
        print(f"ETS training failed during selection: {e}")
        all_model_results['ETS'] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
        all_model_predictions['ETS'] = pd.Series([np.nan] * len(test_df), index=test_df['ds'])


    # --- RandomForest Model ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_predictions, rf_metrics = train_and_forecast_ml_model(train_df, test_df, 'RandomForest')
    all_model_results['RandomForest'] = rf_metrics
    all_model_predictions['RandomForest'] = rf_predictions
    trained_models_for_selection['RandomForest'] = rf_model # Save RF model (will be fitted later on full data)

    # --- XGBoost Model ---
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    xgb_predictions, xgb_metrics = train_and_forecast_ml_model(train_df, test_df, 'XGBoost')
    all_model_results['XGBoost'] = xgb_metrics
    all_model_predictions['XGBoost'] = xgb_predictions
    trained_models_for_selection['XGBoost'] = xgb_model # Save XGB model (will be fitted later on full data)

    best_model_name = None
    min_rmse = float('inf')

    # Select the best model based on RMSE
    for model_name, metrics in all_model_results.items():
        if not np.isnan(metrics['RMSE']) and metrics['RMSE'] < min_rmse:
            min_rmse = metrics['RMSE']
            best_model_name = model_name

    if best_model_name:
        print(f"Best model for {category} ({granularity}): {best_model_name} (RMSE: {min_rmse:.2f})")
        # Re-train the best model on the full dataset before saving for production use
        full_df = df_agg.copy()
        final_trained_model = None

        if best_model_name == 'Naive':
            # No actual model object for Naive, just need the last value logic
            final_trained_model = None # Or a placeholder indicating Naive
        elif best_model_name == 'SARIMA':
            full_series = full_df.set_index('ds')['y']
            try:
                freq = pd.infer_freq(full_series.index)
                if freq:
                    full_series.index.freq = freq
                    model = ARIMA(full_series, order=sarima_order, seasonal_order=sarima_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                    final_trained_model = model.fit()
            except Exception as e:
                print(f"SARIMA re-training on full data failed: {e}")
                final_trained_model = None
        elif best_model_name == 'Prophet':
            model = Prophet(yearly_seasonality=True)
            if granularity == 'quarterly':
                model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
            model.fit(full_df)
            final_trained_model = model
        elif best_model_name == 'ETS':
            full_series = full_df.set_index('ds')['y']
            try:
                freq = pd.infer_freq(full_series.index)
                if freq:
                    full_series.index.freq = freq
                    seasonal_periods = 4 if granularity == 'quarterly' else 1
                    model = ExponentialSmoothing(full_series, seasonal_periods=seasonal_periods, initialization_method="estimated")
                    final_trained_model = model.fit()
            except Exception as e:
                print(f"ETS re-training on full data failed: {e}")
                final_trained_model = None
        elif best_model_name in ['RandomForest', 'XGBoost']:
            full_features = create_time_features(full_df.copy())
            features_to_use = ['year', 'quarter', 'month', 'dayofyear', 'weekofyear']
            existing_features = [f for f in features_to_use if f in full_features.columns]
            if existing_features:
                X_full = full_features[existing_features]
                y_full = full_features['y']
                if best_model_name == 'RandomForest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
                model.fit(X_full, y_full)
                final_trained_model = model
            else:
                print(f"ML Model re-training: Missing features for {best_model_name}. Skipping.")
                final_trained_model = None
        else:
            final_trained_model = None # Fallback for unknown best_model_name

        return best_model_name, final_trained_model
    else:
        print(f"Could not determine best model for {category} ({granularity}). All models failed or produced NaN RMSE.")
        return None, None

def forecast_with_model(model, model_name, full_df, future_periods, granularity):
    """
    Generates future forecasts using the selected and trained model.
    Args:
        model: The trained model object.
        model_name (str): The name of the model.
        full_df (pd.DataFrame): The full aggregated dataset used for training.
        future_periods (int): Number of future periods to forecast.
        granularity (str): The time granularity ('quarterly' or 'annually').
    Returns:
        pd.Series: Forecasted values with future dates as index.
    """
    last_date = full_df['ds'].max()
    if granularity == 'quarterly':
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=future_periods, freq='QS')
    else:
        future_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), periods=future_periods, freq='AS')

    future_df = pd.DataFrame({'ds': future_dates})

    final_forecast_predictions = pd.Series([np.nan] * future_periods, index=future_df['ds'])

    try:
        if model_name == 'Naive':
            if not full_df.empty:
                final_forecast_predictions = pd.Series([full_df['y'].iloc[-1]] * future_periods, index=future_df['ds'])
        elif model_name == 'SARIMA':
            # model here is the fitted ARIMA model
            if model:
                final_forecast_predictions = model.predict(start=len(full_df), end=len(full_df) + future_periods - 1)
                final_forecast_predictions.index = future_df['ds']
        elif model_name == 'Prophet':
            # model here is the fitted Prophet model
            if model:
                future = model.make_future_dataframe(periods=future_periods, freq='QS' if granularity == 'quarterly' else 'AS')
                forecast = model.predict(future)
                final_forecast_predictions = forecast['yhat'].tail(future_periods)
                final_forecast_predictions = pd.Series(final_forecast_predictions.values, index=future_df['ds'])
        elif model_name == 'ETS':
            # model here is the fitted ExponentialSmoothing model
            if model:
                final_forecast_predictions = model.forecast(future_periods)
                final_forecast_predictions.index = future_df['ds']
        elif model_name in ['RandomForest', 'XGBoost']:
            # model here is the fitted RF/XGBoost model
            if model:
                future_features = create_time_features(future_df.copy())
                features = ['year', 'quarter', 'month', 'dayofyear', 'weekofyear']
                existing_features_future = [f for f in features if f in future_features.columns]
                if existing_features_future:
                    X_future = future_features[existing_features_future]
                    final_forecast_predictions = pd.Series(model.predict(X_future), index=future_df['ds'])
                else:
                    print(f"ML Model forecast: Missing features for {model_name}.")
        else:
            print(f"Unknown model type for forecasting: {model_name}")
    except Exception as e:
        print(f"Error during final forecasting with {model_name}: {e}")

    return final_forecast_predictions

def detect_data_drift(new_data_df, historical_data_df, threshold_factor=0.1):
    """
    Detects data drift by comparing the mean of 'y' in new data
    against the mean of 'y' in historical data.
    This is a simple threshold-based detection.
    Args:
        new_data_df (pd.DataFrame): DataFrame containing recent data points.
        historical_data_df (pd.DataFrame): DataFrame containing historical data points.
        threshold_factor (float): Factor to determine the drift threshold
                                  (e.g., 0.1 means 10% deviation).
    Returns:
        bool: True if drift is detected, False otherwise.
    """
    if historical_data_df.empty or new_data_df.empty:
        print("Cannot detect drift: one of the dataframes is empty.")
        return False

    historical_mean = historical_data_df['y'].mean()
    new_data_mean = new_data_df['y'].mean()

    if historical_mean == 0: # Avoid division by zero
        if new_data_mean != 0:
            print(f"Drift detected: Historical mean is zero, new data mean is {new_data_mean}.")
            return True
        else:
            return False # Both are zero, no drift, no drift

    percentage_change = abs((new_data_mean - historical_mean) / historical_mean)

    print(f"Historical mean: {historical_mean:.2f}, New data mean: {new_data_mean:.2f}")
    print(f"Percentage change: {percentage_change:.2%}, Threshold: {threshold_factor:.2%}")

    if percentage_change > threshold_factor:
        print(f"Data drift detected. Percentage change ({percentage_change:.2%}) exceeds threshold ({threshold_factor:.2%}).")
        return True
    else:
        print("No significant data drift detected.")
        return False

def run_full_training_pipeline(file_path="ITSM_data.csv"):
    """
    Orchestrates the full training pipeline: loads data, preprocesses,
    trains and evaluates models, selects the best, and saves it.
    This function is designed to be called for initial training or retraining.
    Args:
        file_path (str): Path to the ITSM CSV file.
    Returns:
        bool: True if training completed successfully for at least one model, False otherwise.
    """
    print("--- Starting Full Training Pipeline ---")
    daily_incident_data = load_itsm_data(file_path=file_path)

    if daily_incident_data is None or daily_incident_data.empty:
        print("No data loaded or processed from ITSM_data.csv. Cannot train models.")
        return False

    forecast_horizon_quarterly = 4 # Number of quarters to forecast
    forecast_horizon_annually = 2  # Number of years to forecast

    trained_models_info = {} # Dictionary to store metadata about trained models

    for granularity in ['quarterly', 'annually']:
        print(f"\n--- Processing for {granularity.capitalize()} Granularity for Training ---")
        processed_data_by_category = preprocess_data(daily_incident_data, time_granularity=granularity)

        for category, df_agg in processed_data_by_category.items():
            print(f"\n--- Training for Category: {category} ({granularity.capitalize()}) ---")

            test_size = forecast_horizon_quarterly if granularity == 'quarterly' else forecast_horizon_annually
            # Ensure enough data points for training and testing
            if len(df_agg) <= test_size + 2:
                print(f"Not enough data for {granularity} split for {category}. Skipping training. (Need > {test_size+2} points, have {len(df_agg)})")
                continue

            # Train and select the best model for the current category and granularity
            best_model_name, best_model_object = train_and_select_best_model(df_agg, category, granularity, test_size)

            if best_model_name:
                # Save the best model (if it's not Naive, which doesn't have a model object)
                if best_model_name != 'Naive' and best_model_object:
                    save_model(best_model_object, category, granularity, best_model_name)
                elif best_model_name == 'Naive':
                    print(f"Naive model selected for {category}_{granularity}. No model object to save.")
                else:
                    print(f"No best model object found for {category}_{granularity}. Skipping save.")

                # Store metadata about the trained model, including a snapshot of the data
                trained_models_info[f"{category}_{granularity}"] = {
                    "model_name": best_model_name,
                    "data_snapshot": df_agg.to_json(orient='records', date_format='iso') # Store data snapshot for drift detection
                }
            else:
                print(f"No best model found for {category}_{granularity}. Skipping save.")

    # Save the overall metadata about all trained models to a JSON file
    metadata_path = os.path.join(MODEL_DIR, "trained_models_metadata.json")
    with open(metadata_path, "w") as f:
        import json
        json.dump(trained_models_info, f, indent=4)
    print(f"Model metadata saved to: {metadata_path}")
    print("--- Full Training Pipeline Completed ---")
    return True if trained_models_info else False # Return True if any model was successfully trained
