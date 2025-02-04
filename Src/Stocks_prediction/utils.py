import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_returns(data):
    """Calculate daily returns"""
    return data['close'].pct_change()

def calculate_volatility(data, window=20):
    """Calculate rolling volatility"""
    returns = calculate_returns(data)
    return returns.rolling(window=window).std() * np.sqrt(window)

def get_performance_metrics(true_values, predictions):
    """Calculate various performance metrics"""
    metrics = {
        'mse': np.mean((true_values - predictions) ** 2),
        'rmse': np.sqrt(np.mean((true_values - predictions) ** 2)),
        'mae': np.mean(np.abs(true_values - predictions)),
        'mape': np.mean(np.abs((true_values - predictions) / true_values)) * 100
    }
    return metrics

def prepare_next_day_prediction(data, sequence_length=5):
    """Prepare the most recent data for next-day prediction"""
    return data[-sequence_length:].values.reshape(1, sequence_length, -1)