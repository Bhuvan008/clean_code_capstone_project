import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def evaluate_model(actual, predicted):
    """
    Evaluate time series model predictions with common metrics: MAE, MSE, and RMSE.
    
    Parameters:
    - actual: Array-like, true values.
    - predicted: Array-like, model's predictions.
    
    Returns:
    A dictionary with MAE, MSE, and RMSE values.
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual,predicted)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape,
        'RMSE': rmse
    }
