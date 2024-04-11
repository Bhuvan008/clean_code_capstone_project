from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

def build_and_tune_model(data, param_grid, periods=30):
    """
    Build and tune a Prophet forecasting model using the given data and parameter grid.

    Parameters:
    data (pandas.DataFrame): The historical data to fit the model on. Must include 'Date' and 'Close' columns.
    param_grid (dict): A dictionary with parameter names (string) as keys and lists of parameter settings to try as values.
    periods (int): Number of periods to forecast into the future.

    Returns:
    tuple: The best performing Prophet model and the forecast DataFrame.
    """
    
    # Prepare the data for the Prophet model
    prophet_data = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize a list to store the results for each parameter combination
    all_results = []
    # Loop through all possible combinations of parameters
    for params in ParameterGrid(param_grid):
        # Create and fit a new Prophet model with the given parameters
        model = Prophet(**params).fit(prophet_data)
        # Perform cross-validation
        cv_results = cross_validation(model, initial='180 days', period='14 days', horizon='30 days')
        # Calculate performance metrics for the cross-validation results
        performance = performance_metrics(cv_results)
        # Store the parameters and their corresponding MAPE (Mean Absolute Percentage Error)
        all_results.append((params, performance['mape'].mean()))
    
    # Identify the best parameters and score based on the lowest MAPE
    best_params, best_score = min(all_results, key=lambda x: x[1])
    print("Best MAPE:", best_score, "with parameters:", best_params)
    
    # Re-train the model with the best parameters on the full dataset
    best_model = Prophet(**best_params).fit(prophet_data)
    # Create a future DataFrame for the specified number of periods
    future = best_model.make_future_dataframe(periods=periods)
    # Predict the future trends
    forecast = best_model.predict(future)
    # Plot the forecast results
    best_model.plot(forecast)
    
    return best_model, forecast
