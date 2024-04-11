import sys
import os
import pandas as pd
import pickle

# Add the parent directory to sys.path to make the 'src' package available
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import necessary functions from their respective modules
from src.data_collection import fetch_stock_data, save_to_csv
from src.data_processing import clean_data
from src.feature_engineering import calculate_moving_average, calculate_rsi
from src.model_building import build_and_tune_model
from src.model_evaluation import evaluate_model
from src.utils import save_model

def main():
# Define parameters for data collection
    symbol = "AMZN"
    start_date = "2023-01-01"
    end_date = "2024-01-31"
    raw_data_file_path = "data/raw/amazon_raw_data.csv"
    processed_data_file_path = "data/processed/amazon_processed_data.csv"
    model_save_path = "models/amazon_model.pkl"

    # Step 1: Fetch and save stock data
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    save_to_csv(stock_data, raw_data_file_path)

    # Step 2: Clean data
    clean_stock_data = clean_data(raw_data_file_path)


    # Step 3: Feature Engineering
    clean_stock_data['rolling_avg_30d'] = calculate_moving_average(clean_stock_data['Close'])
    clean_stock_data['rsi'] = calculate_rsi(clean_stock_data['Close'])

    save_to_csv(clean_stock_data, processed_data_file_path)
    # Assume the cleaned data is now ready for modeling and saved properly
    # The cleaned data should be in a format acceptable by the Prophet model

    # Step 4: Model Building and Hyperparameter Tuning
    param_grid = {
        "changepoint_prior_scale": [0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.1, 1.0, 10.0]
        # "changepoint_prior_scale": [0.01],
        # "seasonality_prior_scale": [0.1],
    }
    best_model, forecast = build_and_tune_model(clean_stock_data, param_grid)


    # # Step 5: Evaluate Model
    # # Loading seperate validation data set of Jan Month and doing necessary processing
    # val_data = pd.read_csv('data/raw/amazon_raw_val_data.csv')
    # val_data['Date'] = pd.to_datetime(val_data['Date'])
    # prophet_val_data = val_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # forecasted_values = forecast[['ds', 'yhat']].merge(prophet_val_data, on='ds')

    # #removing the train data, to calculate the metrics on Val data
    # forecasted_values = forecasted_values[forecasted_values['ds'] >= '2024-01-01']

    # print(evaluate_model(forecasted_values['y'], forecasted_values['yhat']))

    # Step 6: Save the Model
    save_model(best_model, model_save_path)



if __name__ == '__main__':
    main()