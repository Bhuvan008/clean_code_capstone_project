import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetches historical stock data for a given symbol between start_date and end_date.

    Parameters:
    symbol (str): The stock symbol to fetch the data for.
    start_date (str): The start date for the data in YYYY-MM-DD format.
    end_date (str): The end date for the data in YYYY-MM-DD format.

    Returns:
    pandas.DataFrame: A DataFrame with historical stock data for the given symbol.
    """
    try:
        # Download the stock data using yfinance
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error fetching stock data: {e}")
        return None

def save_to_csv(data, file_path):
    """
    Saves a DataFrame to a CSV file at the specified file_path.

    Parameters:
    data (pandas.DataFrame): The DataFrame to be saved to CSV.
    file_path (str): The file path where the CSV will be saved.

    Returns:
    None
    """
    try:
        # Save the DataFrame to a CSV file
        data.to_csv(file_path, index=True)
        print(f"Data saved to {file_path}")
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error saving data: {e}")
