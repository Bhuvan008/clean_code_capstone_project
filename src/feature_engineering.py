import pandas as pd

def calculate_moving_average(data, window_size=30):
    """
    Calculate the moving average of the given data over a specified window size.

    Parameters:
    data (pandas.Series or pandas.DataFrame): The data to calculate the moving average on.
    window_size (int, optional): The number of periods to consider for the moving average. Default is 30.

    Returns:
    pandas.Series or pandas.DataFrame: The moving average of the data.
    """
    try:
        # Calculate and return the rolling mean
        return data.rolling(window=window_size).mean()
    except Exception as e:
        # If an error occurs, print an error message and return None
        print(f"Error calculating moving average: {e}")
        return None

def calculate_rsi(data, window_size=14):
    """
    Calculate the Relative Strength Index (RSI) for the given data over a specified window size.

    Parameters:
    data (pandas.Series): The data to calculate the RSI on.
    window_size (int, optional): The number of periods to consider for the RSI calculation. Default is 14.

    Returns:
    pandas.Series: The RSI of the data.
    """
    try:
        # Calculate price changes
        delta = data.diff(1)
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
        # Calculate the RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        # If an error occurs, print an error message and return None
        print(f"Error calculating RSI: {e}")
        return None
