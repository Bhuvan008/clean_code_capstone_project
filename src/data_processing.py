import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(raw_data_file_path):
    """
    Cleans the raw data from a CSV file by handling missing values.
    
    Parameters:
    raw_data_file_path (str): The file path of the raw data CSV file.
    
    Returns:
    pandas.DataFrame: The cleaned data with missing values handled.
    """
    # Read the data from the CSV file
    data = pd.read_csv(raw_data_file_path)
    
    # Check for missing values and handle them. This example fills them with the previous value.
    if data.isnull().sum().sum() > 0:
        data.fillna(method='ffill', inplace=True)
        print("Missing values handled.")
    
    return data

def normalize_data(data):
    """
    Normalizes the data using the MinMaxScaler from scikit-learn.
    
    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data to be normalized.
    
    Returns:
    tuple: A tuple containing the normalized data as a DataFrame and the scaler object.
    """
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fit the scaler to the data and transform it
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    
    print("Data normalized.")
    return data_scaled, scaler
