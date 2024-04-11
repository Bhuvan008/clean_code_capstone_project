import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

def save_model(model, filepath):
    """
    Save the given model to a specified file path using Python's pickle module.

    Parameters:
    model (object): The model object to be saved. This can be any Python object that is pickle-able.
    filepath (str): The destination file path for the saved model.
    
    Returns:
    None
    """
    # Open the file in write-binary mode and save the model using pickle
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)


def plot_actual_vs_predicted(df, date_col, actual_col, predicted_col, title='Actual vs Predicted', x_label='Date', y_label='Values'):
    """
    Plots actual vs predicted values over dates from a pandas DataFrame.
    
    Parameters:
    - df: Pandas DataFrame containing the data.
    - date_col: String, the name of the column in df that contains the dates.
    - actual_col: String, the name of the column in df that contains the actual values.
    - predicted_col: String, the name of the column in df that contains the predicted values.
    - title: String, the title of the graph.
    - x_label: String, label for the x-axis.
    - y_label: String, label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[date_col], df[actual_col], label='Actual', color='blue')
    plt.plot(df[date_col], df[predicted_col], label='Predicted', color='red')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))  # Adjust interval to fit your data's density
    plt.gcf().autofmt_xdate()  # Rotation
    plt.grid(True)
    plt.show()
