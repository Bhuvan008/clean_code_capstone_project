from prophet import Prophet
import pickle
import numpy as np
import pandas as pd
import os, sys

# Add the parent directory to sys.path to make the 'src' package available
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.model_evaluation import evaluate_model
with open('models/amazon_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Load the test data for monitoring
X_test = np.load('models/X_test.npy')
y_test = np.load('models/y_test.npy')


future = model.make_future_dataframe(periods=30, freq='D')
forecast = model.predict(future)

test_data = pd.DataFrame({'ds':X_test,'y':y_test})


forecasted_values = forecast[['ds', 'yhat']].merge(test_data, on='ds')

#removing the train data, to calculate the metrics on Val data
forecasted_values = forecasted_values[forecasted_values['ds'] >= '2024-02-01']


print(evaluate_model(forecasted_values['y'], forecasted_values['yhat']))



