from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Placeholder for loading your model
# For demonstration, this part is skipped. In practice, you'd load or retrain your model here.

@app.route('/predict', methods=['GET'])
def predict():
    # Number of days to predict, default is 30
    days = request.args.get('days', default=30, type=int)
    
    # Generate future dataframe for the specified number of days
    future = pd.DataFrame({'ds': pd.date_range(start=pd.Timestamp.today(), periods=days, freq='D')})
    
    # Assuming 'model' is your preloaded or dynamically trained Prophet model
    model = Prophet()  # Placeholder: In practice, load or initialize your model appropriately
    
    # Make predictions (this part should be adapted based on how your actual model is loaded and used)
    forecast = model.predict(future)
    
    # For simplicity, returning just the 'ds' (date) and 'yhat' (predicted value) columns as a JSON response
    predictions = forecast[['ds', 'yhat']].to_dict(orient='records')
    
    # Log the prediction
    logging.info(f'Predicted {days} days of stock prices.')

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
