Stock Market Prediction Project:

Dependencies:
    - Flask: A lightweight WSGI web application framework
    - pandas: Data manipulation and analysis library
    - numpy: Fundamental package for scientific computing with Python
    - yfinance: Yahoo Finance market data downloader
    - fbprophet: Forecasting library for time series data
    - scikit-learn: Machine learning library for the Python programming language
    - pickle: For compatibility with pickle in Python versions before 3.8

Usage:
    1. Ensure the above packages are installed.

    2. Run the Flask app to deploy the machine learning model for predicting stock prices:
       ```
       python deploy.py
       ```

    3. Use the provided functions to fetch stock data, preprocess it, build the model with HyperParameter tuning and perform predictions.

Author Information:
    Name: Bhavani Bhasutkar
    Date: 16 Mar 2024

Abstract/Description:

This project is aimed at predicting stock market prices using historical data. It involves fetching stock data using the yfinance library, preprocessing this data with pandas and numpy, and then applying machine learning and time series forecasting techniques using scikit-learn and Prophet. The project also includes a Flask application for deploying the machine learning model, allowing for easy prediction and analysis of stock data.

Change Log:
    - 16 Mar 2024: Initial creation.
    - [Date]: Updated with new features or data. -- Use this format for updates.
