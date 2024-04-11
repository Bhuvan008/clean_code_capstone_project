import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.model_building import build_and_tune_model

class TestBuildAndTuneModel(unittest.TestCase):
    @patch('src.model_building.Prophet')
    @patch('src.model_building.cross_validation')
    @patch('src.model_building.performance_metrics')
    def test_build_and_tune_model(self, mock_performance_metrics, mock_cross_validation, mock_Prophet):
        # Setup mock data
        data = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=100),
            'y': np.random.rand(100)
        })
        
        # Configure the mock objects
        mock_model = MagicMock()
        mock_Prophet.return_value = mock_model
        mock_model.fit.return_value = mock_model
        mock_model.make_future_dataframe.return_value = pd.DataFrame({
            'ds': pd.date_range(start='2020-04-10', periods=30)
        })
        mock_model.predict.return_value = pd.DataFrame({
            'ds': pd.date_range(start='2020-04-10', periods=30),
            'yhat': np.random.rand(30)
        })
        mock_cross_validation.return_value = pd.DataFrame({
            'ds': pd.date_range(start='2020-04-10', periods=30),
            'yhat': np.random.rand(30),
            'y': np.random.rand(30)
        })
        mock_performance_metrics.return_value = pd.DataFrame({
            'mape': [0.1, 0.2, 0.15]
        })
        
        param_grid = {'seasonality_mode': ('multiplicative', 'additive')}
        
        # Execute the function under test
        best_model, forecast = build_and_tune_model(data, param_grid)
        
        # Assertions to verify the expected behavior
        self.assertIsInstance(best_model, MagicMock)  # This checks if the returned model is indeed a mock
        self.assertEqual(len(forecast), 30)  # Assuming the forecast dataframe has 30 rows
        mock_Prophet.assert_called()  # Checks if Prophet was called
        mock_cross_validation.assert_called()  # Checks if cross_validation was called
        mock_performance_metrics.assert_called()  # Checks if performance_metrics was called

if __name__ == '__main__':
    unittest.main()
