import unittest
import numpy as np
from src.model_evaluation import evaluate_model

class TestModelEvaluation(unittest.TestCase):

    def test_evaluate_model(self):
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.0, 2.9, 4.1, 4.9])
        
        # Calculate expected results
        expected_mae = np.mean(np.abs(actual - predicted))
        expected_mse = np.mean(np.square(actual - predicted))
        expected_mape = np.mean(np.abs((actual - predicted) / actual))
        expected_rmse = np.sqrt(expected_mse)
        
        # Call the evaluate_model function
        results = evaluate_model(actual, predicted)
        
        # Assert the results
        self.assertAlmostEqual(results['MAE'], expected_mae)
        self.assertAlmostEqual(results['MSE'], expected_mse)
        self.assertAlmostEqual(results['MAPE'], expected_mape)
        self.assertAlmostEqual(results['RMSE'], expected_rmse)

if __name__ == '__main__':
    unittest.main()
