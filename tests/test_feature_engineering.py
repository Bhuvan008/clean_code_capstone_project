import unittest
import pandas as pd
from src.feature_engineering import calculate_moving_average, calculate_rsi

class TestFinancialAnalysis(unittest.TestCase):
    
    def test_calculate_moving_average(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        expected_result = pd.Series([None, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        
        moving_average = calculate_moving_average(data, window_size)
        pd.testing.assert_series_equal(moving_average, expected_result, check_names=False, check_dtype=False)
        
    def test_calculate_rsi(self):
        data = pd.Series([10, 12, 11, 13, 13, 15, 17, 16 ])
        window_size = 3
        expected_rsi = pd.Series([None, None, 66.666667, 80.0, 66.666667, 100.0, 100.0, 80.0])
        
        rsi = calculate_rsi(data, window_size)
        pd.testing.assert_series_equal(rsi.fillna(method='bfill'), expected_rsi.fillna(method='bfill'), check_names=False,check_dtype=False)
        
    def test_calculate_moving_average_with_empty_data(self):
        data = pd.Series([])
        expected_result = pd.Series([])
        
        moving_average = calculate_moving_average(data)
        pd.testing.assert_series_equal(moving_average, expected_result, check_names=False,check_dtype=False)
        
    def test_calculate_rsi_with_empty_data(self):
        data = pd.Series([])
        expected_result = pd.Series([])
        
        rsi = calculate_rsi(data)
        pd.testing.assert_series_equal(rsi, expected_result, check_names=False,check_dtype=False)

if __name__ == '__main__':
    unittest.main()
