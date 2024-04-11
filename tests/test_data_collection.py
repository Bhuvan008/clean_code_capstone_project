import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import os



from src.data_collection import fetch_stock_data, save_to_csv  

class TestStockFunctions(unittest.TestCase):
    
    @patch('yfinance.download')
    def test_fetch_stock_data_success(self, mock_download):
        # Mock successful response from yfinance.download
        mock_data = pd.DataFrame(data={'Close': [100, 101, 102]})
        mock_download.return_value = mock_data
        
        symbol = 'AMZN'
        start_date = '2023-01-01'
        end_date = '2023-01-07'
        result = fetch_stock_data(symbol, start_date, end_date)
        
        mock_download.assert_called_once_with(symbol, start=start_date, end=end_date)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Assuming mock_data has 3 rows

    @patch('yfinance.download')
    def test_fetch_stock_data_failure(self, mock_download):
        # Simulate an exception in yfinance.download
        mock_download.side_effect = Exception('Failed to fetch data')
        
        result = fetch_stock_data('AMZN', '2023-01-01', '2023-01-07')
        self.assertIsNone(result)
    

if __name__ == '__main__':
    unittest.main()
