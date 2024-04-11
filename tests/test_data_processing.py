import unittest
import pandas as pd
import numpy as np
from src.data_processing import clean_data, normalize_data  

class TestDataProcessing(unittest.TestCase):
    
    def test_clean_data_handles_missing_values(self):
        # Create a DataFrame with missing values
        data = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan]
        })
        
        # Expected DataFrame after handling missing values
        expected_data = pd.DataFrame({
            'A': [1, 1, 3],
            'B': [4, 5, 5]
        })
        
        cleaned_data = clean_data(data)
        # print(cleaned_data)
        pd.testing.assert_frame_equal(cleaned_data, expected_data, check_dtype=False)
        # pd.testing.assert_frame_equal(cleaned_data, expected_data)

    def test_normalize_data(self):
        # Create a DataFrame
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        normalized_data, scaler = normalize_data(data)
        
        # Check if the normalized data is within the range 0 to 1
        self.assertTrue((normalized_data >= 0).all().all() and (normalized_data <= 1).all().all())
        
        # Check if the scaler is indeed a MinMaxScaler instance
        from sklearn.preprocessing import MinMaxScaler
        self.assertIsInstance(scaler, MinMaxScaler)
        
        # Check if the shape of the normalized data matches the input data
        self.assertEqual(normalized_data.shape, data.shape)

if __name__ == '__main__':
    unittest.main()
