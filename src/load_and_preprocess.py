import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

class TimeSeriesPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.normalized_data = None
        
    def load_data(self):
        """Load CSV and handle missing values"""
        df = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
        # Forward fill then backward fill for robustness
        df = df.fillna(method='ffill').fillna(method='bfill')
        self.data = df
        return df
    
    def check_stationarity(self, series, name=''):
        """Check if time-series is stationary (Augmented Dickey-Fuller test)"""
        # Drop NA values before passing to adfuller to avoid errors
        clean_series = series.dropna()
        if len(clean_series) == 0:
            print(f"{name}: Empty series, cannot check stationarity.")
            return False
            
        result = adfuller(clean_series)
        is_stationary = result[1] < 0.05 
        print(f"{name}: p-value = {result[1]:.4f}, Stationary: {is_stationary}")
        return is_stationary
    
    def normalize(self):
        """Standardize all variables (zero mean, unit variance)"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        scaler = StandardScaler()
        self.normalized_data = pd.DataFrame(
            scaler.fit_transform(self.data),
            columns=self.data.columns,
            index=self.data.index
        )
        return self.normalized_data
