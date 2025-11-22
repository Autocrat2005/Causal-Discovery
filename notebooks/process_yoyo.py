import pandas as pd
import numpy as np
import os
import sys
import pickle
import networkx as nx

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import NSCDPipeline
from src.evaluation.stability import StabilityAnalyzer
from src.data.load_and_preprocess import TimeSeriesPreprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'yoyo.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'yoyo_results.pkl')

def load_yoyo():
    print("Loading yoyo.csv...")
    # Parse dates with dayfirst=True for DD-MM-YYYY format
    df = pd.read_csv(DATA_PATH, parse_dates=['date'], dayfirst=True, index_col='date')
    
    # Select key variables to avoid overwhelming the graph
    # T: Temperature, rh: Relative Humidity, p: Pressure, 
    # SWDR: Radiation (Driver), rain: Rain, wv: Wind Velocity
    # VPact: Vapor Pressure (Derived from T and rh, so might be redundant/deterministic)
    
    cols = ['T', 'rh', 'p', 'SWDR', 'rain', 'wv', 'VPact']
    data = df[cols].copy()
    
    # Resample to hourly to reduce noise and data size (52k -> ~2k)
    # This makes causal discovery faster and often more robust for weather
    data = data.resample('1H').mean().dropna()
    
    return data

def main():
    data = load_yoyo()
    print(f"Data loaded: {data.shape}")
    
    # Normalize
    preprocessor = TimeSeriesPreprocessor('dummy')
    preprocessor.data = data
    norm_data = preprocessor.normalize()
    
    var_names = data.columns.tolist()
    
    # 1. Run Pipeline
    print("Running NSCD Pipeline...")
    pipeline = NSCDPipeline(max_lag=2) # Lower lag for hourly data
    adj_matrix = pipeline.run(norm_data, var_names)
    
    # 2. Stability Analysis
    print("Running Stability Analysis...")
    analyzer = StabilityAnalyzer(n_bootstraps=5) # 5 for speed in this setup script
    stability_scores, stable_dag = analyzer.run(norm_data, var_names)
    
    # 3. Save Results
    results = {
        'data_head': data.head(),
        'var_names': var_names,
        'adj_matrix': adj_matrix,
        'stability_scores': stability_scores,
        'stable_dag': stable_dag,
        'description': "Yoyo Dataset: Hourly Weather Data (T, RH, P, Radiation, Rain, Wind, Vapor Pressure)"
    }
    
    if not os.path.exists(os.path.dirname(RESULTS_PATH)):
        os.makedirs(os.path.dirname(RESULTS_PATH))
        
    with open(RESULTS_PATH, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
