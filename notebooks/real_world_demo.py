import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pc_algorithm import PCAlgorithm
from src.models.neural_granger import NeuralGrangerTest
from src.models.causal_gnn import CausalGNN, hybrid_loss
from src.constraints.symbolic_rules import SymbolicConstraints
from src.data.load_and_preprocess import TimeSeriesPreprocessor

def load_macro_data():
    """Load US Macroeconomic data from statsmodels"""
    print("Loading US Macroeconomic data...")
    dataset = sm.datasets.macrodata.load_pandas()
    df = dataset.data
    
    # Create a date index
    df['year'] = df['year'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    df.index = pd.to_datetime(df.year.astype(str) + 'Q' + df.quarter.astype(str))
    
    # Select key variables
    # realgdp: Real Gross Domestic Product
    # cpi: Consumer Price Index
    # unemp: Unemployment Rate
    # tbilrate: 3-Month Treasury Bill Rate (Interest Rate proxy)
    cols = ['realgdp', 'cpi', 'unemp', 'tbilrate']
    data = df[cols].copy()
    
    return data

def preprocess_data(data):
    """Make data stationary (Growth rates / Differences)"""
    print("\nPreprocessing data...")
    processed = pd.DataFrame(index=data.index)
    
    # Log-difference for GDP and CPI (Growth rates)
    processed['GDP_Growth'] = np.log(data['realgdp']).diff()
    processed['Inflation'] = np.log(data['cpi']).diff()
    
    # Difference for Unemployment and Interest Rate (Changes)
    processed['Unemp_Change'] = data['unemp'].diff()
    processed['Rate_Change'] = data['tbilrate'].diff()
    
    # Drop NaN from differencing
    processed = processed.dropna()
    
    # Check stationarity
    preprocessor = TimeSeriesPreprocessor('dummy_path') # We just need the method
    for col in processed.columns:
        preprocessor.check_stationarity(processed[col], name=col)
        
    # Normalize
    scaler = preprocessor
    scaler.data = processed
    normalized = scaler.normalize()
    
    return normalized

def main():
    # 1. Load and Preprocess
    raw_data = load_macro_data()
    data = preprocess_data(raw_data)
    var_names = data.columns.tolist()
    n_vars = len(var_names)
    print(f"\nVariables: {var_names}")
    print(f"Data shape: {data.shape}")
    
    # 2. PC Algorithm (Skeleton)
    print("\n--- Stage 1: PC Algorithm (Skeleton) ---")
    pc = PCAlgorithm(data, significance=0.05)
    skeleton = pc.fit()
    print("PC Skeleton edges:")
    for u, v in skeleton.edges():
        print(f"{var_names[u]} -- {var_names[v]}")
    
    # 3. Neural Granger (Orientation)
    print("\n--- Stage 2: Neural Granger (Non-linear Direction) ---")
    ng = NeuralGrangerTest(max_lag=4) # 4 quarters = 1 year lag
    granger_adj = np.zeros((n_vars, n_vars))
    
    # Test all pairs
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            # Does j cause i?
            p_val, is_cause = ng.granger_test(i, j, data) 
            if is_cause:
                granger_adj[j, i] = 1 - p_val
                print(f"Detected: {var_names[j]} -> {var_names[i]} (p={p_val:.4f})")
                
    # 4. Causal GNN (Refinement)
    print("\n--- Stage 3: Causal GNN (Refinement) ---")
    x = torch.eye(n_vars) 
    src, dst = np.where(granger_adj > 0.05)
    if len(src) == 0:
        print("No edges detected by Granger. Skipping GNN.")
        refined_adj = granger_adj
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        target_probs = torch.tensor(granger_adj[src, dst], dtype=torch.float)
        
        gnn_model = CausalGNN(num_vars=n_vars)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        
        gnn_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            edge_probs, _ = gnn_model(x, edge_index)
            edge_probs = edge_probs.squeeze()
            loss = hybrid_loss(edge_probs, target_probs)
            loss.backward()
            optimizer.step()
            
        gnn_model.eval()
        with torch.no_grad():
            final_probs, _ = gnn_model(x, edge_index)
            final_probs = final_probs.squeeze().numpy()
            
        refined_adj = np.zeros((n_vars, n_vars))
        refined_adj[src, dst] = final_probs
    
    # 5. Symbolic Constraints
    print("\n--- Stage 4: Symbolic Constraints (Final DAG) ---")
    constraints = SymbolicConstraints()
    final_dag = constraints.apply_all_constraints(refined_adj, sparsity_level=0.2)
    
    print("\nFinal Discovered Causal Graph:")
    rows, cols = np.where(final_dag)
    if len(rows) == 0:
        print("No causal relationships found.")
    else:
        for u, v in zip(rows, cols):
            print(f"{var_names[u]} --> {var_names[v]}")

if __name__ == "__main__":
    main()
