import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pc_algorithm import PCAlgorithm
from src.models.neural_granger import NeuralGrangerTest
from src.models.causal_gnn import CausalGNN, hybrid_loss
from src.constraints.symbolic_rules import SymbolicConstraints
from src.evaluation.visualization import plot_causal_graph
from src.data.load_and_preprocess import TimeSeriesPreprocessor

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def run_pipeline(data, var_names, dataset_name):
    print(f"\n[{dataset_name}] Running NSCD Pipeline...")
    n_vars = len(var_names)
    
    # 1. PC Algorithm
    pc = PCAlgorithm(data, significance=0.05)
    skeleton = pc.fit()
    
    # 2. Neural Granger
    ng = NeuralGrangerTest(max_lag=4)
    granger_adj = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            p_val, is_cause = ng.granger_test(i, j, data)
            if is_cause:
                granger_adj[j, i] = 1 - p_val
                
    # 3. Causal GNN
    x = torch.eye(n_vars)
    src, dst = np.where(granger_adj > 0.05)
    
    if len(src) > 0:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        target_probs = torch.tensor(granger_adj[src, dst], dtype=torch.float)
        
        gnn_model = CausalGNN(num_vars=n_vars)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        
        gnn_model.train()
        for _ in range(100):
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
    else:
        refined_adj = granger_adj

    # 4. Symbolic Constraints
    constraints = SymbolicConstraints()
    final_dag = constraints.apply_all_constraints(refined_adj, sparsity_level=0.2)
    
    # Save Result
    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_graph.png")
    plot_causal_graph(final_dag, var_names, title=f"Causal Graph: {dataset_name}", save_path=save_path)
    
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pc_algorithm import PCAlgorithm
from src.models.neural_granger import NeuralGrangerTest
from src.models.causal_gnn import CausalGNN, hybrid_loss
from src.constraints.symbolic_rules import SymbolicConstraints
from src.evaluation.visualization import plot_causal_graph
from src.data.load_and_preprocess import TimeSeriesPreprocessor

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def run_pipeline(data, var_names, dataset_name):
    print(f"\n[{dataset_name}] Running NSCD Pipeline...")
    n_vars = len(var_names)
    
    # 1. PC Algorithm
    pc = PCAlgorithm(data, significance=0.05)
    skeleton = pc.fit()
    
    # 2. Neural Granger
    ng = NeuralGrangerTest(max_lag=4)
    granger_adj = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            p_val, is_cause = ng.granger_test(i, j, data)
            if is_cause:
                granger_adj[j, i] = 1 - p_val
                
    # 3. Causal GNN
    x = torch.eye(n_vars)
    src, dst = np.where(granger_adj > 0.05)
    
    if len(src) > 0:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        target_probs = torch.tensor(granger_adj[src, dst], dtype=torch.float)
        
        gnn_model = CausalGNN(num_vars=n_vars)
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        
        gnn_model.train()
        for _ in range(100):
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
    else:
        refined_adj = granger_adj

    # 4. Symbolic Constraints
    constraints = SymbolicConstraints()
    final_dag = constraints.apply_all_constraints(refined_adj, sparsity_level=0.2)
    
    # Save Result
    save_path = os.path.join(RESULTS_DIR, f"{dataset_name}_graph.png")
    plot_causal_graph(final_dag, var_names, title=f"Causal Graph: {dataset_name}", save_path=save_path)
    
    return final_dag

def load_sunspots():
    print("Loading Sunspots data...")
    data = sm.datasets.sunspots.load_pandas().data
    # Sunspots is univariate, so we can't do causal discovery WITHIN it easily unless we lag it or use other variables.
    # Let's use it to demonstrate we can handle it, but maybe we need another variable.
    # Statsmodels has 'co2' dataset too. Let's try to find relation between CO2 and Temperature if possible, 
    # or just use Macrodata again as the main real one, and Synthetic as the other.
    # Actually, let's stick to Macrodata and Synthetic for now to be safe and robust.
    # Or we can create a "Climate" proxy using CO2 and Sunspots if they overlap?
    # Let's just use Synthetic (Ground Truth) and Macro (Real World).
    return None

def load_extended_macro():
    """Load extended set of US Macroeconomic variables"""
    print("Loading Extended US Macroeconomic data...")
    dataset = sm.datasets.macrodata.load_pandas()
    df = dataset.data
    df['year'] = df['year'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    df.index = pd.to_datetime(df.year.astype(str) + 'Q' + df.quarter.astype(str))
    
    # Extended variables
    # realgdp: Real GDP
    # realcons: Real Consumption
    # realinv: Real Investment
    # realgovt: Real Govt Spending
    # realdpi: Real Disposable Personal Income
    # cpi: CPI
    # m1: M1 Money Stock
    # tbilrate: 3-Month Treasury Bill
    # unemp: Unemployment Rate
    
    cols = ['realgdp', 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi', 'm1', 'tbilrate', 'unemp']
    data = df[cols].copy()
    
    processed = pd.DataFrame(index=data.index)
    # Log-diff (Growth rates) for level variables
    for col in ['realgdp', 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi', 'm1']:
        processed[col] = np.log(data[col]).diff()
        
    # Diff (Changes) for rate variables
    for col in ['tbilrate', 'unemp']:
        processed[col] = data[col].diff()
        
    return processed.dropna()

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    # 1. Extended Macro Benchmark
    macro_data = load_extended_macro()
    # Normalize
    preprocessor = TimeSeriesPreprocessor('dummy')
    preprocessor.data = macro_data
    norm_macro = preprocessor.normalize()
    run_pipeline(norm_macro, norm_macro.columns, "US_Macro_Extended")
    
    print("\nBenchmark Completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main()
