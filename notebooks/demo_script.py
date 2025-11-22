import sys
import os
import numpy as np
import pandas as pd
import networkx as nx
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pc_algorithm import PCAlgorithm
from src.models.neural_granger import NeuralGrangerTest
from src.models.causal_gnn import CausalGNN, hybrid_loss
from src.constraints.symbolic_rules import SymbolicConstraints
from src.evaluation.metrics import CausalGraphEvaluator

def generate_synthetic_data(n_samples=1000, n_vars=5):
    """Generate synthetic time-series data with known causal structure"""
    np.random.seed(42)
    data = np.zeros((n_samples, n_vars))
    
    # Causal structure: 0 -> 1, 1 -> 2, 2 -> 3, 4 is independent
    # Linear and non-linear relationships
    for t in range(1, n_samples):
        data[t, 0] = 0.8 * data[t-1, 0] + np.random.normal(0, 0.1)
        data[t, 1] = 0.5 * data[t-1, 1] + 0.5 * np.sin(data[t-1, 0]) + np.random.normal(0, 0.1)
        data[t, 2] = 0.7 * data[t-1, 2] - 0.3 * data[t-1, 1] + np.random.normal(0, 0.1)
        data[t, 3] = 0.6 * data[t-1, 3] + 0.4 * (data[t-1, 2] ** 2) + np.random.normal(0, 0.1)
        data[t, 4] = 0.9 * data[t-1, 4] + np.random.normal(0, 0.1)
        
    df = pd.DataFrame(data, columns=[f'V{i}' for i in range(n_vars)])
    
    # Ground truth adjacency matrix
    true_adj = np.zeros((n_vars, n_vars))
    true_adj[0, 1] = 1
    true_adj[1, 2] = 1
    true_adj[2, 3] = 1
    
    return df, true_adj

def main():
    print("Generating synthetic data...")
    data, true_adj = generate_synthetic_data()
    print("Data shape:", data.shape)
    
    # 1. PC Algorithm (Skeleton)
    print("\nRunning PC Algorithm...")
    pc = PCAlgorithm(data, significance=0.05)
    skeleton = pc.fit()
    print("PC Skeleton edges:", skeleton.edges())
    
    # 2. Neural Granger (Orientation & Non-linearity)
    print("\nRunning Neural Granger...")
    ng = NeuralGrangerTest(max_lag=3)
    n_vars = data.shape[1]
    granger_adj = np.zeros((n_vars, n_vars))
    
    # We only test edges present in skeleton (or all if we want to be thorough, but prompt says "Detect... that PC misses" implying we might want to test all, or refine skeleton. 
    # Usually PC gives undirected skeleton. Granger gives direction.
    # Let's test all pairs for this demo to see if it recovers the structure.
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j: continue
            p_val, is_cause = ng.granger_test(j, i, data) # Does i cause j?
            if is_cause:
                granger_adj[i, j] = 1 - p_val # Use 1-p_val as "strength" proxy
                
    print("Neural Granger Adjacency (Thresholded):")
    print((granger_adj > 0.95).astype(int)) # High confidence
    
    # 3. Causal GNN (Refinement)
    print("\nRunning Causal GNN...")
    # Prepare data for GNN
    # Nodes: features (e.g., correlation with others or just identity). 
    # Let's use identity (one-hot) for simplicity as per prompt hint.
    x = torch.eye(n_vars) 
    
    # Initial edges from Granger (or PC + Granger)
    # We'll use Granger output as the "noisy graph" to refine
    src, dst = np.where(granger_adj > 0.05) # Loose threshold to get candidates
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    gnn_model = CausalGNN(num_vars=n_vars)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    
    # Dummy training loop (in real scenario we need ground truth or self-supervised objective)
    # Since this is unsupervised/discovery, we usually use the Hybrid Loss with some proxy or iterative refinement.
    # The prompt says "Training is guided by a Hybrid Loss function that balances data fidelity (Neural) with sparsity preference (Symbolic)."
    # "Fidelity to Neural Granger result" -> So we treat Granger result as "noisy labels"?
    # Let's assume we train to approximate Granger probabilities but with sparsity/structure constraints.
    
    target_probs = torch.tensor(granger_adj[src, dst], dtype=torch.float)
    
    gnn_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        edge_probs, _ = gnn_model(x, edge_index)
        edge_probs = edge_probs.squeeze()
        
        # Loss: match Granger probabilities but be sparse
        loss = hybrid_loss(edge_probs, target_probs)
        loss.backward()
        optimizer.step()
        
    gnn_model.eval()
    with torch.no_grad():
        final_probs, _ = gnn_model(x, edge_index)
        final_probs = final_probs.squeeze().numpy()
        
    # Construct refined adjacency
    refined_adj = np.zeros((n_vars, n_vars))
    refined_adj[src, dst] = final_probs
    
    # 4. Symbolic Constraints
    print("\nApplying Symbolic Constraints...")
    constraints = SymbolicConstraints()
    final_dag = constraints.apply_all_constraints(refined_adj, sparsity_level=0.3)
    
    print("Final DAG Adjacency:")
    print(final_dag)
    
    # Evaluation
    print("\nEvaluation:")
    evaluator = CausalGraphEvaluator(final_dag, true_adj)
    metrics = evaluator.all_metrics()
    print(metrics)

if __name__ == "__main__":
    main()
