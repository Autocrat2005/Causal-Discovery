import numpy as np
from scipy import stats
from itertools import combinations
import networkx as nx

def partial_correlation(x, y, z_indices, data):
    """Compute partial correlation: corr(x, y | z) for CI test"""
    data_values = data.values
    x_idx, y_idx = x, y
    
    if len(z_indices) == 0:
        # Simple correlation
        corr = np.corrcoef(data_values[:, x_idx], data_values[:, y_idx])[0, 1]
    else:
        # Compute residuals after OLS regression on Z
        X = data_values[:, [x_idx] + list(z_indices)]
        Y = data_values[:, [y_idx] + list(z_indices)]
        
        # OLS regression to get residuals
        # We regress x on z, and y on z
        # X[:, 0] is the target (x or y), X[:, 1:] are the predictors (z)
        
        # For x:
        z_data = data_values[:, list(z_indices)]
        x_data = data_values[:, x_idx]
        y_data = data_values[:, y_idx]

        # Add intercept
        Z = np.column_stack([np.ones(len(z_data)), z_data])
        
        # Beta = (Z^T Z)^-1 Z^T X
        # Using lstsq for numerical stability
        beta_x = np.linalg.lstsq(Z, x_data, rcond=None)[0]
        x_resid = x_data - Z @ beta_x
        
        beta_y = np.linalg.lstsq(Z, y_data, rcond=None)[0]
        y_resid = y_data - Z @ beta_y
        
        corr = np.corrcoef(x_resid, y_resid)[0, 1]
    
    if np.isnan(corr):
        return 1.0 # Treat as independent if correlation is undefined

    # Fisher z-transform for significance testing
    n = len(data_values) - len(z_indices)
    if n <= 3:
        return 1.0 # Not enough samples
        
    # Clip correlation to avoid domain errors in log
    corr = np.clip(corr, -0.99999, 0.99999)
    
    z = 0.5 * np.log((1 + corr) / (1 - corr))
    t_stat = z * np.sqrt(n - 3)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
    
    return p_value

class PCAlgorithm:
    def __init__(self, data, significance=0.05):
        self.data = data
        self.n_vars = data.shape[1]
        self.significance = significance
        
    def fit(self):
        """Run PC algorithm loop to learn causal skeleton"""
        # Start with a complete undirected graph (represented as directed for now, but we'll treat edges symmetrically)
        graph = nx.complete_graph(self.n_vars).to_directed()
        
        depth = 0
        while True:
            edge_removed = False
            
            # Get current edges (convert to list to avoid modification during iteration)
            edges = list(graph.edges())
            
            for x, y in edges:
                if not graph.has_edge(x, y):
                    continue
                
                # Neighbors of x excluding y
                neighbors = set(graph.neighbors(x)) - {y}
                
                if len(neighbors) < depth:
                    continue
                    
                for z_set in combinations(neighbors, depth):
                    p_val = partial_correlation(x, y, z_set, self.data)
                    
                    if p_val > self.significance:
                        if graph.has_edge(x, y):
                            graph.remove_edge(x, y)
                        if graph.has_edge(y, x):
                            graph.remove_edge(y, x)
                        edge_removed = True
                        break # Edge removed, move to next edge
            
            depth += 1
            # Stop if max depth reached (number of variables - 2) or no edges removed in this depth
            if not edge_removed and depth > 0: # Allow at least depth 0 to run
                 # Actually, standard PC stops when max degree of any node is less than depth
                 # But for simplicity we can check if any edge was removed. 
                 # If no edge removed at depth k, it's unlikely to remove at k+1? 
                 # Standard PC: stop if for every adjacent pair X, Y, |adj(X)\{Y}| < depth
                 max_degree = 0
                 if graph.number_of_nodes() > 0:
                     max_degree = max([d for n, d in graph.degree()])
                 
                 if depth > max_degree:
                     break
            
            if depth >= self.n_vars - 1:
                break
        
        return graph
