import numpy as np
import networkx as nx

class SymbolicConstraints:
    def __init__(self):
        pass
        
    def sparsity_constraint(self, adj_matrix_probs, sparsity_level=0.2):
        """Keep only top (1-level)% of edges by strength (probability)"""
        adj = adj_matrix_probs.copy()
        edges = []
        
        rows, cols = adj.shape
        for i in range(rows):
            for j in range(cols):
                if adj[i, j] > 0:
                    edges.append((i, j, adj[i, j]))
        
        if edges:
            edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
            n_keep = max(1, int(len(edges_sorted) * (1 - sparsity_level)))
            
            # Zero out weak edges below threshold
            # If n_keep is less than total edges, we cut off the rest
            if n_keep < len(edges):
                threshold = edges_sorted[n_keep-1][2]
                # We want to keep edges >= threshold. 
                # But if multiple edges have same prob as threshold, we might keep more or less.
                # Simple approach: set anything < threshold to 0.
                adj[adj < threshold] = 0
            
        return (adj > 0).astype(int)

    def no_cycles(self, adj_matrix, edge_probs):
        """Iteratively remove the weakest edge in a cycle until DAG is formed"""
        adj = adj_matrix.copy()
        
        while True:
            G = nx.DiGraph(adj)
            try:
                # Find a cycle
                cycle = nx.find_cycle(G)
                
                # Find the weakest edge in the cycle
                weakest_edge = None
                min_prob = float('inf')
                
                for u, v in cycle:
                    # u and v are indices in the adjacency matrix
                    prob = edge_probs[u, v]
                    if prob < min_prob:
                        min_prob = prob
                        weakest_edge = (u, v)
                        
                # Remove the weakest edge
                if weakest_edge:
                    adj[weakest_edge[0], weakest_edge[1]] = 0
                else:
                    # Should not happen if cycle exists
                    break
                    
            except nx.NetworkXNoCycle:
                break
        
        return adj
    
    def apply_all_constraints(self, adj_probs, sparsity_level=0.3):
        """Apply all constraints in order"""
        # Apply soft constraint (sparsity) based on probabilities
        # This returns a binary mask of kept edges
        sparse_mask = self.sparsity_constraint(adj_probs, sparsity_level)
        
        # Apply hard constraint (acyclicity) on the sparse graph
        # We pass the sparse binary adj, but use original probs to decide which to cut in cycles
        # Actually, we should probably pass the masked probs to no_cycles?
        # The prompt says: no_cycles(sparse_adj, adj_probs)
        # sparse_adj is binary 0/1 from sparsity_constraint.
        
        final_dag = self.no_cycles(sparse_mask, adj_probs)
        
        return final_dag
