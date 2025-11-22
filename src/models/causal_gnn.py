import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Fallback GCN implementation if torch_geometric is missing
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (num_nodes, in_features)
        # adj: (num_nodes, num_nodes) - normalized adjacency
        out = self.linear(x)
        out = torch.matmul(adj, out)
        return out

class CausalGNN(nn.Module):
    """Graph Neural Network for causal graph refinement"""
    
    def __init__(self, num_vars, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.use_geometric = False
        try:
            import torch_geometric.nn as gnn
            self.use_geometric = True
            self.gconv1 = gnn.GCNConv(hidden_dim, hidden_dim)
            self.gconv2 = gnn.GCNConv(hidden_dim, hidden_dim)
        except ImportError:
            # Use fallback
            self.gconv1 = GCNLayer(hidden_dim, hidden_dim)
            self.gconv2 = GCNLayer(hidden_dim, hidden_dim)

        # Encoder: Maps time-series features (node identity) to initial embeddings
        self.encoder = nn.Sequential(
            nn.Linear(num_vars, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge predictor: Classifies the probability of an edge (src -> dst)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Takes concatenated [src_embed, dst_embed]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs a probability (0 to 1)
        )
        
    def forward(self, x, edge_index):
        # x is initial node features
        node_embed = self.encoder(x)
        
        if self.use_geometric:
            # Graph convolution using torch_geometric
            node_embed = self.gconv1(node_embed, edge_index)
            node_embed = F.relu(node_embed)
            node_embed = self.gconv2(node_embed, edge_index)
        else:
            # Manual GCN
            # Convert edge_index to adjacency matrix if needed
            # edge_index is (2, num_edges)
            num_nodes = x.shape[0]
            adj = torch.eye(num_nodes).to(x.device) # Self-loops
            
            if edge_index.shape[1] > 0:
                src, dst = edge_index
                adj[src, dst] = 1.0
                
            # Normalize adjacency: D^-0.5 A D^-0.5
            deg = adj.sum(dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
            
            node_embed = self.gconv1(node_embed, norm_adj)
            node_embed = F.relu(node_embed)
            node_embed = self.gconv2(node_embed, norm_adj)
        
        # Predict edges based on refined embeddings
        src, dst = edge_index
        edge_feat = torch.cat([node_embed[src], node_embed[dst]], dim=1)
        edge_probs = self.edge_predictor(edge_feat)
        
        return edge_probs, node_embed
    
def hybrid_loss(edge_probs, true_edges, lambda_causal=1.0, lambda_sparse=0.1):
    """
    Hybrid Loss = Causal Loss (Neural) + Sparsity Penalty (Symbolic)
    """
    # 1. Causal Loss: Binary Cross-Entropy (BCE) measures fidelity to the Neural Granger result
    # true_edges should be the "ground truth" or target labels for these edges.
    # In the context of refinement, this might be the output from Neural Granger?
    
    causal_loss = F.binary_cross_entropy(edge_probs, true_edges.float())
    
    # 2. Sparsity Penalty: L1 norm encourages the model to minimize total edge weight
    sparsity_loss = torch.norm(edge_probs, p=1)
    
    total_loss = (lambda_causal * causal_loss + 
                  lambda_sparse * sparsity_loss)
    
    return total_loss
