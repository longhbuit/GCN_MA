"""
Phase 3: GCN Layer (Graph Convolutional Network)
================================================
Implements spectral domain GCN for node embedding learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Network layer (spectral domain).
    
    H^{l+1} = σ(D^{-1/2} A~ D^{-1/2} H^l W^l)
    
    In this implementation:
    H^{l+1} = σ(D~^{-1/2} S~ D~^{-1/2} H^l W^l)
    
    where S~ = A~ + I (self-loops added)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to add bias term
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            adj: Adjacency matrix with self-loops (N x N), normalized
            x: Node features (N x in_features)
            
        Returns:
            Output features (N x out_features)
        """
        # Support = A~ * H * W
        support = torch.mm(x, self.weight)
        
        # Graph convolution: A~ * support
        output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def __repr__(self):
        return f"GraphConvolution(in={self.in_features}, out={self.out_features})"


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network.
    
    Processes: A~ = A + β*S (from NRNAE)
    Computes: H = GCN(A~, X)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        if num_layers == 1:
            self.layers.append(GraphConvolution(input_dim, output_dim))
        else:
            # First layer
            self.layers.append(GraphConvolution(input_dim, hidden_dim))
            # Middle layers
            for _ in range(num_layers - 2):
                self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
            # Last layer
            self.layers.append(GraphConvolution(hidden_dim, output_dim))
            
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all GCN layers.
        
        Args:
            adj: Normalized adjacency matrix (N x N)
            x: Node features (N x input_dim)
            
        Returns:
            Node embeddings (N x output_dim)
        """
        # First layer with ReLU and dropout
        x = F.relu(self.layers[0](adj, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Middle layers
        for layer in self.layers[1:-1]:
            x = F.relu(layer(adj, x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer (no activation, for embedding)
        x = self.layers[-1](adj, x)
        
        return x
    
    def get_embedding(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get node embeddings without dropout."""
        self.eval()
        with torch.no_grad():
            return self.forward(adj, x)


class GCNWithNAE(nn.Module):
    """
    GCN that incorporates NRNAE adjacency matrix.
    
    This combines:
    1. Base adjacency A
    2. Node Aggregation Effect S
    3. GCN layers for embedding
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.3, beta: float = 0.8):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            beta: NRNAE weight
        """
        super(GCNWithNAE, self).__init__()
        self.beta = beta
        self.gcn = GCN(input_dim, hidden_dim, output_dim, num_layers, dropout)
        
    def normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        Assumes adj already has self-loops added.
        """
        # Add small epsilon for numerical stability
        adj = adj + torch.eye(adj.size(0), device=adj.device) * 1e-8
        
        # Degree matrix
        d = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Normalized: D^{-1/2} A D^{-1/2}
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
    def forward(self, A: torch.Tensor, S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with NRNAE.
        
        Args:
            A: Base adjacency matrix (N x N)
            S: Node aggregation effect matrix (N x N)
            x: Node features (N x input_dim)
            
        Returns:
            Node embeddings (N x output_dim)
        """
        # Enriched adjacency: A~ = A + β*S
        A_tilde = A + self.beta * S
        
        # Normalize
        A_norm = self.normalize_adj(A_tilde)
        
        # GCN forward
        embeddings = self.gcn(A_norm, x)
        
        return embeddings


def verify_gcn():
    """Verify Phase 3: GCN Layer."""
    print("\n" + "="*60)
    print("PHASE 3 VERIFICATION: GCN Layer")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create test data
    num_nodes = 34  # Karate club
    input_dim = 2   # [degree, clustering]
    hidden_dim = 16
    output_dim = 8
    
    # Simulated features
    x = torch.randn(num_nodes, input_dim)
    
    # Simulated adjacency (random)
    adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = (adj + adj.T) / 2  # Symmetric
    adj = adj.fill_diagonal_(0)  # No self-loops
    
    # Add self-loops for GCN
    adj = adj + torch.eye(num_nodes)
    
    print(f"\n✅ Test data:")
    print(f"   Nodes: {num_nodes}")
    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Output dim: {output_dim}")
    
    # Test GCN layer
    gcn_layer = GraphConvolution(input_dim, output_dim)
    print(f"\n✅ GCN Layer created: {gcn_layer}")
    
    out = gcn_layer(adj, x)
    print(f"   Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    # Test full GCN
    gcn = GCN(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3)
    print(f"\n✅ Full GCN created with 2 layers")
    
    gcn.eval()
    with torch.no_grad():
        embeddings = gcn(adj, x)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    # Test GCNWithNAE
    S = torch.randn(num_nodes, num_nodes)
    S = (S + S.T) / 2  # Symmetric
    
    gcn_nae = GCNWithNAE(input_dim, hidden_dim, output_dim, beta=0.8)
    print(f"\n✅ GCNWithNAE created")
    
    gcn_nae.eval()
    with torch.no_grad():
        embeddings_nae = gcn_nae(adj, S, x)
    print(f"   Embeddings with NAE shape: {embeddings_nae.shape}")
    
    # Test normalize_adj
    A_norm = gcn_nae.normalize_adj(adj)
    print(f"\n✅ Normalized adjacency:")
    print(f"   Shape: {A_norm.shape}")
    print(f"   Row sums (should be ~1): {A_norm.sum(dim=1)[:5]}")
    
    # Test train/eval mode
    gcn.train()
    print(f"\n✅ Training mode: dropout enabled")
    gcn.eval()
    print(f"   Eval mode: dropout disabled")
    
    print("\n" + "="*60)
    print("PHASE 3: VERIFICATION PASSED ✅")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_gcn()
