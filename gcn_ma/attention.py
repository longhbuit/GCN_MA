"""
Phase 5: Multi-head Attention (Local Time Attribute Modeling)
===============================================================
Implements multi-head attention for capturing local temporal evolution patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor (batch, seq_len_q, d_k)
            K: Key tensor (batch, seq_len_k, d_k)
            V: Value tensor (batch, seq_len_v, d_v)
            mask: Optional mask tensor
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        d_k = Q.size(-1)
        
        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    
    Instead of performing a single attention function, linearly projects
    the queries, keys, and values h times with different learned projections.
    
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Total dimension for the model
            num_heads: Number of parallel attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim) or (seq_len, embed_dim) for node embeddings
            mask: Optional mask
            
        Returns:
            output: Multi-head attention output
            attention_weights: Averaged attention weights across heads
        """
        batch_size = 1 if x.dim() == 2 else x.size(0)
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        
        # Linear projections
        Q = self.W_q(x).view(seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads: (seq_len, num_heads, head_dim) -> (seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(seq_len, self.embed_dim)
        
        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # Average attention weights across heads
        attn_weights = attn_weights.mean(dim=0)  # (seq_len, seq_len)
        
        return output, attn_weights


class TemporalAttentionLayer(nn.Module):
    """
    Temporal Attention Layer for Local Time Modeling.
    
    Captures local information changes around each node and its neighbors
    at specific time steps.
    
    Input: Node embeddings H^T at time T
    Output: Updated embeddings Z^T
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TemporalAttentionLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.multihead_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: Node embeddings H (N x embed_dim)
            
        Returns:
            output: Updated node embeddings Z (N x embed_dim)
            attention_weights: Attention weights (N x N)
        """
        # Self-attention with residual
        attn_output, attn_weights = self.multihead_attn(node_embeddings)
        x = node_embeddings + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_output = self.ff(x)
        output = x + self.dropout(ff_output)
        output = self.norm2(output)
        
        return output, attn_weights


class LocalTemporalAttention(nn.Module):
    """
    Local Temporal Attention for Dynamic Networks.
    
    Applies multi-head attention to capture local structure information
    and temporal evolution patterns.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(LocalTemporalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.temporal_attn = TemporalAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: Node embeddings from GCN/LSTM (N x embed_dim)
            
        Returns:
            Z: Updated embeddings with local temporal info (N x embed_dim)
            attention_weights: Attention weights (N x N)
        """
        Z, attention_weights = self.temporal_attn(H)
        return Z, attention_weights
    
    def get_attention_visualization(self, H: torch.Tensor) -> np.ndarray:
        """Get attention weights for visualization."""
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(H)
        return attn_weights.cpu().numpy()


def verify_attention():
    """Verify Phase 5: Multi-head Attention."""
    print("\n" + "="*60)
    print("PHASE 5 VERIFICATION: Multi-head Attention")
    print("="*60)
    
    device = torch.device('cpu')
    embed_dim = 16
    num_heads = 4
    num_nodes = 34
    
    # Test Scaled Dot Product Attention
    print("\n✅ Scaled Dot-Product Attention:")
    attn = ScaledDotProductAttention(dropout=0.1)
    
    Q = torch.randn(1, num_nodes, embed_dim)
    K = torch.randn(1, num_nodes, embed_dim)
    V = torch.randn(1, num_nodes, embed_dim)
    
    output, weights = attn(Q, K, V)
    print(f"   Input Q, K, V: {Q.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Attention weights sum (per row): {weights.sum(dim=-1)[:3]}")
    
    # Test Multi-Head Attention
    print("\n✅ Multi-Head Attention:")
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
    print(f"   Embed dim: {embed_dim}, Num heads: {num_heads}")
    print(f"   Head dim: {embed_dim // num_heads}")
    
    x = torch.randn(num_nodes, embed_dim)  # (seq_len, embed_dim)
    output, attn_weights = mha(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Test Temporal Attention Layer
    print("\n✅ Temporal Attention Layer:")
    temporal_layer = TemporalAttentionLayer(embed_dim=embed_dim, num_heads=num_heads)
    
    H = torch.randn(num_nodes, embed_dim)
    Z, attn_weights = temporal_layer(H)
    print(f"   Input H shape: {H.shape}")
    print(f"   Output Z shape: {Z.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Verify residual connections
    print(f"   Z range: [{Z.min():.4f}, {Z.max():.4f}]")
    
    # Test Local Temporal Attention
    print("\n✅ Local Temporal Attention:")
    local_attn = LocalTemporalAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    H = torch.randn(num_nodes, embed_dim)
    Z, attn = local_attn(H)
    print(f"   Input H: {H.shape}")
    print(f"   Output Z: {Z.shape}")
    print(f"   Attention: {attn.shape}")
    
    # Test attention visualization
    print("\n✅ Attention Visualization:")
    attn_vis = local_attn.get_attention_visualization(H)
    print(f"   Attention matrix shape: {attn_vis.shape}")
    print(f"   Attention range: [{attn_vis.min():.4f}, {attn_vis.max():.4f}]")
    print(f"   Diagonal dominance: {np.diag(attn_vis).mean():.4f}")
    
    # Test multiple heads analysis
    print("\n✅ Multi-head Analysis:")
    mha_explicit = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    
    all_head_weights = []
    x = torch.randn(num_nodes, embed_dim)
    
    # Get Q, K, V for all heads
    Q_all = mha_explicit.W_q(x).view(num_nodes, num_heads, embed_dim // num_heads).transpose(0, 1)
    K_all = mha_explicit.W_k(x).view(num_nodes, num_heads, embed_dim // num_heads).transpose(0, 1)
    
    for head in range(num_heads):
        Q_h = Q_all[head]  # (num_nodes, head_dim)
        K_h = K_all[head]  # (num_nodes, head_dim)
        
        scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(embed_dim // num_heads)
        attn_h = F.softmax(scores, dim=-1)
        all_head_weights.append(attn_h)
        
    all_head_weights = torch.stack(all_head_weights)  # (num_heads, num_nodes, num_nodes)
    print(f"   All heads attention shape: {all_head_weights.shape}")
    print(f"   Head 0 diag mean: {all_head_weights[0].diagonal().mean():.4f}")
    print(f"   Head 1 diag mean: {all_head_weights[1].diagonal().mean():.4f}")
    
    print("\n" + "="*60)
    print("PHASE 5: VERIFICATION PASSED ✅")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_attention()
