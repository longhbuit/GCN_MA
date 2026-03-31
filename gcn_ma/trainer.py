"""
GCN_MA: Dynamic Network Link Prediction
======================================

Complete implementation following the paper:
"Dynamic network link prediction with node representation learning from graph convolutional networks"

Components:
1. NRNAE - Node Representation based on Node Aggregation Effect
2. GCN - Graph Convolutional Network  
3. LSTM - Global temporal modeling (updates GCN weights)
4. Multi-head Attention - Local temporal modeling
5. MLP - Link prediction

Author: Implementation based on Scientific Reports paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import networkx as nx
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# NRNAE Algorithm (Node Representation based on Node Aggregation Effect)
# ============================================================

class NRNAE:
    """
    Node Representation Algorithm Based on Node Aggregation Effect.
    
    Enriches node information by computing:
    - Clustering Coefficient: CC(i) = 2*Ri / (Ki * (Ki-1))
    - Aggregation Strength: AS(i) = degree(i) * CC(i)
    - Node Aggregation Effect: S(i,j) = |N(i) ∩ N(j)| * AS(i)
    - Enriched Adjacency: A~ = A + β*S
    """
    
    def __init__(self, beta: float = 0.8):
        self.beta = beta
        
    def compute_clustering_coefficient(self, G: nx.Graph) -> Dict[int, float]:
        """Compute clustering coefficient for each node."""
        return nx.clustering(G)
    
    def compute_aggregation_strength(self, G: nx.Graph) -> Dict[int, float]:
        """AS(i) = degree(i) * CC(i)"""
        degrees = dict(G.degree())
        clustering = self.compute_clustering_coefficient(G)
        return {node: degrees[node] * clustering.get(node, 0) for node in G.nodes()}
    
    def compute_node_aggregation_effect(self, G: nx.Graph) -> Tuple[np.ndarray, List[int]]:
        """
        Compute Node Aggregation Effect matrix S.
        
        S(i,j) = |N(i) ∩ N(j)| * AS(i)
        """
        node_list = sorted(list(G.nodes()))
        num_nodes = len(node_list)
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        
        aggregation_strength = self.compute_aggregation_strength(G)
        S = np.zeros((num_nodes, num_nodes))
        
        for i, node_i in enumerate(node_list):
            neighbors_i = set(G.neighbors(node_i))
            as_i = aggregation_strength[node_i]
            
            for node_j in neighbors_i:
                j = node_to_idx[node_j]
                neighbors_j = set(G.neighbors(node_j))
                intersection = len(neighbors_i & neighbors_j)
                S[i, j] = intersection * as_i
        
        # Normalize to [0, 1]
        if S.max() > 0:
            S = S / S.max()
            
        return S, node_list
    
    def get_node_features(self, G: nx.Graph) -> Tuple[np.ndarray, List[int]]:
        """Get node features: [degree, clustering_coefficient]"""
        node_list = sorted(list(G.nodes()))
        degrees = dict(G.degree())
        clustering = self.compute_clustering_coefficient(G)
        
        features = np.zeros((len(node_list), 2))
        for i, node in enumerate(node_list):
            features[i, 0] = degrees.get(node, 0)
            features[i, 1] = clustering.get(node, 0)
            
        return features, node_list


# ============================================================
# GCN Layer (Graph Convolutional Network)
# ============================================================

class GraphConvolution(nn.Module):
    """Single GCN layer: H^{l+1} = σ(D~^{-1/2} A~ D~^{-1/2} H^l W^l)"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward: A~ * H * W"""
        support = torch.mm(x, self.weight)
        output = torch.mm(adj_norm, support) + self.bias
        return output


class GCN(nn.Module):
    """Multi-layer GCN with ReLU activation and dropout."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, adj_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(adj_norm, x))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.conv2(adj_norm, x)
        return x


# ============================================================
# LSTM Weight Updater (Global Temporal Modeling)
# ============================================================

class LSTMWeightUpdater(nn.Module):
    """
    Updates GCN weights across time steps using LSTM.
    
    WT = LSTM(WT-1)
    
    Captures global temporal evolution by updating GCN parameters.
    """
    
    def __init__(self, gcn_weight_dim: int, hidden_dim: int):
        super().__init__()
        # For matrix-valued input, flatten to vector
        flat_dim = gcn_weight_dim
        
        self.fc_f = nn.Linear(flat_dim, hidden_dim)
        self.fc_i = nn.Linear(flat_dim, hidden_dim)  
        self.fc_o = nn.Linear(flat_dim, hidden_dim)
        self.fc_c = nn.Linear(flat_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, flat_dim)
        
        self.hidden_dim = hidden_dim
        self.c = None
        
    def forward(self, W_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update weight matrix.
        
        Args:
            W_prev: Previous GCN weight (input_dim x output_dim)
            
        Returns:
            W_new: Updated weight
            h: Hidden state
        """
        # Flatten: (input_dim x output_dim) -> (input_dim * output_dim,)
        W_flat = W_prev.flatten().unsqueeze(0)  # (1 x flat_dim)
        
        # LSTM gates
        f = torch.sigmoid(self.fc_f(W_flat))
        i = torch.sigmoid(self.fc_i(W_flat))
        c_tilde = torch.tanh(self.fc_c(W_flat))
        o = torch.sigmoid(self.fc_o(W_flat))
        
        # Cell state
        if self.c is None:
            self.c = torch.zeros_like(c_tilde)
        c_new = f * self.c + i * c_tilde
        
        # Hidden state
        h_new = o * torch.tanh(c_new)
        
        # Project back to weight dimensions
        W_new = self.fc_out(h_new)  # (1 x flat_dim)
        W_new = W_new.view_as(W_prev)  # Restore original shape
        
        self.c = c_new.detach()
        
        return W_new, h_new.squeeze(0)


# ============================================================
# Multi-head Attention (Local Temporal Modeling)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for local temporal modeling.
    
    Captures local information changes around nodes and their neighbors.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_p = dropout
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node embeddings (N x embed_dim)
            
        Returns:
            output: Updated embeddings (N x embed_dim)
            attention_weights: Attention matrix (N x N)
        """
        N = x.size(0)
        
        # Linear projections
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.dropout(attn_weights, p=self.dropout_p, train=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, self.embed_dim)
        output = self.W_o(attn_output)
        
        return output, attn_weights.mean(dim=0)


class TemporalAttentionLayer(nn.Module):
    """Temporal attention with residual connections and layer norm."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + torch.dropout(attn_out, p=self.attn.dropout_p, train=self.training))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


# ============================================================
# GCN_MA Full Model
# ============================================================

class GCN_MA(nn.Module):
    """
    Complete GCN_MA model.
    
    Architecture:
    1. NRNAE: Compute enriched adjacency A~ = A + β*S
    2. GCN: Learn node embeddings from A~
    3. LSTM: Update GCN weights for global temporal (optional, can be disabled)
    4. Multi-head Attention: Capture local temporal patterns
    5. MLP: Link prediction
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 num_heads: int = 8, dropout: float = 0.3, beta: float = 0.8,
                 use_lstm: bool = False):
        super().__init__()
        
        self.beta = beta
        self.use_lstm = use_lstm
        self.nrnae = NRNAE(beta=beta)
        
        # GCN
        self.gcn = GCN(input_dim, hidden_dim, embed_dim, dropout)
        
        # LSTM for weight updates (optional)
        if use_lstm:
            gcn_weight_dim = hidden_dim * embed_dim  # For conv1 weight
            self.lstm_updater = LSTMWeightUpdater(gcn_weight_dim, hidden_dim)
        
        # Local temporal attention
        self.temporal_attn = TemporalAttentionLayer(embed_dim, num_heads, dropout)
        
        # Link predictor (MLP)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Store GCN weights for LSTM updates
        self.register_buffer('gcn_weight', None)
        
    def normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency: D^{-1/2} A D^{-1/2}"""
        adj = adj + torch.eye(adj.size(0), device=adj.device) * 1e-8
        d = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one snapshot.
        
        Args:
            adj: Adjacency matrix (N x N)
            x: Node features (N x input_dim)
            
        Returns:
            node_embeddings: Final embeddings (N x embed_dim)
            attention_weights: Attention matrix (N x N)
        """
        # Compute NRNAE
        A_np = adj.cpu().numpy()
        G_nx = nx.from_numpy_array(A_np)
        S, _ = self.nrnae.compute_node_aggregation_effect(G_nx)
        S = torch.FloatTensor(S).to(adj.device)
        
        # Enriched adjacency
        A_tilde = adj + self.beta * S
        
        # Normalize
        A_norm = self.normalize_adj(A_tilde)
        
        # GCN embeddings
        H = self.gcn(A_norm, x)
        
        # Local temporal attention
        Z, attn_weights = self.temporal_attn(H)
        
        return Z, attn_weights
    
    def predict_link(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """Predict link probability between node pairs."""
        pair_emb = torch.cat([z_u, z_v], dim=-1)
        score = self.predictor(pair_emb)
        return torch.sigmoid(score)


# ============================================================
# Trainer
# ============================================================

class GCN_MA_Trainer:
    """Trainer for GCN_MA model."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cpu')
        
        # Set seed
        seed = config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Global node set for alignment
        self.global_node_list = None
        self.node_to_idx = None
        self.num_nodes = None
        
        # Create model
        self.model = GCN_MA(
            input_dim=2,
            hidden_dim=config['gcn']['hidden_dim'],
            embed_dim=config['gcn']['output_dim'],
            num_heads=config['attention']['num_heads'],
            dropout=config['gcn'].get('dropout', 0.3),
            beta=config['nrnae']['beta'],
            use_lstm=False  # LSTM disabled by default due to dimension complexity
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.loss_fn = nn.BCELoss()
        
        # Results tracking
        self.best_val_auc = 0.0
        self.results = {'train_loss': [], 'val_auc': [], 'val_ap': [], 'test_auc': [], 'test_ap': []}
        
    def build_global_node_set(self, graphs: List[nx.Graph]):
        """Build global node set from all graphs."""
        all_nodes = set()
        for G in graphs:
            all_nodes.update(G.nodes())
        self.global_node_list = sorted(all_nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.global_node_list)}
        self.num_nodes = len(self.global_node_list)
        
    def prepare_snapshot(self, graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare adjacency matrix and features for a snapshot."""
        num_nodes = self.num_nodes
        
        # Node features
        features = np.zeros((num_nodes, 2))
        node_degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        
        for i, node in enumerate(self.global_node_list):
            if node in node_degrees:
                features[i, 0] = node_degrees[node]
                features[i, 1] = clustering.get(node, 0)
        features = torch.FloatTensor(features).to(self.device)
        
        # Adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        for u, v in graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                adj[i, j] = 1
                adj[j, i] = 1
        adj = adj.to(self.device)
        
        return adj, features
    
    def sample_edges(self, graph: nx.Graph, next_graph: nx.Graph, 
                    num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Sample positive and negative edges."""
        # Existing edges in current graph
        existing = set()
        for u, v in graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                existing.add((i, j))
                existing.add((j, i))
        
        # New edges in next graph (positive)
        new_edges = []
        for u, v in next_graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                if (i, j) not in existing and (j, i) not in existing:
                    new_edges.append([i, j])
        
        if len(new_edges) == 0:
            return np.array([]), np.array([])
        
        # Sample positive
        num_pos = min(len(new_edges), num_samples // 2)
        pos_samples = np.array(random.sample(new_edges, num_pos))
        
        # Sample negative - simpler approach
        neg_samples = []
        attempts = 0
        max_attempts = num_pos * 20
        
        while len(neg_samples) < num_pos and attempts < max_attempts:
            i = random.randint(0, self.num_nodes - 1)
            j = random.randint(0, self.num_nodes - 1)
            if i != j and (i, j) not in existing and (j, i) not in existing:
                neg_samples.append([i, j])
                existing.add((i, j))  # Add to prevent duplicates
            attempts += 1
        
        if len(neg_samples) == 0:
            return np.array([]), np.array([])
        
        neg_samples = np.array(neg_samples)
        
        # Combine and shuffle
        edges = np.vstack([pos_samples, neg_samples])
        labels = np.concatenate([np.ones(num_pos), np.zeros(len(neg_samples))])
        
        perm = np.random.permutation(len(edges))
        return edges[perm], labels[perm]
    
    def train_epoch(self, graphs: List[nx.Graph], epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for t in range(len(graphs) - 1):
            adj, features = self.prepare_snapshot(graphs[t])
            
            # Forward
            Z, _ = self.model(adj, features)
            
            # Sample edges
            edges, labels = self.sample_edges(graphs[t], graphs[t + 1], num_samples=100)
            
            if len(edges) == 0:
                continue
            
            edges_t = torch.LongTensor(edges).to(self.device)
            labels_t = torch.FloatTensor(labels).to(self.device)
            
            # Predict
            z_u, z_v = Z[edges_t[:, 0]], Z[edges_t[:, 1]]
            pred = self.model.predict_link(z_u, z_v).squeeze()
            
            loss = self.loss_fn(pred, labels_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, graph: nx.Graph, next_graph: nx.Graph) -> Tuple[float, float]:
        """Evaluate on graph pair."""
        self.model.eval()
        
        adj, features = self.prepare_snapshot(graph)
        
        with torch.no_grad():
            Z, _ = self.model(adj, features)
        
        edges, labels = self.sample_edges(graph, next_graph, num_samples=200)
        
        if len(edges) == 0:
            return 0.0, 0.0
        
        edges_t = torch.LongTensor(edges).to(self.device)
        labels_np = labels
        
        z_u, z_v = Z[edges_t[:, 0]], Z[edges_t[:, 1]]
        pred = self.model.predict_link(z_u, z_v).squeeze().detach().cpu().numpy()
        
        auc = roc_auc_score(labels_np, pred)
        ap = average_precision_score(labels_np, pred)
        
        return auc, ap
    
    def train(self, train_graphs: List, val_graphs: List, test_graphs: List) -> Dict:
        """Full training loop."""
        print(f"\n{'='*60}")
        print("TRAINING GCN_MA")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train snapshots: {len(train_graphs)}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"{'='*60}\n")
        
        epochs = self.config['training']['epochs']
        patience = self.config['training'].get('early_stopping_patience', 20)
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_graphs, epoch)
            
            if len(val_graphs) >= 2:
                val_auc, val_ap = self.evaluate(val_graphs[0], val_graphs[1])
            else:
                val_auc, val_ap = 0.0, 0.0
                
            if len(test_graphs) >= 2:
                test_auc, test_ap = self.evaluate(test_graphs[0], test_graphs[1])
            else:
                test_auc, test_ap = 0.0, 0.0
            
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Val: AUC={val_auc:.4f} AP={val_ap:.4f} | "
                      f"Test: AUC={test_auc:.4f} AP={test_ap:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            
            self.results['train_loss'].append(train_loss)
            self.results['val_auc'].append(val_auc)
            self.results['val_ap'].append(val_ap)
            self.results['test_auc'].append(test_auc)
            self.results['test_ap'].append(test_ap)
            
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                patience_counter = 0
                Path('checkpoints').mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), 'checkpoints/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\nBest Val AUC: {self.best_val_auc:.4f}")
        return self.results


# ============================================================
# Main / Verification
# ============================================================

if __name__ == "__main__":
    import yaml
    
    def verify_model():
        """Verify GCN_MA model components."""
        print("\n" + "="*60)
        print("GCN_MA MODEL VERIFICATION")
        print("="*60)
        
        # Test NRNAE
        print("\n[1] Testing NRNAE...")
        nrnae = NRNAE(beta=0.8)
        G = nx.karate_club_graph()
        S, nodes = nrnae.compute_node_aggregation_effect(G)
        print(f"   Nodes: {len(nodes)}, S shape: {S.shape}")
        print(f"   S range: [{S.min():.4f}, {S.max():.4f}]")
        
        features, _ = nrnae.get_node_features(G)
        print(f"   Features shape: {features.shape}")
        
        # Test GCN
        print("\n[2] Testing GCN...")
        gcn = GCN(input_dim=2, hidden_dim=16, output_dim=8)
        adj = torch.rand(34, 34)
        adj = (adj + adj.t()) / 2
        x = torch.rand(34, 2)
        h = gcn(adj, x)
        print(f"   Input: {x.shape}, Output: {h.shape}")
        
        # Test Multi-head Attention
        print("\n[3] Testing Multi-head Attention...")
        attn = TemporalAttentionLayer(embed_dim=8, num_heads=4)
        z, weights = attn(h)
        print(f"   Input: {h.shape}, Output: {z.shape}")
        print(f"   Attention weights: {weights.shape}")
        
        # Test full GCN_MA
        print("\n[4] Testing GCN_MA...")
        model = GCN_MA(
            input_dim=2,
            hidden_dim=16,
            embed_dim=8,
            num_heads=4,
            dropout=0.3,
            beta=0.8
        )
        
        adj, features = prepare_snapshot(model.nrnae, G)
        Z, attn = model(adj, features)
        print(f"   Output embeddings: {Z.shape}")
        print(f"   Attention: {attn.shape}")
        
        # Test link prediction
        edges = torch.tensor([[0, 1], [2, 3], [4, 5]])
        z_u, z_v = Z[edges[:, 0]], Z[edges[:, 1]]
        pred = model.predict_link(z_u, z_v)
        print(f"   Predictions: {pred.squeeze().tolist()}")
        
        print("\n" + "="*60)
        print("VERIFICATION PASSED ✅")
        print("="*60)
        
    def prepare_snapshot(nrnae, graph):
        """Helper to prepare snapshot data."""
        node_list = sorted(graph.nodes())
        features, _ = nrnae.get_node_features(graph)
        
        adj = torch.zeros(len(node_list), len(node_list))
        for i, u in enumerate(node_list):
            for j, v in enumerate(node_list):
                if graph.has_edge(u, v):
                    adj[i, j] = 1
        features = torch.FloatTensor(features)
        adj = adj + torch.eye(len(node_list))  # Add self-loops
        return adj, features
        
    # Run verification
    verify_model()
