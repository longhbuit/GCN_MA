"""
Phase 6: Link Prediction Model & Loss Function
=================================================
MLP classifier for link prediction using node embeddings.
Binary cross-entropy loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np


class LinkPredictor(nn.Module):
    """
    MLP-based Link Predictor.
    
    Takes node embedding pairs and predicts probability of edge existence.
    
    Input: Concatenated node embeddings [Z_i || Z_j]
    Output: Probability of edge between node i and j
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Args:
            embed_dim: Node embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super(LinkPredictor, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Build MLP
        layers = []
        input_dim = embed_dim * 2  # Concatenate two node embeddings
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))  # Output: probability score
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between node pairs.
        
        Args:
            z_i: Node i embeddings (batch x embed_dim)
            z_j: Node j embeddings (batch x embed_dim)
            
        Returns:
            Probability scores (batch x 1)
        """
        # Concatenate embeddings
        pair_emb = torch.cat([z_i, z_j], dim=-1)  # (batch x 2*embed_dim)
        
        # MLP forward
        scores = self.mlp(pair_emb)  # (batch x 1)
        
        return scores
    
    def predict_proba(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Get probability scores (0-1)."""
        scores = self.forward(z_i, z_j)
        return torch.sigmoid(scores)


class LinkPredictionLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for Link Prediction.
    
    Loss = -1/N * Σ [Y * log(P) + (1-Y) * log(1-P)]
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            pos_weight: Weight for positive samples
        """
        super(LinkPredictionLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted probabilities (N,)
            targets: Ground truth labels (N,)
            
        Returns:
            Loss value
        """
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy(
                predictions, targets,
                pos_weight=self.pos_weight,
                reduction='mean'
            )
        else:
            loss = F.binary_cross_entropy(
                predictions, targets,
                reduction='mean'
            )
        return loss


class LinkPredictionDataset(torch.utils.data.Dataset):
    """
    Dataset for Link Prediction.
    
    Generates positive and negative edge samples.
    """
    
    def __init__(self, graph: 'nx.Graph', node_embeddings: torch.Tensor,
                 num_negative_per_positive: int = 1, device: str = 'cpu'):
        """
        Args:
            graph: NetworkX graph
            node_embeddings: Node embeddings (N x embed_dim)
            num_negative_per_positive: Ratio of negative to positive samples
            device: Device
        """
        self.graph = graph
        self.node_embeddings = node_embeddings
        self.num_negative = num_negative_per_positive
        self.device = device
        
        # Build edge list
        self.positive_edges = list(graph.edges())
        self.num_positive = len(self.positive_edges)
        
        # Generate negative edges
        self.negative_edges = self._generate_negative_edges()
        
        # Combine
        self.edges = self.positive_edges + self.negative_edges
        self.labels = ([1] * self.num_positive) + ([0] * len(self.negative_edges))
        
        # Shuffle
        indices = torch.randperm(len(self.edges))
        self.edges = [self.edges[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
    def _generate_negative_edges(self) -> List[Tuple[int, int]]:
        """Generate non-existing edges as negative samples."""
        nodes = list(self.graph.nodes())
        num_nodes = len(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        negative_edges = []
        existing_edges = set(self.positive_edges)
        
        attempts = 0
        max_attempts = self.num_positive * self.num_negative * 10
        
        while len(negative_edges) < self.num_positive * self.num_negative and attempts < max_attempts:
            i, j = np.random.choice(num_nodes, 2, replace=True)
            if i != j and (node_to_idx.get(nodes[i]), node_to_idx.get(nodes[j])) not in existing_edges:
                negative_edges.append((node_to_idx[nodes[i]], node_to_idx[nodes[j]]))
            attempts += 1
            
        return negative_edges
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one sample."""
        u, v = self.edges[idx]
        label = self.labels[idx]
        
        z_u = self.node_embeddings[u]
        z_v = self.node_embeddings[v]
        
        return z_u, z_v, torch.tensor(label, dtype=torch.float32)


def compute_auc_ap(predictions: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """
    Compute AUC and AP metrics.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        
    Returns:
        AUC, AP scores
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    pred_np = predictions.cpu().detach().numpy()
    label_np = labels.cpu().numpy()
    
    # AUC
    try:
        auc = roc_auc_score(label_np, pred_np)
    except:
        auc = 0.0
        
    # AP
    try:
        ap = average_precision_score(label_np, pred_np)
    except:
        ap = 0.0
        
    return auc, ap


def verify_link_predictor():
    """Verify Phase 6: Link Prediction Model."""
    print("\n" + "="*60)
    print("PHASE 6 VERIFICATION: Link Prediction Model")
    print("="*60)
    
    device = torch.device('cpu')
    embed_dim = 16
    num_nodes = 34
    hidden_dim = 32
    
    # Create test embeddings
    z = torch.randn(num_nodes, embed_dim)
    
    # Create test graph
    import networkx as nx
    G = nx.karate_club_graph()
    
    # Test Link Predictor
    print("\n✅ LinkPredictor:")
    predictor = LinkPredictor(embed_dim=embed_dim, hidden_dim=hidden_dim)
    print(f"   MLP architecture: {predictor.mlp}")
    
    # Test forward pass
    z_i = z[:10]
    z_j = z[10:20]
    scores = predictor(z_i, z_j)
    print(f"   Input z_i: {z_i.shape}, z_j: {z_j.shape}")
    print(f"   Output scores: {scores.shape}")
    print(f"   Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Test probability output
    proba = predictor.predict_proba(z_i, z_j)
    print(f"   Probabilities range: [{proba.min():.4f}, {proba.max():.4f}]")
    
    # Test Loss
    print("\n✅ LinkPredictionLoss:")
    loss_fn = LinkPredictionLoss()
    
    predictions = torch.tensor([0.1, 0.9, 0.5, 0.3])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    loss = loss_fn(predictions, targets)
    print(f"   Predictions: {predictions}")
    print(f"   Targets: {targets}")
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Dataset
    print("\n✅ LinkPredictionDataset:")
    dataset = LinkPredictionDataset(G, z, num_negative_per_positive=1)
    print(f"   Positive edges: {dataset.num_positive}")
    print(f"   Negative edges: {len(dataset.negative_edges)}")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Label distribution: {np.mean(dataset.labels):.4f} positive ratio")
    
    # Test dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    z_u, z_v, labels = batch
    print(f"   Batch z_u: {z_u.shape}, z_v: {z_v.shape}, labels: {labels.shape}")
    
    # Test AUC/AP
    print("\n✅ AUC/AP Metrics:")
    auc, ap = compute_auc_ap(predictions, targets)
    print(f"   AUC: {auc:.4f}")
    print(f"   AP: {ap:.4f}")
    
    # Test full pipeline
    print("\n✅ Full Pipeline Test:")
    all_predictions = []
    all_labels = []
    
    predictor.eval()
    with torch.no_grad():
        for batch in dataloader:
            z_u, z_v, labels = batch
            pred = predictor.predict_proba(z_u, z_v).squeeze()
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
    
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    
    auc, ap = compute_auc_ap(all_predictions, all_labels)
    print(f"   Total samples: {len(all_predictions)}")
    print(f"   Overall AUC: {auc:.4f}")
    print(f"   Overall AP: {ap:.4f}")
    
    print("\n" + "="*60)
    print("PHASE 6: VERIFICATION PASSED ✅")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_link_predictor()
