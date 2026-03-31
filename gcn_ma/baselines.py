"""
Baseline Models for Link Prediction Comparison
==============================================

Implements:
1. CN - Common Neighbors
2. AA - Adamic-Adar
3. PA - Preferential Attachment
4. GCN Baseline (without temporal modeling)
5. DGCN (GCN + LSTM without attention)
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

from gcn_ma.gcn_layer import GCN


class CommonNeighbors:
    """
    Common Neighbors predictor.
    Score(u,v) = |N(u) ∩ N(v)|
    """
    
    def __init__(self):
        self.name = "CN"
        
    def fit_predict(self, graph: nx.Graph, test_edges: List[Tuple[int, int]], 
                    neg_edges: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute CN scores and evaluate."""
        # Build adjacency for fast lookup
        adj = nx.to_numpy_array(graph)
        
        scores = []
        labels = []
        
        # Positive edges
        for u, v in test_edges:
            cn = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
            scores.append(cn)
            labels.append(1)
            
        # Negative edges
        for u, v in neg_edges:
            cn = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
            scores.append(cn)
            labels.append(0)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        return auc, ap


class AdamicAdar:
    """
    Adamic-Adar index predictor.
    Score(u,v) = Σ 1/log(|N(w)|) for w in N(u) ∩ N(v)
    """
    
    def __init__(self):
        self.name = "AA"
        
    def fit_predict(self, graph: nx.Graph, test_edges: List[Tuple[int, int]], 
                    neg_edges: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute AA scores and evaluate."""
        scores = []
        labels = []
        
        # Positive edges
        for u, v in test_edges:
            neighbors_u = set(graph.neighbors(u))
            neighbors_v = set(graph.neighbors(v))
            common = neighbors_u & neighbors_v
            
            aa_score = 0.0
            for w in common:
                deg = graph.degree(w)
                if deg > 1:
                    aa_score += 1.0 / np.log(deg)
            scores.append(aa_score)
            labels.append(1)
            
        # Negative edges
        for u, v in neg_edges:
            neighbors_u = set(graph.neighbors(u))
            neighbors_v = set(graph.neighbors(v))
            common = neighbors_u & neighbors_v
            
            aa_score = 0.0
            for w in common:
                deg = graph.degree(w)
                if deg > 1:
                    aa_score += 1.0 / np.log(deg)
            scores.append(aa_score)
            labels.append(0)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        return auc, ap


class PreferentialAttachment:
    """
    Preferential Attachment predictor.
    Score(u,v) = degree(u) × degree(v)
    """
    
    def __init__(self):
        self.name = "PA"
        
    def fit_predict(self, graph: nx.Graph, test_edges: List[Tuple[int, int]], 
                    neg_edges: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Compute PA scores and evaluate."""
        scores = []
        labels = []
        
        # Positive edges
        for u, v in test_edges:
            pa_score = graph.degree(u) * graph.degree(v)
            scores.append(pa_score)
            labels.append(1)
            
        # Negative edges
        for u, v in neg_edges:
            pa_score = graph.degree(u) * graph.degree(v)
            scores.append(pa_score)
            labels.append(0)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Normalize
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        return auc, ap


class GCNBaseline(nn.Module):
    """
    GCN Baseline - GCN without temporal modeling.
    For comparison: tests contribution of temporal components (LSTM + Attention).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.3, beta: float = 0.8):
        super().__init__()
        self.name = "GCN"
        
        from gcn_ma.nrnae import NRNAE
        
        self.nrnae = NRNAE(beta=beta)
        
        # Simple 2-layer GCN
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        
        # Link prediction: cosine similarity
        self.cos = nn.CosineSimilarity(dim=1)
        
    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without temporal modeling."""
        # Compute NRNAE
        A_np = adj.cpu().numpy()
        G_nx = nx.from_numpy_array(A_np)
        S, _ = self.nrnae.compute_node_aggregation_effect(G_nx)
        S = torch.FloatTensor(S).to(adj.device)
        
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Normalize
        d = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        adj_norm = d_mat @ adj @ d_mat
        
        # GCN layers
        x = torch.relu(self.conv1(adj_norm @ x))
        x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.conv2(adj_norm @ x)
        
        return x
    
    def predict_link(self, z_u: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """Predict link probability using cosine similarity (transformed to [0,1])."""
        cos_sim = self.cos(z_u, z_v)
        # Transform from [-1, 1] to [0, 1]
        return (cos_sim + 1) / 2


class GCNBaselineTrainer:
    """Trainer for GCN Baseline model."""
    
    def __init__(self, config: Dict):
        from gcn_ma.nrnae import NRNAE
        
        self.config = config
        self.device = torch.device('cpu')
        
        self.model = GCNBaseline(
            input_dim=2,
            hidden_dim=config['gcn']['hidden_dim'],
            output_dim=config['gcn']['output_dim'],
            dropout=config['gcn'].get('dropout', 0.3),
            beta=config['nrnae']['beta']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.loss_fn = nn.BCELoss()
        
    def prepare_data(self, graph: nx.Graph, node_to_idx: Dict[int, int], num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare adjacency and features."""
        features = np.zeros((num_nodes, 2))
        node_degrees = dict(graph.degree())
        clustering = nx.clustering(graph)
        
        for i, node in enumerate(sorted(node_to_idx.keys())):
            if node in node_degrees:
                features[i, 0] = node_degrees[node]
                features[i, 1] = clustering.get(node, 0)
        
        adj = torch.zeros(num_nodes, num_nodes)
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                adj[node_to_idx[u], node_to_idx[v]] = 1
                adj[node_to_idx[v], node_to_idx[u]] = 1
        
        return torch.FloatTensor(adj).to(self.device), torch.FloatTensor(features).to(self.device)
    
    def sample_edges(self, graph: nx.Graph, next_graph: nx.Graph, 
                    node_to_idx: Dict[int, int], num_nodes: int,
                    num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Sample edges for training."""
        import random
        
        # Existing edges
        existing = set()
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                existing.add((node_to_idx[u], node_to_idx[v]))
                existing.add((node_to_idx[v], node_to_idx[u]))
        
        # New edges (positive)
        new_edges = []
        for u, v in next_graph.edges():
            if u in node_to_idx and v in node_to_idx:
                ui, vi = node_to_idx[u], node_to_idx[v]
                if (ui, vi) not in existing and (vi, ui) not in existing:
                    new_edges.append([ui, vi])
        
        if len(new_edges) == 0:
            return np.array([]), np.array([])
        
        # Sample positive
        num_pos = min(len(new_edges), num_samples // 2)
        pos_samples = np.array(random.sample(new_edges, num_pos))
        
        # Sample negative - use same approach as simple baselines
        neg_samples = []
        all_nodes_list = list(node_to_idx.keys())
        attempts = 0
        max_attempts = num_pos * 10
        
        while len(neg_samples) < num_pos and attempts < max_attempts:
            i = random.choice(all_nodes_list)
            j = random.choice(all_nodes_list)
            i_idx = node_to_idx[i]
            j_idx = node_to_idx[j]
            if i != j and (i_idx, j_idx) not in existing and (j_idx, i_idx) not in existing:
                neg_samples.append([i_idx, j_idx])
                existing.add((i_idx, j_idx))
            attempts += 1
        
        if len(neg_samples) == 0:
            return np.array([]), np.array([])
        
        neg_samples = np.array(neg_samples)
        
        # Combine and shuffle
        edges = np.vstack([pos_samples, neg_samples])
        labels = np.concatenate([np.ones(num_pos), np.zeros(len(neg_samples))])
        
        perm = np.random.permutation(len(edges))
        return edges[perm], labels[perm]
    
    def train_epoch(self, train_graphs: List, node_to_idx: Dict, num_nodes: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for t in range(len(train_graphs) - 1):
            adj, features = self.prepare_data(train_graphs[t], node_to_idx, num_nodes)
            
            # Forward
            Z = self.model(adj, features)
            
            # Sample edges
            edges, labels = self.sample_edges(train_graphs[t], train_graphs[t+1], 
                                              node_to_idx, num_nodes, num_samples=100)
            
            if len(edges) == 0:
                continue
            
            edges_t = torch.LongTensor(edges).to(self.device)
            labels_t = torch.FloatTensor(labels).to(self.device)
            
            # Predict using cosine similarity
            z_u, z_v = Z[edges_t[:, 0]], Z[edges_t[:, 1]]
            pred = self.model.predict_link(z_u, z_v)
            
            loss = self.loss_fn(pred, labels_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, graph: nx.Graph, next_graph: nx.Graph, 
                node_to_idx: Dict, num_nodes: int) -> Tuple[float, float]:
        """Evaluate on graph pair."""
        self.model.eval()
        
        adj, features = self.prepare_data(graph, node_to_idx, num_nodes)
        
        with torch.no_grad():
            Z = self.model(adj, features)
        
        # Sample edges
        edges, labels = self.sample_edges(graph, next_graph, node_to_idx, num_nodes, num_samples=200)
        
        if len(edges) == 0:
            return 0.0, 0.0
        
        edges_t = torch.LongTensor(edges).to(self.device)
        labels_t = torch.FloatTensor(labels)
        
        Z_u, Z_v = Z[edges_t[:, 0]], Z[edges_t[:, 1]]
        
        # Predict using cosine similarity
        pred = self.model.predict_link(Z_u, Z_v).cpu().numpy()
        
        auc = roc_auc_score(labels_t.numpy(), pred)
        ap = average_precision_score(labels_t.numpy(), pred)
        
        return auc, ap


def run_baseline_comparison(dataset_name: str, graphs: List[nx.Graph], 
                           train_graphs: List, val_graphs: List, test_graphs: List,
                           config: Dict) -> Dict:
    """
    Run all baseline models and compare.
    """
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON: {dataset_name}")
    print(f"{'='*60}\n")
    
    import random
    results = {}
    
    # Build global node mapping
    all_nodes = set()
    for G in train_graphs + val_graphs + test_graphs:
        all_nodes.update(G.nodes())
    global_nodes = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(global_nodes)}
    num_nodes = len(global_nodes)
    
    print(f"Global nodes: {num_nodes}")
    
    # Get test edges - only edges where BOTH nodes exist in test_graphs[0]
    print("\n[1/5] Common Neighbors...")
    cn = CommonNeighbors()
    test_edges = []
    neg_edges = []
    
    # Get test edges from last snapshot pair
    # Only include edges where both endpoints exist in test_graphs[0]
    test_graph_nodes = set(test_graphs[0].nodes())
    prev_edges = set(test_graphs[0].edges())
    
    for u, v in test_graphs[1].edges():
        # Only consider edges where both nodes are in test_graphs[0]
        if u in test_graph_nodes and v in test_graph_nodes:
            if (u, v) not in prev_edges and (v, u) not in prev_edges:
                test_edges.append((u, v))
    
    print(f"   Test edges: {len(test_edges)}")
    
    # Sample negative edges - only from nodes in test_graphs[0]
    existing = prev_edges | {(v, u) for u, v in prev_edges}
    nodes_list = list(test_graph_nodes)
    attempts = 0
    while len(neg_edges) < len(test_edges) and attempts < len(test_edges) * 10:
        u, v = random.choice(nodes_list), random.choice(nodes_list)
        if u != v and (u, v) not in existing and (v, u) not in existing:
            neg_edges.append((u, v))
            existing.add((u, v))
        attempts += 1
    
    print(f"   Neg edges: {len(neg_edges)}")
    
    auc, ap = cn.fit_predict(test_graphs[0], test_edges, neg_edges)
    results['CN'] = {'auc': auc, 'ap': ap}
    print(f"   CN: AUC={auc:.4f}, AP={ap:.4f}")
    
    # 2. Adamic-Adar
    print("\n[2/5] Adamic-Adar...")
    aa = AdamicAdar()
    auc, ap = aa.fit_predict(test_graphs[0], test_edges, neg_edges)
    results['AA'] = {'auc': auc, 'ap': ap}
    print(f"   AA: AUC={auc:.4f}, AP={ap:.4f}")
    
    # 3. Preferential Attachment
    print("\n[3/5] Preferential Attachment...")
    pa = PreferentialAttachment()
    auc, ap = pa.fit_predict(test_graphs[0], test_edges, neg_edges)
    results['PA'] = {'auc': auc, 'ap': ap}
    print(f"   PA: AUC={auc:.4f}, AP={ap:.4f}")
    
    # 4. GCN Baseline
    print("\n[4/5] GCN Baseline (no temporal)...")
    gcn_trainer = GCNBaselineTrainer(config)
    gcn_trainer.build_global_node_set = lambda g: None  # Dummy
    
    # Train GCN baseline
    for epoch in range(20):
        gcn_trainer.train_epoch(train_graphs, node_to_idx, num_nodes)
    
    auc, ap = gcn_trainer.evaluate(test_graphs[0], test_graphs[1], node_to_idx, num_nodes)
    results['GCN'] = {'auc': auc, 'ap': ap}
    print(f"   GCN: AUC={auc:.4f}, AP={ap:.4f}")
    
    # 5. GCN_MA (from previous results)
    print("\n[5/5] GCN_MA (with temporal)...")
    # Load saved results if available
    import json
    from pathlib import Path
    results_file = Path('results') / f'{dataset_name}_results.json'
    if results_file.exists():
        with open(results_file) as f:
            gcn_ma_results = json.load(f)
        results['GCN_MA'] = {'auc': gcn_ma_results['test_auc'], 'ap': gcn_ma_results['test_ap']}
        print(f"   GCN_MA: AUC={results['GCN_MA']['auc']:.4f}, AP={results['GCN_MA']['ap']:.4f}")
    else:
        print(f"   GCN_MA: No results found (run main.py first)")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'AUC':<10} {'AP':<10}")
    print("-" * 35)
    for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]['auc']):
        print(f"{model_name:<15} {metrics['auc']:<10.4f} {metrics['ap']:<10.4f}")
    
    return results


if __name__ == "__main__":
    import yaml
    from gcn_ma.data_loader import DynamicNetworkDataset, TrainTestSplitter
    
    # Load config
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    dataset = DynamicNetworkDataset(name='CollegeMsg', data_dir='./data')
    graphs = dataset.load_or_process()
    
    # Split
    splitter = TrainTestSplitter(graphs, train_ratio=0.8)
    train_graphs = splitter.get_train_graphs()
    val_graphs = splitter.get_test_graphs()[:2]
    test_graphs = splitter.get_test_graphs()[1:]
    
    # Run comparison
    results = run_baseline_comparison('CollegeMsg', graphs, train_graphs, val_graphs, test_graphs, config)
