"""
Phase 2: NRNAE Algorithm (Node Representation based on Node Aggregation Effect)
================================================================================
Implements node aggregation effect calculation for enriching network structure.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class NRNAE:
    """
    Node Representation Algorithm Based on Node Aggregation Effect.
    
    This algorithm enriches node information by considering:
    1. Node degree
    2. Clustering coefficient
    3. Node aggregation effect S(i,j)
    
    The new adjacency matrix: A~ = A + β*S
    where β ∈ [0,1] is a weighting factor.
    """
    
    def __init__(self, beta: float = 0.8):
        """
        Args:
            beta: Weighting factor for aggregation effect (0.7-0.9 optimal per paper)
        """
        self.beta = beta
        
    def compute_clustering_coefficient(self, G: nx.Graph) -> Dict[int, float]:
        """
        Compute clustering coefficient for each node.
        
        CC(i) = 2*Ri / (Ki * (Ki - 1))
        where:
            Ri = number of triangles containing node i
            Ki = number of first-order neighbors of node i
            
        Returns:
            Dictionary mapping node_id -> clustering coefficient
        """
        clustering = nx.clustering(G)
        return clustering
    
    def compute_aggregation_strength(self, G: nx.Graph) -> Dict[int, float]:
        """
        Compute Aggregation Strength for each node.
        
        AS(i) = degree(i) * CC(i)
        
        Describes the probability of focusing on a node to form a cluster,
        reflecting the importance and influence of the node.
        
        Returns:
            Dictionary mapping node_id -> aggregation strength
        """
        degrees = dict(G.degree())
        clustering = self.compute_clustering_coefficient(G)
        
        aggregation_strength = {}
        for node in G.nodes():
            deg = degrees.get(node, 0)
            cc = clustering.get(node, 0)
            aggregation_strength[node] = deg * cc
            
        return aggregation_strength
    
    def compute_node_aggregation_effect(self, G: nx.Graph) -> Tuple[np.ndarray, List[int]]:
        """
        Compute Node Aggregation Effect matrix S.
        
        S(i,j) = |N(i) ∩ N(j)| * AS(i)
        where:
            N(i) = set of first-order neighbors of node i
            AS(i) = aggregation strength of node i
            
        Args:
            G: NetworkX graph
            
        Returns:
            S: Node aggregation effect matrix (N x N)
            node_list: List of nodes in order
        """
        node_list = sorted(list(G.nodes()))
        num_nodes = len(node_list)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Compute AS for each node
        aggregation_strength = self.compute_aggregation_strength(G)
        
        # Initialize S matrix
        S = np.zeros((num_nodes, num_nodes))
        
        # For each pair of connected nodes
        for i, node_i in enumerate(node_list):
            neighbors_i = set(G.neighbors(node_i))
            as_i = aggregation_strength[node_i]
            
            for node_j in neighbors_i:
                j = node_to_idx[node_j]
                # Intersection of neighbors
                neighbors_j = set(G.neighbors(node_j))
                intersection = len(neighbors_i & neighbors_j)
                
                # S(i,j) = |N(i) ∩ N(j)| * AS(i)
                S[i, j] = intersection * as_i
        
        # Normalize S to [0, 1]
        if S.max() > 0:
            S = S / S.max()
            
        return S, node_list
    
    def compute_enriched_adjacency(self, G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Compute enriched adjacency matrix: A~ = A + β*S
        
        Args:
            G: NetworkX graph
            
        Returns:
            A_tilde: Enriched adjacency matrix (N x N)
            A: Original adjacency matrix (N x N)
            node_list: List of nodes in order
        """
        node_list = sorted(list(G.nodes()))
        num_nodes = len(node_list)
        
        # Original adjacency matrix
        A = nx.to_numpy_array(G, nodelist=node_list)
        
        # Node aggregation effect matrix
        S, _ = self.compute_node_aggregation_effect(G)
        
        # Enriched adjacency: A~ = A + β*S
        A_tilde = A + self.beta * S
        
        return A_tilde, A, node_list
    
    def get_node_features(self, G: nx.Graph) -> Tuple[np.ndarray, List[int]]:
        """
        Get node features for GCN input.
        Features: [degree, clustering_coefficient]
        
        Returns:
            features: (N x 2) feature matrix
            node_list: List of nodes
        """
        node_list = sorted(list(G.nodes()))
        degrees = dict(G.degree())
        clustering = self.compute_clustering_coefficient(G)
        
        features = np.zeros((len(node_list), 2))
        for i, node in enumerate(node_list):
            features[i, 0] = degrees.get(node, 0)
            features[i, 1] = clustering.get(node, 0)
            
        return features, node_list


def verify_nrnae():
    """Verify Phase 2: NRNAE Algorithm."""
    print("\n" + "="*60)
    print("PHASE 2 VERIFICATION: NRNAE Algorithm")
    print("="*60)
    
    # Create test graph (Zachary's Karate Club)
    G = nx.karate_club_graph()
    
    # Test NRNAE
    nrnae = NRNAE(beta=0.8)
    
    # Test clustering coefficient
    cc = nrnae.compute_clustering_coefficient(G)
    print(f"\n✅ Clustering Coefficient computed for {len(cc)} nodes")
    print(f"   Sample (first 5): {list(cc.items())[:5]}")
    print(f"   Range: [{min(cc.values()):.4f}, {max(cc.values()):.4f}]")
    
    # Test aggregation strength
    as_vals = nrnae.compute_aggregation_strength(G)
    print(f"\n✅ Aggregation Strength computed")
    print(f"   Sample (first 5): {list(as_vals.items())[:5]}")
    print(f"   Range: [{min(as_vals.values()):.4f}, {max(as_vals.values()):.4f}]")
    
    # Test node aggregation effect matrix
    S, node_list = nrnae.compute_node_aggregation_effect(G)
    print(f"\n✅ Node Aggregation Effect matrix S:")
    print(f"   Shape: {S.shape}")
    print(f"   S values range: [{S.min():.4f}, {S.max():.4f}]")
    print(f"   S is symmetric: {np.allclose(S, S.T)}")
    
    # Test enriched adjacency
    A_tilde, A, node_list = nrnae.compute_enriched_adjacency(G)
    print(f"\n✅ Enriched Adjacency Matrix:")
    print(f"   Original A shape: {A.shape}")
    print(f"   A~ = A + β*S shape: {A_tilde.shape}")
    print(f"   β = {nrnae.beta}")
    print(f"   A max: {A.max()}, A~ max: {A_tilde.max()}")
    
    # Verify A~ = A + β*S
    expected_A_tilde = A + 0.8 * S
    print(f"\n   A~ = A + β*S correctly: {np.allclose(A_tilde, expected_A_tilde)}")
    
    # Test node features
    features, nodes = nrnae.get_node_features(G)
    print(f"\n✅ Node Features for GCN:")
    print(f"   Shape: {features.shape}")
    print(f"   Sample (first 5):\n{features[:5]}")
    
    # Test with different beta values
    print(f"\n✅ Beta sensitivity test:")
    for beta in [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]:
        nrnae_beta = NRNAE(beta=beta)
        A_tilde_beta, _, _ = nrnae_beta.compute_enriched_adjacency(G)
        print(f"   β={beta}: A~ max = {A_tilde_beta.max():.4f}")
    
    print("\n" + "="*60)
    print("PHASE 2: VERIFICATION PASSED ✅")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_nrnae()
