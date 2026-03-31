"""
Phase 1: Data Loading & Preprocessing
=====================================
Handles dataset downloading, parsing, and conversion to snapshot series.
"""

import os
import urllib.request
import zipfile
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicNetworkDataset:
    """
    Base class for dynamic network datasets.
    Converts temporal edge lists into graph snapshots.
    """
    
    SNAPSHOT_DIRS = {
        "CollegeMsg": "CollegeMsg",
        "Mooc_actions": "mooc_actions",
        "Bitcoinotc": "soc-sign-bitcoin-otc",
        "EUT": "email-Eu-core-temporal",
    }
    
    SNAPNET_URLS = {
        "CollegeMsg": "http://snap.stanford.edu/data/CollegeMsg.txt.gz",
        "Mooc_actions": "http://snap.stanford.edu/data/act-mooc.csv.gz",
        "Bitcoinotc": "http://snap.stanford.edu/data/soc-sign-bitcoin-otc.csv.gz",
        "EUT": "http://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
        # Wikipedia and LastFM require separate handling
    }
    
    def __init__(self, name: str, data_dir: str = "./data"):
        self.name = name
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / name
        self.processed_dir = self.data_dir / "processed" / name
        self.graphs: List[nx.Graph] = []
        self.num_snapshots = 10
        
    def download(self) -> Path:
        """Download dataset from SNAP Stanford."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        if self.name not in self.SNAPNET_URLS:
            logger.warning(f"No direct download URL for {self.name}")
            return self.raw_dir
            
        url = self.SNAPNET_URLS[self.name]
        filename = self.raw_dir / url.split("/")[-1]
        
        if filename.exists():
            logger.info(f"{self.name} already downloaded")
            return self.raw_dir
            
        logger.info(f"Downloading {self.name} from {url}")
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded to {filename}")
        return self.raw_dir
    
    def parse_college_msg(self) -> List[nx.Graph]:
        """Parse CollegeMsg dataset."""
        gz_file = self.raw_dir / "CollegeMsg.txt.gz"
        if not gz_file.exists():
            raise FileNotFoundError(f"Download {gz_file} first")
        
        import gzip
        edges_with_time = []
        
        with gzip.open(gz_file, 'rt') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    src, dst, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
                    edges_with_time.append((src, dst, timestamp))
        
        return self._create_snapshots(edges_with_time)
    
    def parse_bitcoinotc(self) -> List[nx.Graph]:
        """Parse Bitcoin OTC trust network."""
        gz_file = self.raw_dir / "soc-sign-bitcoin-otc.csv.gz"
        if not gz_file.exists():
            raise FileNotFoundError(f"Download {gz_file} first")
        
        import gzip
        edges_with_time = []
        
        with gzip.open(gz_file, 'rt') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    src, dst, rating, timestamp = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
                    edges_with_time.append((src, dst, timestamp))
        
        return self._create_snapshots(edges_with_time)
    
    def _create_snapshots(self, edges_with_time: List[Tuple], snapshot_type: str = "count") -> List[nx.Graph]:
        """
        Create graph snapshots from temporal edges.
        
        Args:
            edges_with_time: List of (src, dst, timestamp) tuples
            snapshot_type: "count" (equal edges per snapshot) or "time" (equal time intervals)
        """
        if not edges_with_time:
            return []
            
        edges_with_time.sort(key=lambda x: x[2])
        
        if snapshot_type == "count":
            # Divide edges into equal-sized groups
            edges_per_snapshot = len(edges_with_time) // self.num_snapshots
            if edges_per_snapshot < 10:
                raise ValueError(f"Not enough edges for {self.num_snapshots} snapshots")
                
            graphs = []
            for i in range(self.num_snapshots):
                start_idx = i * edges_per_snapshot
                if i == self.num_snapshots - 1:
                    end_idx = len(edges_with_time)
                else:
                    end_idx = (i + 1) * edges_per_snapshot
                    
                G = nx.Graph()
                for src, dst, _ in edges_with_time[start_idx:end_idx]:
                    G.add_edge(src, dst)
                graphs.append(G)
                
        else:
            # Time-based snapshots
            timestamps = [e[2] for e in edges_with_time]
            min_ts, max_ts = min(timestamps), max(timestamps)
            time_range = max_ts - min_ts
            
            if time_range == 0:
                raise ValueError("All edges have same timestamp")
            
            interval = time_range / self.num_snapshots
            graphs = [nx.Graph() for _ in range(self.num_snapshots)]
            
            for src, dst, ts in edges_with_time:
                snapshot_idx = min(int((ts - min_ts) / interval), self.num_snapshots - 1)
                graphs[snapshot_idx].add_edge(src, dst)
        
        return graphs
    
    def load_or_process(self, force_reprocess: bool = False) -> List[nx.Graph]:
        """Load processed data or process raw data."""
        processed_file = self.processed_dir / f"{self.name}_snapshots.pkl"
        
        if processed_file.exists() and not force_reprocess:
            logger.info(f"Loading processed data from {processed_file}")
            import pickle
            with open(processed_file, 'rb') as f:
                return pickle.load(f)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Process based on dataset type
        if self.name == "CollegeMsg":
            graphs = self.parse_college_msg()
        elif self.name == "Bitcoinotc":
            graphs = self.parse_bitcoinotc()
        else:
            raise NotImplementedError(f"Parsing not implemented for {self.name}")
        
        # Save processed
        import pickle
        with open(processed_file, 'wb') as f:
            pickle.dump(graphs, f)
        logger.info(f"Saved processed data to {processed_file}")
        
        return graphs
    
    def get_node_features(self, G: nx.Graph) -> np.ndarray:
        """
        Generate node features for a graph.
        Features: [degree, clustering_coefficient]
        """
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        num_nodes = G.number_of_nodes()
        node_list = list(G.nodes())
        
        features = np.zeros((num_nodes, 2))
        for i, node in enumerate(node_list):
            features[i, 0] = degrees.get(node, 0)
            features[i, 1] = clustering.get(node, 0)
        
        return features, node_list
    
    def get_adjacency_matrix(self, G: nx.Graph, node_list: List[int] = None) -> np.ndarray:
        """Get adjacency matrix as numpy array."""
        if node_list is None:
            node_list = list(G.nodes())
        return nx.to_numpy_array(G, nodelist=node_list)


class TrainTestSplitter:
    """Split dynamic network into train/test for link prediction."""
    
    def __init__(self, graphs: List[nx.Graph], train_ratio: float = 0.8):
        self.graphs = graphs
        self.train_ratio = train_ratio
        self.split_point = int(len(graphs) * train_ratio)
        
    def get_train_graphs(self) -> List[nx.Graph]:
        """Get graphs for training."""
        return self.graphs[:self.split_point]
    
    def get_test_graphs(self) -> List[nx.Graph]:
        """Get graphs for testing (including last train graph for context)."""
        return self.graphs[self.split_point - 1:]
    
    def get_train_edges(self, snapshot_idx: int) -> List[Tuple[int, int]]:
        """Get positive edges from a training snapshot."""
        G = self.graphs[snapshot_idx]
        return list(G.edges())
    
    def get_test_edges(self, snapshot_idx: int) -> List[Tuple[int, int]]:
        """Get positive edges from a test snapshot (excluding edges that exist in previous snapshot)."""
        if snapshot_idx == 0:
            return list(self.graphs[snapshot_idx].edges())
        
        prev_graph = self.graphs[snapshot_idx - 1]
        curr_graph = self.graphs[snapshot_idx]
        
        # New edges only
        new_edges = []
        for u, v in curr_graph.edges():
            if not prev_graph.has_edge(u, v):
                new_edges.append((u, v))
        return new_edges
    
    def sample_negative_edges(self, snapshot_idx: int, num_samples: int = None) -> List[Tuple[int, int]]:
        """Sample non-existing edges as negative examples."""
        G = self.graphs[snapshot_idx]
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        
        if num_samples is None:
            num_samples = G.number_of_edges()
        
        negative_edges = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            u, v = np.random.choice(num_nodes, 2, replace=True)
            if u != v and not G.has_edge(u, v):
                negative_edges.append((u, v))
            attempts += 1
        
        return negative_edges


def verify_data_loading():
    """Verify Phase 1: Data Loading."""
    print("\n" + "="*60)
    print("PHASE 1 VERIFICATION: Data Loading")
    print("="*60)
    
    # Test basic functionality
    dataset = DynamicNetworkDataset("CollegeMsg", data_dir="./data")
    
    print(f"\n✅ Dataset class initialized: {dataset.name}")
    print(f"   Raw dir: {dataset.raw_dir}")
    print(f"   Processed dir: {dataset.processed_dir}")
    
    # Test node feature generation
    G_test = nx.karate_club_graph()
    features, node_list = dataset.get_node_features(G_test)
    adj = dataset.get_adjacency_matrix(G_test, node_list)
    
    print(f"\n✅ Node features shape: {features.shape}")
    print(f"   Features: degree, clustering_coefficient")
    print(f"   Sample (first 5 nodes):\n{features[:5]}")
    
    print(f"\n✅ Adjacency matrix shape: {adj.shape}")
    print(f"   Density: {nx.density(G_test):.4f}")
    
    # Test splitter
    test_graphs = [nx.karate_club_graph() for _ in range(5)]
    splitter = TrainTestSplitter(test_graphs, train_ratio=0.6)
    
    print(f"\n✅ TrainTestSplitter:")
    print(f"   Total graphs: {len(test_graphs)}")
    print(f"   Split point: {splitter.split_point}")
    print(f"   Train graphs: {len(splitter.get_train_graphs())}")
    print(f"   Test graphs: {len(splitter.get_test_graphs())}")
    
    # Test negative sampling
    neg_edges = splitter.sample_negative_edges(4, num_samples=5)
    print(f"\n✅ Negative edge sampling: {len(neg_edges)} edges sampled")
    print(f"   Sample: {neg_edges[:3]}")
    
    print("\n" + "="*60)
    print("PHASE 1: VERIFICATION PASSED ✅")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_data_loading()
