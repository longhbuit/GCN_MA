"""
GCN_MA: Dynamic Network Link Prediction with Node Representation Learning
=========================================================================

A PyTorch implementation of the GCN_MA framework for link prediction
in dynamic networks.

Components:
- NRNAE: Node Representation based on Node Aggregation Effect
- GCN: Graph Convolutional Network
- LSTM: Global temporal modeling via weight updates
- Multi-head Attention: Local temporal modeling
- MLP: Link prediction classifier
"""

from gcn_ma.data_loader import DynamicNetworkDataset, TrainTestSplitter
from gcn_ma.nrnae import NRNAE
from gcn_ma.gcn_layer import GCN, GCNWithNAE
from gcn_ma.lstm_updater import LSTMWeightUpdater, GCNLSTMModel
from gcn_ma.attention import MultiHeadAttention, LocalTemporalAttention
from gcn_ma.model import LinkPredictor, LinkPredictionDataset
from gcn_ma.trainer import GCN_MA_Model, GCN_MA_Trainer

__version__ = "1.0.0"
__all__ = [
    'DynamicNetworkDataset',
    'TrainTestSplitter',
    'NRNAE',
    'GCN',
    'GCNWithNAE',
    'LSTMWeightUpdater',
    'GCNLSTMModel',
    'MultiHeadAttention',
    'LocalTemporalAttention',
    'LinkPredictor',
    'LinkPredictionDataset',
    'GCN_MA_Model',
    'GCN_MA_Trainer',
]
