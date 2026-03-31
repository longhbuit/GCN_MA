# Dynamic Network Link Prediction with Node Representation Learning from Graph Convolutional Networks

**Tạp chí:** Scientific Reports (Nature, 2023)  
**DOI:** PMC10766634  
**Nguồn:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10766634/

---

## Abstract

Dynamic network link prediction is extensively applicable in various scenarios, and it has progressively emerged as a focal point in data mining research. The comprehensive and accurate extraction of node information, as well as a deeper understanding of the temporal evolution pattern, are particularly crucial in the investigation of link prediction in dynamic networks. To address this issue, this paper introduces a node representation learning framework based on Graph Convolutional Networks (GCN), referred to as GCN_MA. This framework effectively combines GCN, Recurrent Neural Networks (RNN), and multi-head attention to achieve comprehensive and accurate representations of node embedding vectors. It aggregates network structural features and node features through GCN and incorporates an RNN with multi-head attention mechanisms to capture the temporal evolution patterns of dynamic networks from both global and local perspectives. Additionally, a node representation algorithm based on the node aggregation effect (NRNAE) is proposed, which synthesizes information including node aggregation and temporal evolution to comprehensively represent the structural characteristics of the network. The effectiveness of the proposed method for link prediction is validated through experiments conducted on six distinct datasets. The experimental outcomes demonstrate that the proposed approach yields satisfactory results in comparison to state-of-the-art baseline methods.

**Subject terms:** Mathematics and computing, Computer science

---

## Introduction

The objective of link prediction for dynamic networks is to evaluate the probability of future connections between nodes. Owing to the rapid advancement of communication networks, the Internet, and the big data era, dynamic network analysis has emerged as a crucial research problem, attracting the attention of experts from various fields towards dynamic network link prediction.

### Applications

**In biology:**
- Protein interaction prediction
- Evolution of metabolic network
- Conversion mechanism of signal transduction between proteins
- Drug design, disease understanding and gene regulation

**In social domain:**
- Network evolution of social users
- Social media marketing
- Information dissemination research
- Social dynamic analysis

**Other fields:** finance, transportation networks, environmental science

### Challenges

The existing works on dynamic network link prediction face two primary challenges:

1. **Employing a single neighbor information** to represent nodes overlooks the influence of node clustering, neighbor relationship, and time evolution in the network.

2. **Constructing temporal attribute models** — the spotlight is often narrowed down to the evolution pattern of the global time step, neglecting the impact of short-term connections and feature changes between nodes and their neighbors in a single time step.

### Proposed Solution

This study presents a GCN_MA framework with the following approach:
- **NRNAE algorithm**: enriches the node information representation using node degree, clustering coefficient and neighbor relationship
- **GCN**: aggregates multi-dimensional features to learn node embedding vectors
- **RNN with multi-head attention**: models time attributes from global and local perspectives

### Contributions

- Proposes a GCN-based node representation learning framework that captures temporal attributes by examining global and local information fluctuations
- Proposes a novel **NRNAE algorithm** to enrich the structural features of the network
- Conducts comprehensive experiments to validate the efficacy of the GCN_MA framework

---

## Related Research

In recent years, a multitude of dynamic network link prediction techniques have been proposed:

### Similarity-based Methods
- **NCC and NCCP** (Chen et al.): similarity measures founded on the clustering coefficient of neighboring nodes
- **Dynamic similarity prediction** (Wu et al.): calculates similarity via algorithm based on node ranking
- **Node centrality with time series** (Zhang et al.): appraises impact of common neighbors in dynamic networks

### Graph Convolutional Network Methods
- **DyGCN** (Cui et al.): GCN variant that updates node embeddings to propagate embedding information for dynamic networks
- **HGCN** (Chami et al.): hyperbolic graph convolutional neural networks for hierarchical and scale-free graphs

### Deep Learning Methods
- **GraphLP** (Xian et al.): link prediction model based on network reconstruction theory
- **Triadic closure-based evolution patterns** (Zhou et al.)
- **DynGEM** (Goyal et al.): deep autoencoder model which progressively updates node embeddings
- **GC-LSTM** (Chen et al.): combines LSTM and GCN to capture local structural attributes

---

## Definitions and Methods

### Definition of a Dynamic Network

A dynamic network can be represented as a sequence of discrete snapshots:

```
G = {G1, G2, …, GT}
```

Where `Gt = (V, Et, At)` (t ∈ [1,T]) represents the t-th time network snapshot:
- `V`: set of all nodes
- `Et`: set of edges within a fixed time interval [t-τ, t]
- `At`: adjacency matrix of Gt (At(i,j)=1 if there is a link between nodes i and j, otherwise 0)

### Definition of Link Prediction in Dynamic Networks

The link prediction in dynamic networks aims to forecast the adjacency matrix A^T+1 corresponding to the next time step snapshot GT+1 at time step T+1:

```
A^T+1 = f(A1, A2, …, AT)
```

Where:
- `f(·)`: the model to be constructed
- `A^T+1 ∈ R^N×N`: the predicted value

### Node Representation Learning Framework (GCN_MA)

The framework integrates:
1. **NRNAE algorithm**: mines network information, enriching network structure features {A~1, A~2, …, A~T}
2. **Node degree matrix**: used as node features {X1, X2, …, XT}
3. **GCN**: aggregates multi-dimensional features to learn embedding vectors {HT1, HT2, …, HTN}
4. **Improved LSTM**: continuously updates GCN parameters to model global time attributes
5. **Multi-head attention**: captures local structure information for each time snapshot
6. **MLP**: calculates probability value of edges for link prediction

### Node Information Representation (NRNAE Algorithm)

#### Clustering Coefficient (CC)
```
CC(i) = 2Ri / Ki(Ki-1)
```
Where:
- `Ri`: number of triangles formed by node i and its first-order neighbor nodes
- `Ki`: number of first-order neighbor nodes of i
- `CC(i) ∈ [0,1]`

#### Aggregation Strength (AS)
```
AS(i) = degree(i) × CC(i)
```
Describes the probability of focusing on a node to form a cluster, reflecting the importance and influence of the node.

#### Node Aggregation Effect
```
S(i,j) = |N(i) ∩ N(j)| × AS(i)
```
Where:
- `N(i)`: set of first-order neighbor nodes of node i
- `j ∈ N(i)`

#### New Adjacency Matrix
```
A~T = AT + βST
```
Where `β ∈ [0,1]` is a weighting factor.

### Global Time Attribute Modeling Based on LSTM

LSTM is used to capture temporal properties across different time steps. The weight matrix WT-1 generated by the GCN at the previous time step is utilized as the input to the LSTM to produce the weight matrix WT at the subsequent time step:

```
WT = LSTM(WT-1)
```

LSTM equations:
```
FT = σ(MFWT-1 + UFHT-1 + bF)    # Forget gate
IT = σ(MIWT-1 + UIHT-1 + bI)    # Input gate
OT = σ(MOWT-1 + UOHT-1 + bO)    # Output gate
C~T = tanh(MCWT-1 + UCHT-1 + bC) # Cell state update
CT = FT ∘ CT-1 + IT ∘ C~T       # Cell state
WT = OT ∘ tanh(CT)              # Weight output
```

### Local Time Attribute Modeling Based on Multi-Head Attention

The multi-head attention model captures the rapid shifts in link status and node characteristics from a local perspective.

For each attention head:
```
QT = WqHT ∈ R^(DK×N)   # Query
KT = WkHT ∈ R^(DK×N)   # Key
VT = WvHT ∈ R^(DV×N)   # Value
```

Scaled dot-product attention:
```
Attention(QT, KT, VT) = softmax(QTKTT / √dK) VT
```

Multi-head concatenation:
```
headi = Attention(QWiQ, KWiK, VWiV)
ZT = Concat(head1, head2, …, headh) Wo
```

### Link Prediction and Loss Function

Reformulated as binary classification using MLP:
```
PT = σ(MLP(ZT))
```

Binary cross-entropy loss:
```
Loss = -1/N² ∑∑ (YT log(PT) + (1-YT) log(1-PT))
```

---

## Results and Analysis

### Dataset Statistics

| Dataset | Data Type | Number of Nodes | Number of Edges | Time Steps |
|---------|-----------|-----------------|-----------------|------------|
| CollegeMsg | Social | 1,899 | 59,835 | 47 |
| Mooc_actions | Education | 7,047 | 411,749 | 72 |
| Bitcoinotc | Financial | 6,005 | 35,592 | 62 |
| EUT (Email-eu-core-temporal) | Communication | 1,005 | 332,334 | 127 |
| LastFM | Music | 1,000 | 1,293,103 | 76 |
| Wikipedia | Collaboration | 5,684 | 87,931 | 42 |

### Baseline Methods

- **DGCN**: Uses dice similarity and LSTM to capture global structural information
- **HTGN**: Hyperbolic graph neural networks with hyperbolic gated recurrent neural networks
- **DyGNN**: Continuously updates node information by capturing edge sequence information
- **EvolveGCN**: Evolves GCN parameters using RNN (GRU/LSTM)

### Evaluation Metrics

- **AUC (Area Under Curve)**: Measures overall accuracy
- **AP (Average Precision)**: Considers accuracy for top-L ranked edges

### Experimental Results

#### AUC Comparison

| Methods | Mooc-action | CollegeMsg | ETU | Bitcoinotc | LastFM | Wikipedia |
|---------|-------------|------------|-----|------------|--------|-----------|
| DyGNN | 0.9242 | 0.8856 | 0.7527 | 0.8769 | 0.8034 | 0.8371 |
| EvolveGCN | 0.7794 | 0.7867 | 0.8494 | 0.7811 | 0.8593 | 0.6289 |
| HTGN | 0.9712 | 0.8491 | 0.8694 | 0.8814 | 0.7151 | 0.8414 |
| DGCN | 0.9720 | 0.8799 | 0.8947 | 0.9046 | 0.8201 | 0.8472 |
| **GCN_MA** | **0.9880** | **0.9149** | **0.9222** | **0.9120** | **0.8757** | **0.8742** |

#### AP Comparison

| Methods | Mooc-action | CollegeMsg | ETU | Bitcoinotc | LastFM | Wikipedia |
|---------|-------------|------------|-----|------------|--------|-----------|
| DyGNN | 0.9179 | 0.8839 | 0.7519 | 0.8617 | 0.8067 | 0.8371 |
| EvolveGCN | 0.7602 | 0.7620 | 0.8453 | 0.7838 | 0.8566 | 0.6163 |
| HTGN | 0.9773 | 0.8813 | 0.8642 | 0.8752 | 0.7088 | 0.8624 |
| DGCN | 0.9657 | 0.8813 | 0.8788 | 0.8884 | 0.7781 | 0.8379 |
| **GCN_MA** | **0.9863** | **0.8926** | **0.9082** | **0.8943** | **0.8704** | **0.8575** |

### Parameter Analysis

The parameter β in NRNAE algorithm was varied from 0.0 to 1.0:

- **Optimal range**: β between 0.7 and 0.9
- At optimal β, GCN_MA shows an average increase of **0.52% in AUC** and **0.45% in AP** compared to β=0

### Ablation Experiments

#### AUC Results

| Methods | LastFM | Bitcoinotc | ETU | Wikipedia | CollegeMsg | Mooc_action |
|---------|--------|------------|-----|-----------|------------|--------------|
| GCN_MA (Full) | **0.8757** | **0.9120** | **0.9222** | **0.8742** | **0.9149** | **0.9880** |
| GCN_MultiAttention | 0.8702 | 0.8916 | 0.9158 | 0.8407 | 0.8809 | 0.9865 |
| GCN_LSTM | 0.7999 | 0.8743 | 0.8880 | 0.8241 | 0.8694 | 0.9734 |
| GCN (baseline) | 0.7774 | 0.8718 | 0.8755 | 0.8205 | 0.8722 | 0.9626 |

#### Ablation Analysis

- **GCN_MA vs GCN_MultiAttention**: +1.69% AUC, +0.9% AP → LSTM contribution for global time evolution
- **GCN_MA vs GCN_LSTM**: +4.3% AUC, +3.34% AP → Multi-head attention crucial for local information changes
- **GCN_MA vs GCN**: +5.12% AUC, +3.62% AP → Combination of both mechanisms is effective

---

## Conclusion

### Summary

The paper proposes **GCN_MA**, a node representation learning framework based on graph convolutional networks that:

1. Uses **NRNAE algorithm** to enrich node information representation
2. Employs **GCN** to aggregate structural features and node features
3. Combines **RNN with multi-head attention** to model dynamic networks from global and local perspectives
4. Achieves best performance across 6 datasets compared to 4 state-of-the-art baseline methods

### Limitations (Acknowledged by Authors)

- Focuses only on **discrete dynamic networks with homogeneous network type**
- Does not cover heterogeneous dynamic networks and time-continuous dynamic networks

### Future Work

- Shift attention towards **heterogeneous dynamic networks** and **time-continuous dynamic networks**
- Consider both **structural similarity** and **feature-based similarity** as measures for node similarity

---

## Author Contributions

- **P.M. and Y.Z.**: Proposed GCN_MA framework and NRNAE algorithms
- **P.M. and Y.Z.**: Conceived and designed simulation experiments
- **P.M.**: Conducted experiments
- **P.M. and Y.Z.**: Analyzed results and wrote manuscript

## Funding

Natural Science Foundation of Inner Mongolia Province of China (Grant No. 2022MS06006)

## Data Availability

Datasets available at:
- http://snap.stanford.edu/data/act-mooc.html
- http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- http://snap.stanford.edu/data/email-Eu-core-temporal.html
- https://meta.wikimedia.org/wiki/Data_dumps

---

## References

1. Xu F, et al. Specificity and competition of mRNAs dominate droplet pattern in protein phase separation. Phys. Rev. Res. 2023;5(2):023159.
2. Sun F, Sun J, Zhao Q. A deep learning method for predicting metabolite-disease associations via graph neural network. Brief Bioinform. 2022;23(4):266.
3. Li X, et al. Caspase-1 and Gasdermin D afford the optimal targets with distinct switching strategies in NLRP1b inflammasome-induced cell death. Research. 2022.
4. Li X, et al. RIP1-dependent linear and nonlinear recruitments of caspase-8 and RIP3 respectively to necrosome specify distinct cell death outcomes. Protein Cell. 2021;12(11):858–876.
5. Wang T, Sun J, Zhao Q. Investigating cardiotoxicity related with hERG channel blockers using molecular fingerprints and graph attention mechanism. Comput. Biol. Med. 2023;153:106464.
6. Liu W, et al. NSCGRN: A network structure control method for gene regulatory network inference. Brief Bioinform. 2023;23(5):106464.
7. Daud NN, et al. Applications of link prediction in social networks: A review. J. Netw. Comput. Appl. 2022;166:102716.

---

*Document generated on: 2026-03-31*
