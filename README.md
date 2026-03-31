# GCN_MA: Dynamic Network Link Prediction

## 📖 Mục lụ
- [Giới thiệu](#giới-thiệu)
- [Cài đặt](#cài-đặt)
- [Cấu trúc Project](#cấu-trúc-project)
- [Các lệnh thực thi](#các-lệnh-thựcthi)
- [Dataset](#dataset)
- [Mô hình](#mô-hình)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Jupyter Notebook Report](#jupyter-notebook-report)
- [Cấu hình](#cấu-hình)

---

## Giới thiệu

**GCN_MA** (Graph Convolutional Networks with Multi-head Attention) là implementation của paper:

> *Dynamic network link prediction with node representation learning from graph convolutional networks*  
> Scientific Reports (Nature), 2023

### Đóng góp chính:
- **NRNAE Algorithm**: Enriches node features using node aggregation effect
- **GCN**: Graph Convolutional Networks for structural learning
- **LSTM**: Global temporal modeling (weight updates)
- **Multi-head Attention**: Local temporal modeling

---

## Cài đặt

### Yêu cầu
- Python 3.8+
- PyTorch 2.0+
- NetworkX
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn

### Các bước cài đặt

```bash
# 1. Clone hoặc di chuyển vào thư mục project
cd /Volumes/DataCS/PycharmProjects/mxh

# 2. Cài đặt Python dependencies
pip install -r requirements.txt

# Hoặc cài thủ công:
pip install torch numpy networkx scikit-learn matplotlib seaborn pyyaml tqdm
```

### Kiểm tra cài đặt
```bash
python -c "import torch; import networkx; print('Installation OK')"
```

---

## Cấu trúc Project

```
mxh/
├── configs/
│   └── config.yaml              # Cấu hình hyperparameters
├── data/
│   ├── raw/                     # Dữ liệu gốc (sau khi download)
│   └── processed/               # Dữ liệu đã xử lý (pickle files)
├── gcn_ma/
│   ├── __init__.py
│   ├── baselines.py             # Baseline models (CN, AA, PA, GCN)
│   ├── attention.py             # Multi-head Attention
│   ├── data_loader.py          # Data loading & preprocessing
│   ├── gcn_layer.py            # GCN layers
│   ├── lstm_updater.py         # LSTM Weight Updater
│   ├── model.py                # Link Predictor
│   ├── nrnae.py                # NRNAE Algorithm
│   └── trainer.py               # Training & Evaluation
├── results/
│   ├── training_curves.png
│   ├── baseline_comparison.png
│   └── final_comparison.png
├── checkpoints/
│   └── best_model.pt            # Model checkpoint
├── main.py                      # Entry point chính
├── GCN_MA_Report.ipynb         # Jupyter notebook báo cáo
├── requirements.txt
└── README.md
```

---

## Các lệnh thực thi

### 1. Download Dataset

```bash
# Download CollegeMsg dataset
python main.py --dataset CollegeMsg --download
```

### 2. Training Model

```bash
# Training với CollegeMsg (mặc định)
python main.py --dataset CollegeMsg

# Training với các tham số tùy chỉnh
python main.py --dataset CollegeMsg --config configs/config.yaml
```

### 3. Baseline Comparison

```bash
# So sánh với các baseline methods
python -c "
import sys
sys.path.insert(0, '.')
from gcn_ma.baselines import run_baseline_comparison
from gcn_ma.data_loader import DynamicNetworkDataset, TrainTestSplitter
import yaml

# Load
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

dataset = DynamicNetworkDataset(name='CollegeMsg', data_dir='./data')
graphs = dataset.load_or_process()

splitter = TrainTestSplitter(graphs, train_ratio=0.8)
train_graphs = splitter.get_train_graphs()
val_graphs = splitter.get_test_graphs()[:2]
test_graphs = splitter.get_test_graphs()[1:]

results = run_baseline_comparison('CollegeMsg', graphs, train_graphs, val_graphs, test_graphs, config)
"
```

### 4. Verify Model Components

```bash
# Kiểm tra từng module
python gcn_ma/trainer.py
```

### 5. Jupyter Notebook

```bash
# Mở notebook
jupyter notebook GCN_MA_Report.ipynb

# Hoặc convert sang HTML
jupyter nbconvert --to html GCN_MA_Report.ipynb
```

---

## Dataset

### Các Dataset được hỗ trợ

| Dataset | Mô tả | Link |
|---------|-------|------|
| **CollegeMsg** | Social network messages | SNAP Stanford |
| **Mooc_actions** | MOOC student actions | SNAP Stanford |
| **Bitcoinotc** | Bitcoin trust network | SNAP Stanford |
| **EUT** | Email network | SNAP Stanford |

### Download Dataset thủ công

Nếu auto-download lỗi, có thể download thủ công:

```bash
# Tạo thư mục
mkdir -p data/raw/CollegeMsg

# Download từ SNAP Stanford
# http://snap.stanford.edu/data/CollegeMsg.txt.gz

# Đặt file .gz vào thư mục trên
# File sẽ được tự động parse khi chạy
```

### Xử lý Dataset

Dataset được tự động xử lý thành các snapshots:

```python
from gcn_ma.data_loader import DynamicNetworkDataset

dataset = DynamicNetworkDataset(name='CollegeMsg', data_dir='./data')
graphs = dataset.load_or_process()

# graphs: list of NetworkX graphs (10 snapshots)
print(f"Total snapshots: {len(graphs)}")
print(f"Snapshot 0: {graphs[0].number_of_nodes()} nodes")
```

---

## Mô hình

### Architecture Overview

```
Input: Dynamic Network Snapshots
         │
         ▼
┌─────────────────────────┐
│   NRNAE Algorithm       │
│   A~ = A + β×S         │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Graph Convolutional   │
│   Network (2 layers)   │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   LSTM Weight Updater  │
│   (Global Temporal)    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Multi-head Attention  │
│   (Local Temporal)     │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   MLP Link Predictor    │
│   [Z_i ⊕ Z_j] → P(i,j)│
└─────────────────────────┘
```

### Components

1. **NRNAE (Node Representation based on Node Aggregation Effect)**
   - Clustering Coefficient: CC(i) = 2*Ri / (Ki * (Ki-1))
   - Aggregation Strength: AS(i) = degree(i) × CC(i)
   - Node Aggregation Effect: S(i,j) = |N(i) ∩ N(j)| × AS(i)

2. **GCN Layer**
   - H^{l+1} = σ(D~^{-1/2} A~ D~^{-1/2} H^l W^l)

3. **Multi-head Attention**
   - 8 attention heads
   - Captures local temporal patterns

4. **Link Predictor**
   - MLP với 2 layers
   - Input: Concatenated node embeddings [Z_i || Z_j]

---

## Kết quả thực nghiệm

### Dataset: CollegeMsg

| Model | AUC | AP |
|-------|-----|-----|
| **GCN_MA** | **0.8638** | **0.8652** |
| PA (Preferential Attachment) | 0.8287 | 0.8140 |
| AA (Adamic-Adar) | 0.5799 | 0.5825 |
| CN (Common Neighbors) | 0.5791 | 0.5716 |
| GCN (without temporal) | 0.4406 | 0.4441 |

### Training Details

- **Best Validation AUC**: 0.9151
- **Best Epoch**: 28
- **Early Stopping**: Epoch 48 (patience=20)
- **Training Time**: ~1.7s per epoch

---

## Jupyter Notebook Report

Mở file `GCN_MA_Report.ipynb` để xem báo cáo chi tiết:

```bash
jupyter notebook GCN_MA_Report.ipynb
```

Hoặc xem trực tiếp:

```bash
# Convert sang HTML
jupyter nbconvert --to html GCN_MA_Report.ipynb

# Mở file HTML trong browser
open GCN_MA_Report.html
```

---

## Cấu hình

### File config.yaml

```yaml
# Model
model:
  name: "GCN_MA"
  seed: 42

# NRNAE parameters
nrnae:
  beta: 0.8  # Weight for aggregation effect

# GCN parameters
gcn:
  hidden_dim: 128
  output_dim: 128
  dropout: 0.3

# LSTM parameters
lstm:
  hidden_dim: 128
  num_layers: 1

# Attention parameters
attention:
  num_heads: 8
  dropout: 0.1

# Training parameters
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
  early_stopping_patience: 20
  device: "cuda"  # hoặc "cpu"
```

### Các tham số quan trọng

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nrnae.beta` | 0.8 | NRNAE weight (0.7-0.9 optimal) |
| `gcn.hidden_dim` | 128 | Hidden dimension |
| `gcn.output_dim` | 128 | Output embedding dimension |
| `attention.num_heads` | 8 | Number of attention heads |
| `training.learning_rate` | 0.001 | Learning rate |
| `training.epochs` | 100 | Max epochs |

---

## Troubleshooting

### Lỗi thường gặp

#### 1. CUDA out of memory
```yaml
# Đổi device sang cpu trong config.yaml
training:
  device: "cpu"
```

#### 2. Dataset download lỗi 404
```bash
# Download thủ công từ SNAP Stanford
# Đặt file vào data/raw/<dataset_name>/
```

#### 3. Module not found
```bash
pip install -r requirements.txt
```

### Performance Tips

1. **GPU Training**: Đảm bảo CUDA available
   ```python
   torch.cuda.is_available()  # Kiểm tra
   ```

2. **Large Datasets**: Giảm batch size hoặc num_samples trong code

3. **Memory Optimization**: Sử dụng `--device cpu` nếu GPU RAM không đủ

---

## License

Project này được phát triển cho mục đích nghiên cứu.

## Citation

```bibtex
@article{gcn_ma_2023,
  title={Dynamic network link prediction with node representation learning from graph convolutional networks},
  journal={Scientific Reports},
  year={2023}
}
```

---

## Liên hệ / Hỗ trợ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trong repository.
