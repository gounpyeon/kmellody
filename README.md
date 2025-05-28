# [K-Melloddy] ADMET Prediction Model Prototype

Multi-task learning model for ADMET property prediction based on DrugBank data, using ChemBERTa as encoder.  
Supports both centralized training and federated learning simulation (FedAvg) for research purposes.

---

## 📂 Directory Structure
```
.
├── data/                # DrugBank-derived data split into clients (for federated sim)
├── model.py             # ChemBERTa-based multi-task model
├── trainer.py           # PyTorch Lightning training module
├── dataset.py           # Dataset & DataModule for multi-task SMILES
├── utils.py             # Helper functions (e.g., scaling, metrics)
├── run.py               # Centralized training script
├── fedavg_merge.py      # Aggregates weights from client models (FedAvg)
├── fedavg_test.py       # Evaluates merged model
├── results/             # Checkpoints, logs, and test results
└── requirements.txt     # Python dependencies
```
---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Centralized Training
```bash
python run.py
```

### 3. Federated Simulation (Optional)
```bash
python fedavg_merge.py     # Merge client checkpoints (FedAvg)
python fedavg_test.py      # Evaluate the averaged model
```

## ✍️ Author

Developed by [Goun Pyeon / ISoftLab / Chungnam National University]