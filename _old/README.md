# [K-Melloddy] ADMET Prediction Model Prototype

Multi-task learning model for ADMET property prediction based on DrugBank data, using ChemBERTa as encoder.  
Supports both centralized training and federated learning simulation (FedAvg) for research purposes.

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
#### Step 1. Train each client model
```bash
python run.py --data_dir data/client1 --output_dir results/client1
python run.py --data_dir data/client2 --output_dir results/client2
python run.py --data_dir data/client3 --output_dir results/client3
```

#### Step 2. Merge models using FedAvg
```bash
python fedavg_merge.py \
    --input_dirs results/client1 results/client2 results/client3 \
    --output_dir results/fedavg_merge
```
#### Step 3. Evaluate the merged model
```bash
python fedavg_test.py \
    --data_dir data/ \
    --model_path results/fedavg_merge/model_weights.pt
```

## ✍️ Author

Developed by [Goun Pyeon / ISoftLab / Chungnam National University]