# trainer.py - 모델 훈련 및 평가 로직 (모듈화 및 깔끔하게 재정리)
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Optional, List
from model import ChemBERTaMultiTaskLightning
from dataset import ChemMultiTaskDataModule
from utils import get_task_list, NORMAL_FILTER_COLS, REDUCE_FILTER_COLS, INT_COLS, FLOAT_COLS
from transformers import AutoTokenizer
import numpy as np
import shutil


class ChemBERTaTrainer:
    def __init__(self,
                 model_name: str,
                 data_folder: str,
                 output_dir: str = "./results",
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 epochs: int = 10,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 500,
                 task_type: str = 'cls',
                 data_type: str = 'none',
                 use_attention: bool = True,
                 hidden_dim: int = 256,
                 task_weights: Optional[Dict[str, float]] = None,
                 early_stopping_patience: int = 10):

        self.model_name = model_name
        self.data_folder = data_folder
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.task_type = task_type
        self.data_type = data_type
        # self.use_attention = use_attention
        self.use_attention = False if data_type == 'none' else use_attention
        self.hidden_dim = hidden_dim
        self.early_stopping_patience = early_stopping_patience

        self.task_list = get_task_list(task_type, data_type)
        self.task_types = {task: 'classification' if task_type == 'cls' else 'regression' for task in self.task_list}
        self.task_weights = task_weights or {task: 1.0 for task in self.task_list}

        self.model = None
        self.data_module = None
        self.trainer = None

    def setup(self):
        # Load dataset
        self.data_module = ChemMultiTaskDataModule(
            data_folder=self.data_folder,
            batch_size=self.batch_size,
            scaling=True,
            task_type=self.task_type,
            model_name=self.model_name,
            data_type=self.data_type
        )
        self.data_module.setup()

        import json

        vocab_dir = os.path.join(self.output_dir, "vocab")
        os.makedirs(vocab_dir, exist_ok=True)
        for task, vocab in self.data_module.all_vocabs.items():
            with open(os.path.join(vocab_dir, f"{task}.json"), "w") as f:
                json.dump(vocab, f)

        num_classes = None
        if self.task_type == 'cls':
            num_classes = {
                task: len(self.data_module.all_vocabs.get(task, [0, 1]))
                for task in self.task_list
            }
        
        if self.data_type == 'normal':
            filter_cols = NORMAL_FILTER_COLS
        elif self.data_type == 'reduce':
            filter_cols = REDUCE_FILTER_COLS
        elif self.data_type == 'none':
            filter_cols = []
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")

        if self.data_type == 'none':
            self.use_attention = False

        self.model = ChemBERTaMultiTaskLightning(
            model_name=self.model_name,
            filter_cols=filter_cols,
            task_list=self.task_list,
            task_types=self.task_types,
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            use_attention=self.use_attention,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            task_weights=self.task_weights
        )

        logger = TensorBoardLogger(save_dir=self.output_dir, name="logs")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, mode='min'),
            ModelCheckpoint(monitor='val_loss', dirpath=self.output_dir,
                            filename='best-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min'),
            LearningRateMonitor(logging_interval='step')
        ]

        strategy = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else 'auto'

        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=self.output_dir,
            accelerator='auto',
            devices='auto',
            strategy=strategy,
            log_every_n_steps=10,
            accumulate_grad_batches=2,
            gradient_clip_val=1.0,
            precision='16-mixed'
        )

    def train(self):
        if self.data_module is None or self.model is None:
            self.setup()

        train_loader, val_loader, test_loader = self.data_module.get_dataloaders()
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return self.trainer.test(self.model, dataloaders=test_loader)

    def save_model(self, path=None):
        path = path or os.path.join(self.output_dir, "final_model")
        os.makedirs(path, exist_ok=True)

        best_ckpt = self.trainer.checkpoint_callback.best_model_path if hasattr(self.trainer, 'checkpoint_callback') else None
        if best_ckpt and os.path.exists(best_ckpt):
            shutil.copy(best_ckpt, os.path.join(path, "best_model.ckpt"))

        torch.save(self.model.state_dict(), os.path.join(path, "model_weights.pt"))
        if hasattr(self.model, 'hparams') and hasattr(self.model.hparams, 'to_dict'):
            import json
            with open(os.path.join(path, "model_config.json"), 'w') as f:
                json.dump(self.model.hparams.to_dict(), f)

    def load_model(self, path):
        if self.model is None:
            self.setup()

        import json
        vocab_dir = os.path.join(path, "vocab")
        if os.path.exists(vocab_dir):
            for file in os.listdir(vocab_dir):
                if file.endswith(".json"):
                    task = file.replace(".json", "")
                    with open(os.path.join(vocab_dir, file), "r") as f:
                        self.data_module.all_vocabs[task] = json.load(f)

        ckpt_path = os.path.join(path, "best_model.ckpt")
        weights_path = os.path.join(path, "model_weights.pt")

        if os.path.exists(ckpt_path):
            self.model = ChemBERTaMultiTaskLightning.load_from_checkpoint(ckpt_path)
        elif os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        else:
            print("모델 파일이 존재하지 않습니다.")
        return self.model

    def predict(self, smiles_list):
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        self.model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        batch_size = min(len(smiles_list), 16)
        predictions = []

        # filter_cols = NORMAL_FILTER_COLS if self.data_type == 'normal' else REDUCE_FILTER_COLS

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                enc = tokenizer(batch, padding='max_length', max_length=510, truncation=True, return_tensors='pt')
                input_ids, attention_mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)

                # cat_feats = [torch.zeros((len(batch), 1), dtype=torch.long).to(device) for _ in filter_cols]
                # int_feats = torch.zeros((len(batch), len(INT_COLS)), dtype=torch.float).to(device)
                # float_feats = torch.zeros((len(batch), len(FLOAT_COLS)), dtype=torch.float).to(device)
                # features = cat_feats + [int_feats, float_feats]

                if self.data_type == 'normal':
                    filter_cols = NORMAL_FILTER_COLS
                elif self.data_type == 'reduce':
                    filter_cols = REDUCE_FILTER_COLS
                else:  # 'none'
                    filter_cols = []
                
                if self.data_type == 'none':
                    features = []  # SMILES-only
                else:
                    cat_feats = [torch.zeros((len(batch), 1), dtype=torch.long).to(device) for _ in filter_cols]
                    int_feats = torch.zeros((len(batch), len(INT_COLS)), dtype=torch.float).to(device)
                    float_feats = torch.zeros((len(batch), len(FLOAT_COLS)), dtype=torch.float).to(device)
                    features = cat_feats + [int_feats, float_feats]

                outputs, _ = self.model(features, input_ids, attention_mask)

                batch_preds = {}
                for task in self.task_list:
                    if self.task_types[task] == 'classification':
                        probs = torch.nn.functional.softmax(outputs[task], dim=1)
                        preds = torch.argmax(probs, dim=1)
                        batch_preds[task] = {
                            'probabilities': probs.cpu().numpy(),
                            'predictions': preds.cpu().numpy()
                        }
                    else:
                        batch_preds[task] = outputs[task].cpu().numpy()
                predictions.append(batch_preds)

        merged = {}
        for task in self.task_list:
            if self.task_types[task] == 'classification':
                merged[task] = {
                    'probabilities': np.concatenate([p[task]['probabilities'] for p in predictions]),
                    'predictions': np.concatenate([p[task]['predictions'] for p in predictions])
                }
            else:
                merged[task] = np.concatenate([p[task] for p in predictions])

        return merged
