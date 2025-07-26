# trainer.py - 모델 훈련 및 평가 로직 (모듈화 및 깔끔하게 재정리)
import os, json
import numpy as np
import shutil
from typing import Dict, Optional, List

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from _modules.model import ChemBERTaMultiTaskLightning
from _modules.dataset import ChemMultiTaskDataModule
from _modules.utils import get_task_list, DIDB_FILTER_COLS


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
                 missing_label_strategy: str = 'any',
                 hidden_dim: int = 256,
                 task_weights: Optional[Dict[str, float]] = None,
                 early_stopping_patience: int = 10,
                 data_type: str = "admet"):

        self.model_name = model_name
        self.data_folder = data_folder
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.task_type = task_type
        self.missing_label_strategy = missing_label_strategy
        self.hidden_dim = hidden_dim
        self.early_stopping_patience = early_stopping_patience
        self.data_type = data_type

        self.task_list = get_task_list(task_type)
        # 태스크별 유형을 지정 (classification, regression, multi_reg 지원)
        if task_type == 'cls':
            self.task_types = {task: 'classification' for task in self.task_list}
        elif task_type == 'reg':
            self.task_types = {task: 'regression' for task in self.task_list}
        elif task_type == 'multi_reg':
            self.task_types = {task: 'multi_layer_regression' for task in self.task_list}
        else:
            raise ValueError(f"알 수 없는 task_type: {task_type}")
        
        self.task_weights = task_weights or {task: 1.0 for task in self.task_list}

        self.model = None
        self.data_module = None
        self.trainer = None

    def setup(self, use_wandb=False, wandb_project=None, wandb_entity=None):
        # Load dataset
        self.data_module = ChemMultiTaskDataModule(
            data_folder=self.data_folder,
            batch_size=self.batch_size,
            scaling=True,
            task_type=self.task_type,
            model_name=self.model_name,
            missing_label_strategy=self.missing_label_strategy,
            data_type=self.data_type
        )
        self.data_module.setup()

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
        
        filter_cols = DIDB_FILTER_COLS
        
        self.model = ChemBERTaMultiTaskLightning(
            model_name=self.model_name,
            filter_cols=filter_cols,
            task_list=self.task_list,
            task_types=self.task_types,
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            task_weights=self.task_weights
        )

        if os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Logger 선택
        if use_wandb:
            # WandB Logger 사용
            logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity
            )
            print(f"WandBLogger 초기화됨 - 프로젝트: {wandb_project}")
        else:
            # TensorBoard Logger 사용
            logger = TensorBoardLogger(save_dir=self.output_dir, name="logs")
            print("TensorBoardLogger 초기화됨")

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
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")
        if train_loader is None or val_loader is None:
            raise ValueError("학습 또는 검증 데이터로더가 없습니다.")
        
        self.trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        if test_loader is not None:
            return self.trainer.test(self.model, dataloaders=test_loader)
        else:
            print("테스트 데이터로더가 없습니다.")
            return None

    def save_model(self, path=None):
        path = path or os.path.join(self.output_dir, "final_model")
        os.makedirs(path, exist_ok=True)

        best_ckpt = None
        for cb in getattr(self.trainer, "callbacks", []):
            if hasattr(cb, "best_model_path"):
                best_ckpt = cb.best_model_path
                break
            
        if best_ckpt and os.path.exists(best_ckpt):
            shutil.copy(best_ckpt, os.path.join(path, "best_model.ckpt"))

        torch.save(self.model.state_dict(), os.path.join(path, "model_weights.pt"))
        if hasattr(self.model, 'hparams') and hasattr(self.model.hparams, 'to_dict'):
            import json
            with open(os.path.join(path, "model_config.json"), 'w') as f:
                json.dump(dict(self.model.hparams), f)

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

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                enc = tokenizer(batch, padding='max_length', max_length=510, truncation=True, return_tensors='pt')
                input_ids, attention_mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)

                outputs = self.model(input_ids, attention_mask)

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
