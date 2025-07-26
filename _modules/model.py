"""
model.py - ChemBERTa 기반 Multi-Task 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, r2_score
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class ChemBERTaMultiTask(nn.Module):
    """ChemBERTa 기반 Multi-Task Learning 모델"""
    
    def __init__(
        self,
        model_name: Optional[str], 
        filter_cols: List[str], 
        task_list: List[str], 
        task_types: Dict[str, str], 
        num_classes: Optional[Dict[str, int]] = None, 
        hidden_dim: int = 256, 
        data_type: str = 'normal'  # data_type 파라미터 추가
    ):
        """
        Args:
            model_name: ChemBERTa 모델 이름
            filter_cols: 범주형 특성 컬럼 목록
            task_list: 예측할 태스크 목록
            task_types: 각 태스크의 유형 (classification 또는 regression / multi_layer_regression)
            num_classes: 각 분류 태스크의 클래스 수 (dict)
            hidden_dim: 은닉층 차원
            use_attention: 어텐션 메커니즘 사용 여부
            data_type: 데이터 유형 (normal, didb_reduce 등)
        """
        super().__init__()
        
        self.model_name = model_name
        self.filter_cols = filter_cols
        self.task_list = task_list
        self.task_types = task_types
        self.hidden_dim = hidden_dim
        self.data_type = data_type  # data_type 저장
        
        # ChemBERTa 모델 설정
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.encoder_hidden = self.config.hidden_size

        # Attention_layer 출력차원  
        self.attention_layer = nn.Linear(self.encoder_hidden, self.hidden_dim)
        
        self.task_heads = nn.ModuleDict()
        self.reg_tasks = []

        for task in task_list:
            safe_task_name = task.replace('.', '__')
            if task_types[task] == 'classification':
                self.task_heads[safe_task_name] = nn.Linear(self.hidden_dim, num_classes[task] if num_classes else 2)
            elif task_types[task] == 'regression':
                self.reg_tasks.append(task)
                self.task_heads['merged'] = nn.Linear(self.hidden_dim, len(self.filter_cols))
            elif task_types[task] == 'multi_layer_regression':
                self.task_heads[safe_task_name] = nn.Linear(self.hidden_dim, 1)
            
    
    def forward(self, input_ids, attention_mask):
        """순전파"""
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask).pooler_output
        feature_merged = self.attention_layer(encoder_output)
            
        # 어텐션 없이 출력 계산
        task_outputs = {}
        if len(self.reg_tasks) > 0:
            reg_outputs = self.task_heads['merged'](feature_merged)
            task_outputs = {f'{task}': reg_outputs[:, i] for i, task in enumerate(self.reg_tasks)}

        for task in self.task_list:
            safe_task_name = task.replace('.', '__')
            if self.task_types[task] == 'classification':
                task_outputs[task] = self.task_heads[safe_task_name](feature_merged)
            elif self.task_types[task] == 'multi_layer_regression':
                task_outputs[task] = self.task_heads[safe_task_name](feature_merged).squeeze(-1)

        return task_outputs


class ChemBERTaMultiTaskLightning(pl.LightningModule):
    """PyTorch Lightning 기반 ChemBERTa Multi-Task Learning 모델"""
    
    def __init__(
        self,
        model_name: Optional[str],
        filter_cols: List[str],
        task_list: List[str],
        task_types: Dict[str, str],
        num_classes: Optional[Dict[str, int]] = None,
        hidden_dim: int = 256,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        task_weights: Optional[Dict[str, float]] = None,
        data_type: str = 'normal'  # data_type 파라미터 추가
    ):
        """
        Args:
            model_name: ChemBERTa 모델 이름
            filter_cols: 범주형 특성 컬럼 목록
            task_list: 예측할 태스크 목록
            task_types: 각 태스크의 유형 (classification 또는 regression)
            num_classes: 각 분류 태스크의 클래스 수 (dict)
            hidden_dim: 은닉층 차원
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            warmup_steps: 워밍업 스텝 수
            task_weights: 각 태스크의 손실 가중치 (dict)
            data_type: 데이터 유형 (normal, didb_reduce 등)
        """
        super().__init__()
        
        # PyTorch Lightning 하이퍼파라미터 저장
        self.save_hyperparameters()
        
        self.model = ChemBERTaMultiTask(
            model_name=model_name,
            filter_cols=filter_cols,
            task_list=task_list,
            task_types=task_types,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            data_type=data_type  # data_type 전달
        )
        
        self.task_list = task_list
        self.task_types = task_types
        self.task_weights = task_weights or {task: 1.0 for task in task_list}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # 평가 지표
        self.metrics = {}
        for task in task_list:
            if task_types[task] == 'classification':
                self.metrics[task] = {
                    'accuracy': accuracy_score,
                    'precision_recall_f1': lambda y_true, y_pred: precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division='0')[:3],
                    'auc': lambda y_true, y_pred_proba: roc_auc_score(
                        y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
                }
            else:
                self.metrics[task] = {
                    'mse': mean_squared_error,
                    'r2': r2_score
                }
    
    def forward(self, input_ids, attention_mask):
        """모델 순전파"""
        return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        """학습 단계"""
        labels = batch['labels']
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        task_outputs = self(input_ids, attention_mask)

        losses = {}
        for i, task in enumerate(self.task_list):
            task_labels = labels[:, i]
            mask = ~torch.isnan(task_labels)
            if mask.sum() > 0:
                valid_labels = task_labels[mask]
                valid_outputs = task_outputs[task][mask]
                if self.task_types[task] == 'classification':
                    task_loss = F.cross_entropy(valid_outputs, valid_labels.long())
                else:
                    task_loss = F.mse_loss(valid_outputs, valid_labels.float())
                losses[task] = task_loss

        if len(losses) > 0:
            total_loss = sum(losses.values())
            # PyTorch Lightning 로깅
            self.log('train_loss', total_loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
            for task, loss in losses.items():
                self.log(f'train_{task}_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
            return total_loss
        else:
            # 모든 태스크가 결측이면 None 반환(이 step에 대해 학습X)
            return None
    
    def validation_step(self, batch, batch_idx):
        """검증 단계"""
        labels = batch['labels']
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        task_outputs = self(input_ids, attention_mask)

        losses = {}
        predictions = {}

        for i, task in enumerate(self.task_list):
            task_labels = labels[:, i]
            mask = ~torch.isnan(task_labels)
            if mask.sum() > 0:
                valid_labels = task_labels[mask]
                valid_outputs = task_outputs[task][mask]
                if self.task_types[task] == 'classification':
                    task_loss = F.cross_entropy(valid_outputs, valid_labels.long())
                    task_preds = torch.argmax(valid_outputs, dim=1)
                    task_probs = F.softmax(valid_outputs, dim=1)
                    predictions[task] = {
                        'labels': valid_labels.detach().cpu(),
                        'preds': task_preds.detach().cpu(),
                        'probs': task_probs.detach().cpu()
                    }
                else:
                    task_loss = F.mse_loss(valid_outputs, valid_labels.float())
                    predictions[task] = {
                        'labels': valid_labels.detach().cpu(),
                        'preds': valid_outputs.detach().cpu()
                    }
                losses[task] = task_loss
            else:
                continue

        if len(losses) > 0:
            total_loss = sum(losses.values())
            # PyTorch Lightning 로깅
            self.log('val_loss', total_loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            for task, loss in losses.items():
                self.log(f'val_{task}_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        else:
            total_loss = None

        output = {'val_loss': total_loss, 'predictions': predictions}
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 처리"""
        # 모든 배치의 예측 결과 수집
        all_predictions = {task: {'labels': [], 'preds': [], 'probs': []} 
                        for task in self.task_list if self.task_types[task] == 'classification'}
        
        all_predictions.update({task: {'labels': [], 'preds': []} 
                            for task in self.task_list if self.task_types[task] == 'regression' or self.task_types[task] == 'multi_layer_regression'})
        
        for output in self.validation_step_outputs:
            for task in self.task_list:
                if task not in output['predictions']:
                    continue  # 해당 배치에 예측값이 없으면 건너뜀
                task_preds = output['predictions'][task]
                
                if self.task_types[task] == 'classification':
                    all_predictions[task]['labels'].extend(task_preds['labels'])
                    all_predictions[task]['preds'].extend(task_preds['preds'])
                    all_predictions[task]['probs'].extend(task_preds['probs'])
                else:
                    all_predictions[task]['labels'].extend(task_preds['labels'])
                    all_predictions[task]['preds'].extend(task_preds['preds'])
        
        # 각 태스크별 평가 지표 계산
        for task in self.task_list:
            task_preds = all_predictions[task]
            # 비어있거나, 텐서가 아닌 값이 들어있으면 metric 계산 건너뜀
            if not task_preds['labels'] or not task_preds['preds']:
                continue
            if not all(isinstance(x, torch.Tensor) for x in task_preds['labels']):
                continue
            if not all(isinstance(x, torch.Tensor) for x in task_preds['preds']):
                continue
            
            if self.task_types[task] == 'classification':
                # 분류 태스크 평가
                labels_np = torch.stack(task_preds['labels']).numpy()
                preds_np = torch.stack(task_preds['preds']).numpy()
                
                acc = self.metrics[task]['accuracy'](labels_np, preds_np)
                prec, rec, f1 = self.metrics[task]['precision_recall_f1'](labels_np, preds_np)
                
                # 로깅
                self.log(f'val_{task}_acc', acc, sync_dist=True)
                self.log(f'val_{task}_precision', prec, sync_dist=True)
                self.log(f'val_{task}_recall', rec, sync_dist=True)
                self.log(f'val_{task}_f1', f1, sync_dist=True)
                
                # AUC 계산 (가능한 경우)
                try:
                    probs = torch.stack(task_preds['probs']).numpy()
                    if probs.shape[1] >= 2:  # 이진 분류 이상인 경우
                        auc = self.metrics[task]['auc'](labels_np, probs)
                        self.log(f'val_{task}_auc', auc, sync_dist=True)
                except Exception as e:
                    print(f"AUC 계산 중 오류 발생: {e}")
            else:
                # 회귀 태스크 평가
                labels_np = torch.stack(task_preds['labels']).numpy()
                preds_np = torch.stack(task_preds['preds']).numpy()
                # === NaN 마스킹 추가 ===
                valid_mask = ~np.isnan(labels_np)
                labels_np_valid = labels_np[valid_mask]
                preds_np_valid = preds_np[valid_mask]
                if len(labels_np_valid) == 0:  # 모두 결측인 경우
                    continue
                mse = self.metrics[task]['mse'](labels_np_valid, preds_np_valid)
                r2 = self.metrics[task]['r2'](labels_np_valid, preds_np_valid)
                # 로깅
                self.log(f'val_{task}_mse', mse, sync_dist=True)
                self.log(f'val_{task}_r2', r2, sync_dist=True)
        
        # 단계가 끝나면 출력 목록 초기화
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        task_outputs = self(input_ids, attention_mask)

        losses = {}
        predictions = {}

        for i, task in enumerate(self.task_list):
            task_labels = labels[:, i]
            mask = ~torch.isnan(task_labels)
            if mask.sum() > 0:
                valid_labels = task_labels[mask]
                valid_outputs = task_outputs[task][mask]
                if self.task_types[task] == 'classification':
                    task_loss = F.cross_entropy(valid_outputs, valid_labels.long())
                    task_preds = torch.argmax(valid_outputs, dim=1)
                    task_probs = F.softmax(valid_outputs, dim=1)
                    predictions[task] = {
                        'labels': valid_labels.detach().cpu(),
                        'preds': task_preds.detach().cpu(),
                        'probs': task_probs.detach().cpu()
                    }
                else:
                    task_loss = F.mse_loss(valid_outputs, valid_labels.float())
                    predictions[task] = {
                        'labels': valid_labels.detach().cpu(),
                        'preds': valid_outputs.detach().cpu()
                    }
                losses[task] = task_loss
            else:
                continue

        if len(losses) > 0:
            total_loss = sum(losses.values())
            # PyTorch Lightning 로깅 (WandB 호환)
            self.log('test_loss', total_loss, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True)
            for task, loss in losses.items():
                self.log(f'test_{task}_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        else:
            total_loss = None

        output = {'test_loss': total_loss, 'predictions': predictions}
        self.test_step_outputs.append(output)
        return output

    def test_step_end(self, outputs):
        """테스트 단계 종료 시 처리 (WandB 호환)"""
        # step별 메트릭 계산 및 로깅
        if outputs['test_loss'] is not None:
            # step별 loss 로깅
            self.log('test_step_loss', outputs['test_loss'], on_step=True, on_epoch=False)
            
            # 각 task별 step별 loss 로깅
            for task in self.task_list:
                if task in outputs['predictions']:
                    task_preds = outputs['predictions'][task]
                    if self.task_types[task] == 'classification':
                        # 분류 메트릭 계산
                        labels = task_preds['labels']
                        preds = task_preds['preds']
                        if len(labels) > 0 and len(preds) > 0:
                            acc = (labels == preds).float().mean()
                            self.log(f'test_{task}_step_acc', acc, on_step=True, on_epoch=False)
                    else:
                        # 회귀 메트릭 계산
                        labels = task_preds['labels']
                        preds = task_preds['preds']
                        if len(labels) > 0 and len(preds) > 0:
                            mse = F.mse_loss(preds, labels)
                            self.log(f'test_{task}_step_mse', mse, on_step=True, on_epoch=False)
        
        return outputs
    
    def on_test_epoch_end(self):
        """테스트 에포크 종료 시 처리 (검증과 유사)"""
        # 모든 배치의 예측 결과 수집
        all_predictions = {task: {'labels': [], 'preds': [], 'probs': []} 
                        for task in self.task_list if self.task_types[task] == 'classification'}
        
        all_predictions.update({task: {'labels': [], 'preds': []} 
                            for task in self.task_list if self.task_types[task] == 'regression' or self.task_types[task] == 'multi_layer_regression'})
        
        for output in self.test_step_outputs:
            for task in self.task_list:
                if task not in output['predictions']:
                    continue  # 해당 배치에 예측값이 없으면 건너뜀
                task_preds = output['predictions'][task]
                
                if self.task_types[task] == 'classification':
                    all_predictions[task]['labels'].extend(task_preds['labels'])
                    all_predictions[task]['preds'].extend(task_preds['preds'])
                    all_predictions[task]['probs'].extend(task_preds['probs'])
                else:
                    all_predictions[task]['labels'].extend(task_preds['labels'])
                    all_predictions[task]['preds'].extend(task_preds['preds'])
        
        # 각 태스크별 평가 지표 계산 및 저장
        results = {}
        for task in self.task_list:
            task_preds = all_predictions[task]
            # 비어있거나, 텐서가 아닌 값이 들어있으면 metric 계산 건너뜀
            if not task_preds['labels'] or not task_preds['preds']:
                continue
            if not all(isinstance(x, torch.Tensor) for x in task_preds['labels']):
                continue
            if not all(isinstance(x, torch.Tensor) for x in task_preds['preds']):
                continue
            
            if self.task_types[task] == 'classification':
                # 분류 태스크 평가
                labels_np = torch.stack(task_preds['labels']).numpy()
                preds_np = torch.stack(task_preds['preds']).numpy()
                
                acc = self.metrics[task]['accuracy'](labels_np, preds_np)
                prec, rec, f1 = self.metrics[task]['precision_recall_f1'](labels_np, preds_np)
                
                # 결과 저장
                results[f'{task}_acc'] = acc
                results[f'{task}_precision'] = prec
                results[f'{task}_recall'] = rec
                results[f'{task}_f1'] = f1
                
                # 로깅
                self.log(f'test_{task}_acc', acc, sync_dist=True)
                self.log(f'test_{task}_precision', prec, sync_dist=True)
                self.log(f'test_{task}_recall', rec, sync_dist=True)
                self.log(f'test_{task}_f1', f1, sync_dist=True)
            else:
                # 회귀 태스크 평가
                labels_np = torch.stack(task_preds['labels']).numpy()
                preds_np = torch.stack(task_preds['preds']).numpy()
                # === NaN 마스킹 추가 ===
                valid_mask = ~np.isnan(labels_np)
                labels_np_valid = labels_np[valid_mask]
                preds_np_valid = preds_np[valid_mask]
                if len(labels_np_valid) == 0:  # 모두 결측인 경우
                    continue
                mse = self.metrics[task]['mse'](labels_np_valid, preds_np_valid)
                r2 = self.metrics[task]['r2'](labels_np_valid, preds_np_valid)
                # 수정: test metric만 로깅
                self.log(f'test_{task}_mse', mse, sync_dist=True)
                self.log(f'test_{task}_r2', r2, sync_dist=True)
        
        # 단계가 끝나면 출력 목록 초기화
        self.test_step_outputs.clear()
        
        # 전체 결과 반환
        self.log_dict(results, sync_dist=True)
        return results
    
    def configure_optimizers(self):
        """옵티마이저 설정"""
        # 가중치 감쇠 제외 파라미터 설정
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # 스케줄러 설정 (워밍업 포함)
        # 총 스텝 수 계산 (대략적으로 추정)
        train_batches = 1000  # 기본값으로 설정
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 1
        num_training_steps = max_epochs * train_batches // num_devices
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }