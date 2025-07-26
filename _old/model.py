"""
model.py - ChemBERTa 기반 Multi-Task 모델 정의
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
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
        use_attention: bool = True,
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
        self.use_attention = use_attention
        self.data_type = data_type  # data_type 저장
        
        # ChemBERTa 모델 설정
        if model_name:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            self.encoder_hidden = self.config.hidden_size
        else:
            self.encoder = None
            self.encoder_hidden = 0

        self.cat_layers = nn.ModuleDict({
                str(k): nn.Embedding(2, hidden_dim) for k in range(len(filter_cols))
            })

        self.int_layer = nn.Linear(5, hidden_dim)  # INT_COLS 길이

        self.float_layer = nn.Linear(4, hidden_dim)   # FLOAT_COLS 길이

        # didb_reduce 모드가 아닐 때만 범주형 특성 임베딩 사용
        if self.data_type != 'didb_reduce':
            # 특성 통합
            feature_dim = len(filter_cols) * hidden_dim + 2 * hidden_dim + self.encoder_hidden
            self.attention_layer = nn.Linear(feature_dim, hidden_dim)
        else:
            # Attention_layer 출력차원  
            self.attention_layer = nn.Linear(self.encoder_hidden, hidden_dim)
            feature_dim = hidden_dim
            

        if self.use_attention:
            # 어텐션 메커니즘 설정
            self.attention_hidden_layer = nn.Linear(hidden_dim + len(filter_cols) + 2 + (1 if model_name else 0), hidden_dim)
            
            # 태스크별 출력 레이어
            self.task_heads = nn.ModuleDict()
            self.reg_tasks = []

            for task in task_list:
                # 모듈 이름에 점(.)이 포함되면 안되므로 대체
                safe_task_name = task.replace('.', '__')
                
                if task_types[task] == 'classification':
                    self.task_heads[safe_task_name] = nn.Linear(hidden_dim, num_classes[task] if num_classes else 2)
                elif task_types[task] == 'regression':
                    self.reg_tasks.append(task)
                    self.task_heads['merged'] = nn.Linear(hidden_dim, len(self.filter_cols))
                elif task_types[task] == 'multi_layer_regression':
                    self.task_heads[safe_task_name] = nn.Linear(hidden_dim, 1)

        else:
            self.task_heads = nn.ModuleDict()
            self.reg_tasks = []

            for task in task_list:
                safe_task_name = task.replace('.', '__')
                if task_types[task] == 'classification':
                    self.task_heads[safe_task_name] = nn.Linear(feature_dim, num_classes[task] if num_classes else 2)
                elif task_types[task] == 'regression':
                    self.reg_tasks.append(task)
                    self.task_heads['merged'] = nn.Linear(feature_dim, len(self.filter_cols))
                elif task_types[task] == 'multi_layer_regression':
                    self.task_heads[safe_task_name] = nn.Linear(feature_dim, 1)
    
    def forward(self, features, input_ids=None, attention_mask=None):
        """순전파"""
        if self.data_type != 'didb_reduce':
            # 기존 처리 방식
            # 범주형 특성 처리
            cat_embs = []
            for i, k in enumerate(range(len(self.filter_cols))):
                if features and i < len(features):
                    cat_embs.append(self.cat_layers[str(k)](features[i]))
                else:
                    device = input_ids.device if input_ids is not None else torch.device('cpu')
                    batch_size = input_ids.size(0) if input_ids is not None else 1
                    cat_embs.append(torch.zeros((batch_size, self.hidden_dim), device=device))
            
            # 정수형 특성 처리
            if features and len(features) > len(self.filter_cols):
                int_value = self.int_layer(features[len(self.filter_cols)].float())
                float_value = self.float_layer(features[len(self.filter_cols)+1].float())
            else:
                device = input_ids.device if input_ids is not None else torch.device('cpu')
                batch_size = input_ids.size(0) if input_ids is not None else 1
                int_value = torch.zeros((batch_size, self.hidden_dim), device=device)

            # 실수형 특성 처리
            if features and len(features) > len(self.filter_cols) + 1:
                float_value = self.float_layer(features[len(self.filter_cols) + 1].float())
            else:
                device = input_ids.device if input_ids is not None else torch.device('cpu')
                batch_size = input_ids.size(0) if input_ids is not None else 1
                float_value = torch.zeros((batch_size, self.hidden_dim), device=device)
            
            # ChemBERTa 인코더 출력 (있는 경우)
            if self.encoder and input_ids is not None:
                encoder_output = self.encoder(input_ids, attention_mask=attention_mask).pooler_output
                all_features = cat_embs + [int_value, float_value, encoder_output]
            else:
                all_features = cat_embs + [int_value, float_value]

            # 모든 특성 연결
            if cat_embs:              # 필터 특성이 있을 때만 concat
                cat_values = torch.cat(cat_embs, dim=-1)
            else:                      # 필터 컬럼이 0개인 data_type='none' 설정
                batch_size = input_ids.size(0)
                device = input_ids.device
                cat_values = torch.zeros((batch_size, 0), device=device)   # 길이 0 텐서
            
            if self.encoder and input_ids is not None:
                feature_merged = torch.cat([cat_values, int_value, float_value, encoder_output], dim=-1)
            else:
                feature_merged = torch.cat([cat_values, int_value, float_value], dim=-1)
                
        else:
            # didb_reduce 모드에서는 float 특성만 처리
            encoder_output = self.encoder(input_ids, attention_mask=attention_mask).pooler_output
            feature_merged = self.attention_layer(encoder_output)
        
        # 어텐션 메커니즘 적용 (사용하는 경우)
        if self.use_attention:
            # 어텐션 계산
            attention_hidden = self.attention_layer(feature_merged)
            
            # 특성 스택 생성
            if self.encoder and input_ids is not None:
                feature_stack = torch.stack(cat_embs + [int_value, float_value, feature_merged], dim=1)
            else:
                feature_stack = torch.stack(cat_embs + [int_value, float_value], dim=1)
            
            # 어텐션 스코어 계산
            attention_scores = torch.bmm(feature_stack, attention_hidden.unsqueeze(dim=2))
            
            # 소프트맥스 적용
            num_features = len(self.filter_cols) + 2 + (1 if self.encoder and input_ids is not None else 0)
            attention_softmax = F.softmax(attention_scores.reshape(-1, 1, num_features), dim=-1)
            
            # 컨텍스트 벡터 계산
            context_vec = torch.bmm(attention_softmax, feature_stack)
            context_vec = context_vec.squeeze(1)
            
            # 최종 은닉 상태 계산
            combined_hidden = torch.cat([attention_hidden, context_vec], dim=-1)
            final_hidden = F.tanh(self.attention_hidden_layer(combined_hidden))
            
            # 태스크별 출력 계산
            task_outputs = {}
            for task in self.task_list:
                safe_task_name = task.replace('.', '__')
                if self.task_types[task] == 'classification':
                    task_outputs[task] = self.task_heads[safe_task_name](final_hidden)
                else:
                    task_outputs[task] = self.task_heads[safe_task_name](final_hidden).squeeze(-1)
            
            return task_outputs, attention_scores.squeeze(dim=-1)
        
        else:
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

            return task_outputs, None


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
        use_attention: bool = True,
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
            use_attention: 어텐션 메커니즘 사용 여부
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
            use_attention=use_attention,
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
                        y_true, y_pred, average='binary', zero_division=0)[:3],
                    'auc': lambda y_true, y_pred_proba: roc_auc_score(
                        y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
                }
            else:
                self.metrics[task] = {
                    'mse': mean_squared_error,
                    'r2': r2_score
                }
    
    def forward(self, features, input_ids=None, attention_mask=None):
        """모델 순전파"""
        return self.model(features, input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        """훈련 단계"""
        # 입력 데이터 준비
        features = batch['features']
        labels = batch['labels']
        
        if 'input_ids' in batch and 'attention_mask' in batch:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            task_outputs, _ = self(features, input_ids, attention_mask)
        else:
            task_outputs, _ = self(features)
        
        # 손실 계산
        losses = {}
        for i, task in enumerate(self.task_list):
            task_labels = labels[:, i]
            
            if self.task_types[task] == 'classification':
                # 분류 태스크
                task_loss = F.cross_entropy(task_outputs[task], task_labels.long())
            else:
                # 회귀 태스크
                task_loss = F.mse_loss(task_outputs[task], task_labels.float())
            
            losses[task] = task_loss * self.task_weights[task]
        
        # 전체 손실 계산
        total_loss = sum(losses.values())
        
        # 로깅
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True)
        for task, loss in losses.items():
            self.log(f'train_{task}_loss', loss, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """검증 단계"""
        # 입력 데이터 준비
        features = batch['features']
        labels = batch['labels']
        
        if 'input_ids' in batch and 'attention_mask' in batch:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            task_outputs, _ = self(features, input_ids, attention_mask)
        else:
            task_outputs, _ = self(features)
        
        # 손실 및 예측 계산
        losses = {}
        predictions = {}
        
        for i, task in enumerate(self.task_list):
            task_labels = labels[:, i]
            
            if self.task_types[task] == 'classification':
                # 분류 태스크
                task_loss = F.cross_entropy(task_outputs[task], task_labels.long())
                task_preds = torch.argmax(task_outputs[task], dim=1)
                task_probs = F.softmax(task_outputs[task], dim=1)
                predictions[task] = {
                    'labels': task_labels.detach().cpu(),
                    'preds': task_preds.detach().cpu(),
                    'probs': task_probs.detach().cpu()
                }
            else:
                # 회귀 태스크
                task_loss = F.mse_loss(task_outputs[task], task_labels.float())
                predictions[task] = {
                    'labels': task_labels.detach().cpu(),
                    'preds': task_outputs[task].detach().cpu()
                }
            
            losses[task] = task_loss
        
        # 전체 손실 계산
        total_loss = sum(losses.values())
        
        # 로깅
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
        for task, loss in losses.items():
            self.log(f'val_{task}_loss', loss, sync_dist=True)
        
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
                
                mse = self.metrics[task]['mse'](labels_np, preds_np)
                r2 = self.metrics[task]['r2'](labels_np, preds_np)
                
                # 로깅
                self.log(f'val_{task}_mse', mse, sync_dist=True)
                self.log(f'val_{task}_r2', r2, sync_dist=True)
        
        # 단계가 끝나면 출력 목록 초기화
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """테스트 단계 (검증 단계와 유사)"""
        output = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(output)
        return output
    
    def on_test_epoch_end(self):
        """테스트 에포크 종료 시 처리 (검증과 유사)"""
        # 모든 배치의 예측 결과 수집
        all_predictions = {task: {'labels': [], 'preds': [], 'probs': []} 
                        for task in self.task_list if self.task_types[task] == 'classification'}
        
        all_predictions.update({task: {'labels': [], 'preds': []} 
                            for task in self.task_list if self.task_types[task] == 'regression' or self.task_types[task] == 'multi_layer_regression'})
        
        for output in self.test_step_outputs:
            for task in self.task_list:
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
                
                mse = self.metrics[task]['mse'](labels_np, preds_np)
                r2 = self.metrics[task]['r2'](labels_np, preds_np)
                
                # 결과 저장
                results[f'{task}_mse'] = mse
                results[f'{task}_r2'] = r2
                
                # 로깅
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
        num_training_steps = self.trainer.max_epochs * train_batches // num_devices
        
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