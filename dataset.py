"""
dataset.py - 데이터셋 클래스와 데이터 처리 로직
"""
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from utils import (
    NORMAL_FILTER_COLS, REDUCE_FILTER_COLS,
    INT_COLS, FLOAT_COLS,
    standard_scaling, load_vocabs, prepare_reduced_dataset, get_task_list
)


class ChemMultiTaskDataset(Dataset):
    """ChemBERTa 기반 Multi-Task Learning을 위한 데이터셋"""
    
    def __init__(
        self, 
        df_dataset: pd.DataFrame, 
        all_vocabs: Dict[str, Dict[str, int]], 
        model_name: Optional[str] = None, 
        task_type: str = 'cls', 
        data_type: str = 'normal'
    ):
        """
        Args:
            df_dataset: 입력 데이터프레임
            all_vocabs: 분류 태스크를 위한 레이블 매핑 사전
            model_name: 사용할 transformer 모델 이름 (ChemBERTa)
            task_type: 'cls' 또는 'reg' (분류 또는 회귀)
            data_type: 'normal' 또는 'reduce' 또는 'none' (필터 컬럼 이용 여부)
        """
        self.model_name = model_name
        self.task_type = task_type
        self.data_type = data_type
        
        # 데이터 유형에 따른 컬럼 설정
        if data_type == 'normal':   # normal
            self.filter_cols = NORMAL_FILTER_COLS
        elif data_type == 'reduce': # reduce
            self.filter_cols = REDUCE_FILTER_COLS
        elif data_type == 'none':
            self.filter_cols = []
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # 태스크 목록 설정
        self.y_cols = get_task_list(task_type, data_type)            
        self.all_vocabs = all_vocabs
        
        # 데이터 저장
        self.df = df_dataset.copy()
        
        # Transformer 입력 준비
        if model_name is not None:
            self.input_ids = []
            self.attention_masks = []
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # SMILES 문자열 토큰화
            print("토큰화 진행 중...")
            for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
                objs = self.tokenizer(row['smiles'], padding='max_length', max_length=510, truncation=True)
                self.input_ids.append(objs['input_ids'])
                self.attention_masks.append(objs['attention_mask'])
    
    def __getitem__(self, idx):
        """데이터셋에서 idx번째 항목 반환"""
        row = self.df.iloc[idx]
        
        # 범주형 특성 처리
        cat_features = []
        for col in self.filter_cols:
            cat_features.append(torch.tensor([int(row[col])], dtype=torch.long))
        
        # 정수형 특성 처리
        int_features = torch.tensor([row[col] for col in INT_COLS], dtype=torch.float32).unsqueeze(0)
        
        # 실수형 특성 처리
        float_features = torch.tensor([row[col] for col in FLOAT_COLS], dtype=torch.float32).unsqueeze(0)
        
        # 출력 레이블 처리
        if self.task_type == 'cls':
            labels = []
            for target in self.y_cols:
                cls_value = str(row[target]).lower()
                if target in self.all_vocabs and cls_value in self.all_vocabs[target]:
                    label_id = self.all_vocabs[target][cls_value]
                else:
                    # 어휘에 없는 경우 기본값 사용
                    label_id = 0  # 기본값으로 0 사용
                labels.append(label_id)
        else:  # reg
            labels = [row[target] for target in self.y_cols]
        
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # 결과 반환
        result = {
            'features': cat_features + [int_features, float_features],
            'labels': labels
        }
        
        # Transformer 입력이 있는 경우
        if self.model_name is not None:
            result['input_ids'] = torch.tensor(self.input_ids[idx], dtype=torch.long)
            result['attention_mask'] = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        
        return result
    
    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.df)


class ChemMultiTaskDataModule:
    """ChemBERTa Multi-Task Learning을 위한 데이터 모듈"""
    
    def __init__(
        self, 
        data_folder: str, 
        batch_size: int = 32, 
        scaling: bool = True, 
        task_type: str = 'cls', 
        model_name: Optional[str] = None, 
        data_type: str = 'normal'
    ):
        """
        Args:
        data_folder: 데이터 폴더 경로
        batch_size: 배치 크기
        scaling: 스케일링 적용 여부
        task_type: 'cls' 또는 'reg' (분류 또는 회귀)
        model_name: 사용할 transformer 모델 이름 (ChemBERTa)
        data_type: 'normal' 또는 'reduce' 또는 'none' (필터 컬럼 이용 여부)
        """

        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.model_name = model_name
        self.task_type = task_type
        self.data_type = data_type
        
        # 데이터 유형에 따른 필터 컬럼 설정
        if data_type == 'normal':
            from utils import NORMAL_FILTER_COLS 
            self.filter_cols = NORMAL_FILTER_COLS
            from utils import NORMAL_CLS_COLS
            base_cols = NORMAL_CLS_COLS
        elif data_type == 'reduce':  # reduce
            from utils import REDUCE_FILTER_COLS
            self.filter_cols = REDUCE_FILTER_COLS
            from utils import REDUCE_CLS_COLS
            base_cols = REDUCE_CLS_COLS
        elif data_type == 'none':
            self.filter_cols = []
            from utils import NORMAL_CLS_COLS   # task list 동일하게 사용
            base_cols = NORMAL_CLS_COLS
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if data_type == 'none':
            self.use_attention = False
        
        # 태스크 목록 설정
        if task_type == 'cls':
            self.task_list = [f'{x}.cls' for x in base_cols]
        else:  # reg
            self.task_list = base_cols
        
        # 데이터 로딩
        x_path = os.path.join(data_folder, 'admet_X.tsv')
        y_path = os.path.join(data_folder, 'admet_Y.tsv')
        scaler_path = os.path.join(data_folder, 'scale_config.csv')
        
        x_df = pd.read_csv(x_path, sep='\t')
        y_df = pd.read_csv(y_path, sep='\t')
        
        # 데이터 병합 및 전처리
        self.all_df = pd.merge(x_df, y_df, on='id')
        
        # 스케일링 적용
        if scaling:
            self.all_df = standard_scaling(self.all_df, scaler_path)
        
        # 축소된 데이터셋 사용 시 전처리
        if data_type == 'reduce':
            self.all_df = prepare_reduced_dataset(self.all_df)
        
        # 어휘 및 데이터셋 준비
        self.all_vocabs = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        if data_type == 'none' and len(self.filter_cols) != 0:
            raise RuntimeError("data_type 'none' should result in zero filter columns.")

    def setup(self):
        """데이터셋 준비 및 분할"""
        # 학습/검증/테스트 데이터셋 분할
        T = int(len(self.all_df) * 0.7)  # 70% 훈련 데이터
        train_df = self.all_df.sample(n=T, random_state=42)
        t_indexs = self.all_df.index.isin(train_df.index)
        other_df = self.all_df[~t_indexs]
        
        V = int(len(other_df) * 0.5)  # 남은 데이터의 50%를 검증 데이터로
        valid_df = other_df.sample(n=V, random_state=42)
        v_indexs = other_df.index.isin(valid_df.index)
        test_df = other_df[~v_indexs]
        
        # 어휘 로딩 (분류 작업용)
        if self.task_type == 'cls':
            # 클래스 레이블이 있는 컬럼 찾기
            cls_targets = [x for x in self.all_df.columns.tolist() if x.endswith('.cls')]
            self.all_vocabs = load_vocabs(self.data_folder, cls_targets)
        else:
            self.all_vocabs = {}
        
        # 데이터셋 생성
        self.train_dataset = ChemMultiTaskDataset(
            train_df, self.all_vocabs, self.model_name, self.task_type, self.data_type)
        
        self.valid_dataset = ChemMultiTaskDataset(
            valid_df, self.all_vocabs, self.model_name, self.task_type, self.data_type)
        
        self.test_dataset = ChemMultiTaskDataset(
            test_df, self.all_vocabs, self.model_name, self.task_type, self.data_type)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """훈련, 검증, 테스트 데이터로더 반환"""
        if self.train_dataset is None:
            self.setup()
            
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=2,  # 워커 수 감소
            pin_memory=True  # 메모리 효율성 향상
        )
        
        valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=2,  # 워커 수 감소
            pin_memory=True  # 메모리 효율성 향상
        )
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=2,  # 워커 수 감소
            pin_memory=True  # 메모리 효율성 향상
        )
        
        return train_loader, valid_loader, test_loader
    
    def _collate_fn(self, batch):
        """배치 데이터 조합 함수"""
        if not batch:  # 빈 배치 처리
            raise ValueError("Empty batch encountered")
            
        # 결과 준비
        if self.model_name is not None:
            # Transformer 모델 사용 시
            # 범주형 특성 처리
            cat_features = []
            for i in range(len(self.filter_cols)):
                cat_tensor = torch.cat([item['features'][i] for item in batch], dim=0)
                cat_features.append(cat_tensor)
            
            # 정수형 및 실수형 특성 처리
            int_tensor = torch.cat([item['features'][len(self.filter_cols)] for item in batch], dim=0)
            float_tensor = torch.cat([item['features'][len(self.filter_cols)+1] for item in batch], dim=0)
            
            # 레이블 처리
            labels = torch.stack([item['labels'] for item in batch])
            
            # 입력 ID 및 어텐션 마스크 처리
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            
            return {
                'features': cat_features + [int_tensor, float_tensor],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            # Transformer 모델 없이 특성만 사용 시
            # 범주형 특성 처리
            cat_features = []
            for i in range(len(self.filter_cols)):
                cat_tensor = torch.cat([item['features'][i] for item in batch], dim=0)
                cat_features.append(cat_tensor)
            
            # 정수형 및 실수형 특성 처리
            int_tensor = torch.cat([item['features'][len(self.filter_cols)] for item in batch], dim=0)
            float_tensor = torch.cat([item['features'][len(self.filter_cols)+1] for item in batch], dim=0)
            
            # 레이블 처리
            labels = torch.stack([item['labels'] for item in batch])
            
            return {
                'features': cat_features + [int_tensor, float_tensor],
                'labels': labels
            }