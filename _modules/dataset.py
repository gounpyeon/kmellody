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

from _modules.utils import DIDB_FILTER_COLS, standard_scaling, get_task_list, load_vocabs, LOAD_DATA_PATH


class ChemMultiTaskDataset(Dataset):
    """ChemBERTa 기반 Multi-Task Learning을 위한 데이터셋"""
    def __init__(
        self,
        df_dataset: pd.DataFrame,
        all_vocabs: Dict[str, Dict[str, int]],
        model_name: Optional[str] = None,
        task_type: str = 'cls',
    ):
        """
        Args:
            df_dataset: 입력 데이터프레임
            all_vocabs: 분류 태스크를 위한 레이블 매핑 사전
            model_name: 사용할 transformer 모델 이름 (ChemBERTa)
            task_type: 'cls' 또는 'reg' (분류 또는 회귀)
        """
        self.model_name = model_name
        self.task_type = task_type
        
        # 태스크 목록 설정
        self.y_cols = get_task_list(task_type)
        self.all_vocabs = all_vocabs

        # 데이터 저장
        self.df = df_dataset.copy()

        # Tokenizer는 __init__에서 준비만 하고, 실제 토큰화는 __getitem__에서 수행
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.df)

    def __getitem__(self, idx):
        """데이터셋에서 idx번째 항목 반환"""
        row = self.df.iloc[idx]

        # Tokenization은 여기서 수행
        objs = self.tokenizer(row['SMILES'], padding='max_length', max_length=510,
                              truncation=True, return_tensors='pt')
        
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

        # squeeze(0)로 배치 차원 제거
        return {'input_ids': objs['input_ids'].squeeze(0),
                'attention_mask': objs['attention_mask'].squeeze(0),
                'labels': labels}


class ChemMultiTaskDataModule:
    def __init__(
        self, 
        data_folder: str, 
        batch_size: int = 32, 
        scaling: bool = True, 
        task_type: str = 'cls', 
        model_name: Optional[str] = None, 
        missing_label_strategy: str = 'any',
        data_type: str = 'admet',
    ):
        """
        Args:
            data_folder: 데이터 폴더 경로
            batch_size: 배치 크기
            scaling: 스케일링 적용 여부
            task_type: 'cls' 또는 'reg' (분류 또는 회귀)
            model_name: 사용할 transformer 모델 이름 (ChemBERTa)
            missing_label_strategy: 
                'any' - (default) 한 개 이상의 클래스에 라벨이 있으면 포함
                'all' - 모든 클래스 라벨이 있어야 포함
        """
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.model_name = model_name
        self.task_type = task_type
        self.missing_label_strategy = missing_label_strategy  # <--- 추가
        self.data_type = data_type
        
        # 데이터 유형에 따른 필터 컬럼 설정
        self.filter_cols = DIDB_FILTER_COLS

        # 태스크 목록 설정
        if task_type == 'cls':
            self.task_list = [f'{x}.cls' for x in self.filter_cols]
        else:
            self.task_list = self.filter_cols

        self.all_df = self._load_and_merge_data()
        scaler_path = os.path.join(data_folder, 'scale_config_didb.csv')

        # [1] 결측 행 처리 (여기서 옵션 적용)
        self.all_df = self.all_df[['SMILES'] + self.filter_cols].reset_index(drop=True)

        if self.missing_label_strategy == 'all':
            # 모든 컬럼이 결측 아닌 행만 유지
            valid_rows = self.all_df[self.filter_cols].notna().all(axis=1)
        elif self.missing_label_strategy == 'any':
            valid_rows = self.all_df[self.filter_cols].notna().any(axis=1)
        else:
            raise ValueError(f"알 수 없는 missing_label_strategy: {self.missing_label_strategy}")
        
        self.all_df = self.all_df[valid_rows].reset_index(drop=True)

        # [2] NaN이 아닌 값에 대해서 0~10 값 유지
        # (주석대로 NaN이 아닌 값만 0~10 범위로 유지, NaN은 그대로 두고 필터링)
        if self.data_type == "raw":
            for col in self.filter_cols:
                mask = self.all_df[col].isna() | ((self.all_df[col] >= 0) & (self.all_df[col] <= 10))
                self.all_df = self.all_df[mask].reset_index(drop=True)

        # [3] 스케일링 적용
        if scaling:
            self.all_df = standard_scaling(self.all_df, scaler_path)

        # 어휘 및 데이터셋 준비
        self.all_vocabs = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def _load_and_merge_data(self):
        paths = []
        if self.data_type == "raw":
            paths = [LOAD_DATA_PATH["raw"]]
        elif self.data_type == "admet":
            paths = [LOAD_DATA_PATH["preprocess"]]
        elif self.data_type == "portal":
            paths = [LOAD_DATA_PATH["preprocess"], LOAD_DATA_PATH["portal"]]
        elif self.data_type == "all":
            paths = [LOAD_DATA_PATH["preprocess"], LOAD_DATA_PATH["portal"], LOAD_DATA_PATH["kist"]]
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")
        dfs = []
        cols = ["SMILES"] + DIDB_FILTER_COLS
        for path in paths:
            df = pd.read_csv(path)
            for col in cols:
                if col not in df.columns:
                    df[col] = np.nan
            # SMILES를 제외한 DIDB_FILTER_COLS 컬럼에 대해 문자열 숫자 변환
            for col in DIDB_FILTER_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs.append(df[cols])
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df

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
            train_df, self.all_vocabs, self.model_name, self.task_type)
        
        self.valid_dataset = ChemMultiTaskDataset(
            valid_df, self.all_vocabs, self.model_name, self.task_type)
        
        self.test_dataset = ChemMultiTaskDataset(
            test_df, self.all_vocabs, self.model_name, self.task_type)
    
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
        if not batch:
            raise ValueError("Empty batch encountered")
            
        # 레이블 처리
        labels = torch.stack([item['labels'] for item in batch])
        
        # 입력 ID 및 어텐션 마스크 처리
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }