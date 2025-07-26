"""
utils.py - 상수 정의 및 유틸리티 함수
"""
import os, json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

LOAD_DATA_PATH = {
    "raw": "./data/processed_admet_data_clearance.csv",
    "preprocess":"./data/merged_admet.csv",
    "portal":"./data/merged_log_caco2.csv",
    "kist":"./data/kist_logp_Fup.csv"
}

# 데이터셋 설정을 위한 Class name 정의
DIDB_FILTER_COLS = [
    "logP",
    "pKa",
    "solubility",
    "permeability", 
    "plasma_protein_binding",
    # "fm_in_vitro",
    "fu_in_vitro",
    # "CYP450_Enzyme"
]

def standard_scaling(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    processed_admet_data_clearance.csv에서 불러온 df의 DIDB_FILTER_COLS 컬럼에 대해 표준화(standard scaling)를 수행하고,
    스케일러(mean, std)를 scaler_path에 저장한 뒤, 스케일된 DataFrame을 반환합니다.

    Args:
        df: 입력 데이터프레임 (processed_admet_data_clearance.csv에서 불러온 데이터)
        scaler_path: 스케일러(mean, std) 저장 경로

    Returns:
        표준화된 데이터프레임
    """
    scaled_df = df.copy()
    scaler_data = []

    # 각 DIDB_FILTER_COLS 컬럼에 대해 mean, std 계산 및 scaling
    for col in DIDB_FILTER_COLS:
        if col in df.columns:
            # 문자열 등 비수치 데이터 NaN으로 변환
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 결측치가 있을 수 있으므로 skipna=True로 평균/표준편차 계산
            mean = df[col].mean(skipna=True)
            std = df[col].std(skipna=True)
            # 표준편차가 0이거나 NaN인 경우 1로 대체
            if pd.isna(std) or std == 0:
                std = 1.0
            # scaling
            scaled_df[col] = (df[col] - mean) / std
            scaler_data.append({'feature': col, 'mean': mean, 'std': std})

    # 스케일러 정보 저장
    scaler_df = pd.DataFrame(scaler_data)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler_df.to_csv(scaler_path, index=False)

    return scaled_df


def get_task_list(task_type: str) -> List[str]:
    """
    태스크 유형과 데이터 유형에 따른 태스크 목록을 반환합니다.
    
    Args:
        task_type: 태스크 유형 ('cls' 또는 'reg' 또는 'multi_reg')
        
    Returns:
        태스크 목록
    """
    y_cols_base = DIDB_FILTER_COLS
    
    # 태스크 유형에 따른 출력 컬럼 설정
    if task_type == 'cls':
        return [f'{x}.cls' for x in y_cols_base]
    else:  # reg and multi_reg
        return y_cols_base
    
def load_vocabs(vocab_path: str) -> Dict[str, Dict[str, int]]:
    """
    vocab_path에 저장된 어휘 파일을 로드하고, 어휘 사전을 반환합니다.
    """
    with open(vocab_path, 'r') as f:
        vocabs = json.load(f)
    return vocabs