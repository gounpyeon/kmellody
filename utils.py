"""
utils.py - 상수 정의 및 유틸리티 함수
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# 데이터셋 설정을 위한 상수들
NORMAL_FILTER_COLS = [
    'ghose filter',
    "rule of five",
    "veber's rule",
    "mddr-like rule", 
    "bioavailability"
]

REDUCE_FILTER_COLS = ["bioavailability"]

NORMAL_CLS_COLS = [
    'hia', 'bbb', 'caco2', 'p_glycoprotein_substrate',
    'p_glycoprotein_inhibitor_1', 'p_glycoprotein_inhibitor_2',
    'renal_organic_cation_transporter', 'cyp450_2c9_substrate',
    'cyp450_2d6_substrate', 'cyp450_3a4_substrate', 'cyp450_1a2_substrate',
    'cyp450_2c9_inhibitor', 'cyp450_2d6_inhibitor', 'cyp450_2c19_inhibitor',
    'cyp450_3a4_inhibitor', 'cyp_inhibitory_promiscuity', 'ames_toxicity',
    'carcinogenicity', 'biodegradation', 'herg_inhibitor_1',
    'herg_inhibitor_2'
]

REDUCE_CLS_COLS = [
    'hia', 'bbb', 'p_glycoprotein_substrate', 'p_glycoprotein_inhibitor',
    'renal_organic_cation_transporter', 'cyp450_2c9_substrate',
    'cyp450_2d6_substrate', 'cyp450_3a4_substrate', 'cyp450_1a2_substrate',
    'cyp450_2c9_inhibitor', 'cyp450_2d6_inhibitor', 'cyp450_2c19_inhibitor',
    'cyp450_3a4_inhibitor', 'cyp_inhibitory_promiscuity', 'ames_toxicity',
    'carcinogenicity', 'biodegradation', 'herg_inhibitor'
]

INT_COLS = [
    'h bond acceptor count',
    'h bond donor count',
    'rotatable bond count',
    "number of rings",
    "physiological charge"
]

FLOAT_COLS = [
    'molecular weight',
    'refractivity',
    "monoisotopic weight",
    "polar surface area (psa)"
]

def standard_scaling(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """
    데이터프레임의 수치형 특성을 표준화(z-score normalization)합니다.
    
    Args:
        df: 입력 데이터프레임
        scaler_path: 스케일러 설정 파일 경로
        
    Returns:
        표준화된 데이터프레임
    """
    # 기존 스케일러 설정 불러오기 (없으면 생성)
    try:
        if os.path.exists(scaler_path):
            scaler_df = pd.read_csv(scaler_path)
            
            # 스케일러 설정 파일 형식 확인
            if 'feature' in scaler_df.columns and 'mean' in scaler_df.columns and 'std' in scaler_df.columns:
                scaled_df = df.copy()
                
                # 수치형 특성 표준화
                for col in INT_COLS + FLOAT_COLS:
                    if col in df.columns:
                        feature_rows = scaler_df[scaler_df['feature'] == col]
                        if len(feature_rows) > 0:
                            mean = feature_rows['mean'].values[0]
                            std = feature_rows['std'].values[0]
                            scaled_df[col] = (df[col] - mean) / std
            else:
                # 형식이 다른 경우 새로 생성
                print(f"경고: 스케일러 설정 파일 {scaler_path}의 형식이 올바르지 않습니다. 새로 생성합니다.")
                return _create_new_scaler(df, scaler_path)
        else:
            # 스케일러 설정 파일이 없는 경우 새로 생성
            print(f"정보: 스케일러 설정 파일 {scaler_path}을(를) 찾을 수 없습니다. 새로 생성합니다.")
            return _create_new_scaler(df, scaler_path)
            
        return scaled_df
    
    except Exception as e:
        print(f"경고: 스케일링 중 오류 발생: {e}. 새로운 스케일러를 생성합니다.")
        return _create_new_scaler(df, scaler_path)

def _create_new_scaler(df, scaler_path):
    """새로운 스케일러 생성"""
    scaled_df = df.copy()
    scaler_data = []
    
    # 수치형 특성 표준화
    for col in INT_COLS + FLOAT_COLS:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:  # 표준편차가 0인 경우 처리
                std = 1.0
            scaled_df[col] = (df[col] - mean) / std
            scaler_data.append({'feature': col, 'mean': mean, 'std': std})
    
    # 스케일러 설정 저장
    scaler_df = pd.DataFrame(scaler_data)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler_df.to_csv(scaler_path, index=False)
    
    return scaled_df

def load_vocabs(data_folder: str, cls_targets: List[str]) -> Dict[str, Dict[str, int]]:
    """
    분류 태스크용 어휘 사전을 로드합니다.
    
    Args:
        data_folder: 데이터 폴더 경로
        cls_targets: 분류 태스크 레이블 목록
        
    Returns:
        어휘 사전 (태스크별 {레이블: 인덱스})
    """
    all_vocabs = {}
    
    for target in cls_targets:
        vocab_path = os.path.join(data_folder, f'vocabs/{target}.vocab')
        
        if target not in all_vocabs:
            all_vocabs[target] = {}
            
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    vocab, idx = line.split('\t')
                    idx = int(idx)
                    all_vocabs[target][vocab] = idx
        except FileNotFoundError:
            # 파일이 없는 경우 기본값 사용
            print(f"Warning: Vocabulary file not found for {target}. Using default values.")
            all_vocabs[target] = {'inhibitor': 1, 'non-inhibitor': 0}
    
    return all_vocabs

def prepare_reduced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터셋을 축소 버전으로 변환합니다.
    
    Args:
        df: 입력 데이터프레임
        
    Returns:
        축소된 데이터프레임
    """
    # 데이터프레임 복사
    df_reduced = df.copy()
    
    # P-glycoprotein 데이터 병합
    p_glyco_cls = ['Inhibitor' if p1_cls == 'Inhibitor' or p2_cls == 'Inhibitor' else 'Non-inhibitor' 
                  for p1_cls, p2_cls in zip(df['p_glycoprotein_inhibitor_1.cls'],
                                          df['p_glycoprotein_inhibitor_2.cls'])]
    
    p_glyco = [(p1 + p2) / 2 if p1_cls == p2_cls else p1 if p1_cls == 'Inhibitor' else p2 
              for p1_cls, p2_cls, p1, p2 in zip(df['p_glycoprotein_inhibitor_1.cls'],
                                              df['p_glycoprotein_inhibitor_2.cls'],
                                              df['p_glycoprotein_inhibitor_1'],
                                              df['p_glycoprotein_inhibitor_2'])]
    
    # hERG 데이터 병합
    herg_cls = ['Inhibitor' if g1 == 'Strong inhibitor' or g2 == 'Inhibitor' else 'Non-inhibitor' 
               for g1, g2 in zip(df['herg_inhibitor_1.cls'],
                               df['herg_inhibitor_2.cls'])]
    
    herg = [(h1 + h2) / 2 if (h1_cls == 'Strong inhibitor' and h2_cls == 'Inhibitor') or 
                            (h1_cls == 'Weak inhibitor' and h2_cls == 'Non-inhibitor') 
           else h1 if h1_cls == 'Strong inhibitor' else h2
           for h1_cls, h2_cls, h1, h2 in zip(df['herg_inhibitor_1.cls'],
                                           df['herg_inhibitor_2.cls'],
                                           df['herg_inhibitor_1'],
                                           df['herg_inhibitor_2'])]
    
    # 새 컬럼 추가
    df_reduced['p_glycoprotein_inhibitor.cls'] = p_glyco_cls
    df_reduced['p_glycoprotein_inhibitor'] = p_glyco
    df_reduced['herg_inhibitor.cls'] = herg_cls
    df_reduced['herg_inhibitor'] = herg
    
    # 불필요한 컬럼 제거
    df_reduced.drop(['ghose filter', "veber's rule", 'rule of five', 'mddr-like rule', 'caco2',
                      'p_glycoprotein_inhibitor_1.cls', 'p_glycoprotein_inhibitor_2.cls',
                      'p_glycoprotein_inhibitor_1', 'p_glycoprotein_inhibitor_2', 
                      'herg_inhibitor_1.cls', 'herg_inhibitor_2.cls', 
                      'herg_inhibitor_1', 'herg_inhibitor_2'], axis=1, inplace=True)
    
    return df_reduced

def get_task_list(task_type: str, data_type: str) -> List[str]:
    """
    태스크 유형과 데이터 유형에 따른 태스크 목록을 반환합니다.
    
    Args:
        task_type: 태스크 유형 ('cls' 또는 'reg')
        data_type: 데이터 유형 ('normal' 또는 'reduce')
        
    Returns:
        태스크 목록
    """
    if data_type == 'normal':
        y_cols_base = NORMAL_CLS_COLS
    elif data_type == 'reduce':  # reduce
        y_cols_base = REDUCE_CLS_COLS
    elif data_type == 'none':  # none
        y_cols_base = NORMAL_CLS_COLS
    
    # 태스크 유형에 따른 출력 컬럼 설정
    if task_type == 'cls':
        return [f'{x}.cls' for x in y_cols_base]
    else:  # reg
        return y_cols_base