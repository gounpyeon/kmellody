# ChemBERTa Multi-Task Learning

ChemBERTa 기반 다중 태스크 학습을 위한 프로젝트입니다.

## 주요 기능

- **ChemBERTa 모델**: 분자 구조를 이해하는 BERT 기반 모델
- **다중 태스크 학습**: 여러 ADMET 관련 태스크를 동시에 학습
- **WandB 로깅**: 실험 추적 및 하이퍼파라미터 튜닝
- **TensorBoard 로깅**: 대안 로깅 시스템
- **PyTorch Lightning**: 구조화된 딥러닝 훈련 프레임워크

## 프로젝트 구조

```
kmellody/
├── _modules/               # 실행 관련 모듈들
│   ├── trainer.py          # PyTorch Lightning 훈련 모듈
│   ├── model.py            # ChemBERTa Multi-Task 모델 정의
│   ├── dataset.py          # 데이터셋 및 데이터 로더
│   └── utils.py            # 유틸리티 함수 및 상수
├── config/                 # 설정 파일들
│   └── sweep.json          # WandB Sweep 설정
├── results/                # 결과 저장
│   ├── final_model/        # 훈련된 모델 저장
│   ├── logs/               # TensorBoard 로그
│   └── predictions.txt     # 예측 결과
├── run.py                  # 메인 엔트리포인트 (학습/테스트/예측 실행)
├── data/                   # 데이터 파일들
├── wandb/                  # WandB 로그 (자동 생성)
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 일반 학습 (TensorBoard)

```bash
python run.py \
    --mode train \
    --epochs 10 \
    --use_wandb false
```

### 2. WandB 일반 학습

```bash
python run.py \
    --mode train \
    --epochs 10 \
    --use_wandb \
    --wandb_project "chemberta-admet"
```

### 3. WandB Sweep (하이퍼파라미터 튜닝)

```bash
python run.py \
    --sweep \
    --sweep_config config/sweep.json \
    --wandb_project "chemberta-admet"
```

### 4. 테스트

```bash
python run.py \
    --mode test \
    --model_path ./results/final_model
```

### 5. 예측

```bash
python run.py \
    --mode predict \
    --model_path ./results/final_model \
    --smiles_file smiles_list.txt
```

## WandB 설정

### 일반 Logging 모드
- 실험 메트릭, 파라미터, 모델을 WandB에 자동 로깅
- 실시간 대시보드에서 학습 진행 상황 모니터링
- 모델 체크포인트 자동 저장

### Sweep 모드
- 하이퍼파라미터 자동 튜닝
- Grid 방식으로 모든 조합 탐색
- 최적 파라미터 조합 자동 탐색

## Sweep 설정 파일 (JSON)

```json
{
  "method": "grid",
  "metric": {
    "name": "val_loss",
    "goal": "minimize"
  },
  "parameters": {
    "task_type": {
      "values": ["cls", "reg", "multi_reg"]
    },
    "missing_label_strategy": {
      "values": ["any", "all"]
    },
    "data_type": {
      "values": ["raw", "admet", "portal", "all"]
    }
  }
}
```

## 주요 파라미터

- `--model_name`: 사용할 ChemBERTa 모델명
- `--data_folder`: 데이터 폴더 경로
- `--output_dir`: 결과 저장 경로
- `--batch_size`: 배치 크기
- `--learning_rate`: 학습률
- `--epochs`: 에포크 수
- `--task_type`: 태스크 유형 (cls, reg, multi_reg)
- `--data_type`: 데이터 유형 (raw, admet, portal, all)
- `--use_wandb`: WandB 로깅 사용 여부
- `--wandb_project`: WandB 프로젝트명
- `--sweep`: WandB sweep 사용 여부

## VSCode 디버깅

`.vscode/launch.json`에 세 가지 설정이 포함되어 있습니다:

1. **일반 학습 (TensorBoard)**: TensorBoard 로깅 사용
2. **WandB 일반 학습**: WandB 로깅 사용
3. **WandB Sweep**: 하이퍼파라미터 튜닝 실행

## 로깅 시스템

### WandB 로깅
- 실시간 메트릭 추적
- 모델 아티팩트 저장
- 하이퍼파라미터 튜닝
- 실험 비교 및 분석

### TensorBoard 로깅
- 로컬 로깅 시스템
- 실시간 시각화
- 로그 파일 위치: `./results/logs/`

## 결과

- 모델 파일: `./results/final_model/`
- 로그 파일: `./results/logs/` (TensorBoard)
- WandB 대시보드: 실시간 실험 추적

## 참고 자료

- [PyTorch Lightning 문서](https://lightning.ai/docs/pytorch/stable/)
- [WandB 가이드](https://docs.wandb.ai/)
- [WandB Sweep 문서](https://docs.wandb.ai/guides/sweeps)
- [TensorBoard 가이드](https://www.tensorflow.org/tensorboard)