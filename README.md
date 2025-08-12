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
│   ├── client1/            # 클라이언트 1 데이터
│   ├── client2/            # 클라이언트 2 데이터
│   ├── client3/            # 클라이언트 3 데이터
│   └── ...                 # 기타 데이터 파일들
├── results/                # 결과 저장
│   ├── final_model/        # 훈련된 모델 저장
│   ├── logs/               # TensorBoard 로그
│   └── predictions.txt     # 예측 결과
├── run.py                  # 메인 엔트리포인트 (학습/테스트/예측 실행)
├── fedavg_merge.py         # Federated Average 모델 병합
├── fedavg_simulation.sh # Federated Learning 시뮬레이션 자동 실행 스크립트
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
## Federated Learning

### 개요
Federated Learning 시뮬레이션 코드 \
FedAvg(Federated Averaging) 환경을 가정하여 3개 클라이언트가 참여하는 연합학습 환경을 가상으로 구현

### 시뮬레이션 환경
- **클라이언트 수**: 3개 (client1, client2, client3)
- **알고리즘**: FedAvg (Federated Averaging)
- **데이터 분할**: 3가지 다른 시드로 데이터 분할

### 데이터 구조
```
data/
├── client1/          # 첫 번째 시드 - 클라이언트 1
├── client2/          # 첫 번째 시드 - 클라이언트 2  
├── client3/          # 첫 번째 시드 - 클라이언트 3
├── client1_2/        # 두 번째 시드 - 클라이언트 1
├── client2_2/        # 두 번째 시드 - 클라이언트 2
├── client3_2/        # 두 번째 시드 - 클라이언트 3
├── client1_3/        # 세 번째 시드 - 클라이언트 1
├── client2_3/        # 세 번째 시드 - 클라이언트 2
└── client3_3/        # 세 번째 시드 - 클라이언트 3
```

### FedAvg 알고리즘 시뮬레이션 과정

1. **로컬 훈련**: 각 클라이언트별 데이터로 로컬 모델 훈련
2. **모델 병합**: 각 클라이언트의 로컬 모델 가중치 평균하여 하나의 모델로 병합
3. **성능 평가**: 병합된 모델을 전체 테스트 데이터로 평가

### 자동 실행 스크립트
```bash
sh fedavg_simulation.sh
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
- McMahan, H. B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017. 