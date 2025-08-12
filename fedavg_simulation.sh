#!/bin/bash

# GPU 메모리 정리 및 에러 처리
set -e  # 에러 발생시 스크립트 중단
export CUDA_VISIBLE_DEVICES=0

echo "=== Federated Learning Simulation 시작 ==="

# 그룹 1 (첫 번째 시드)
echo "--- 그룹 1 훈련 시작 ---"
python run.py --data_folder ./data/client1 --output_dir ./results/client1 --mode train
python run.py --data_folder ./data/client2 --output_dir ./results/client2 --mode train
python run.py --data_folder ./data/client3 --output_dir ./results/client3 --mode train

echo "--- 그룹 1 모델 병합 ---"
python fedavg_merge.py --input_dirs ./results/client1 ./results/client2 ./results/client3 --output_dir ./results/fedavg_group1

# 그룹 2 (두 번째 시드)
echo "--- 그룹 2 훈련 시작 ---"
python run.py --data_folder ./data/client1_2 --output_dir ./results/client1_2 --mode train
python run.py --data_folder ./data/client2_2 --output_dir ./results/client2_2 --mode train
python run.py --data_folder ./data/client3_2 --output_dir ./results/client3_2 --mode train

echo "--- 그룹 2 모델 병합 ---"
python fedavg_merge.py --input_dirs ./results/client1_2 ./results/client2_2 ./results/client3_2 --output_dir ./results/fedavg_group2

# 그룹 3 (세 번째 시드)
echo "--- 그룹 3 훈련 시작 ---"
python run.py --data_folder ./data/client1_3 --output_dir ./results/client1_3 --mode train
python run.py --data_folder ./data/client2_3 --output_dir ./results/client2_3 --mode train
python run.py --data_folder ./data/client3_3 --output_dir ./results/client3_3 --mode train

echo "--- 그룹 3 모델 병합 ---"
python fedavg_merge.py --input_dirs ./results/client1_3 ./results/client2_3 ./results/client3_3 --output_dir ./results/fedavg_group3

# 테스트
echo "--- 연합 모델 테스트 ---"
python run.py --mode test --model_path ./results/fedavg_group1 --data_folder ./data --output_dir ./results/test_group1

python run.py --mode test --model_path ./results/fedavg_group2 --data_folder ./data --output_dir ./results/test_group2

python run.py --mode test --model_path ./results/fedavg_group3 --data_folder ./data --output_dir ./results/test_group3

echo "=== 모든 실험 완료! ==="