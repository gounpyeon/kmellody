# run.py - 메인 실행 스크립트
import os, gc
import argparse, json
import numpy as np
import torch
import wandb
from copy import deepcopy

from _modules.trainer import ChemBERTaTrainer

WANDB_API_KEY = "YOUR_API_KEY"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            print(f"GPU {i} 메모리 정리됨")


def parse_args():
    parser = argparse.ArgumentParser(description="ChemBERTa 기반 Multi-Task Learning")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--mode", type=str, choices=["train", "test", "predict"], default="train")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--smiles_file", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    # Task parameters
    parser.add_argument("--task_type", type=str, choices=["cls", "reg", "multi_reg"], default="multi_reg")
    parser.add_argument("--missing_label_strategy", type=str, choices=["any", "all"], default="any")
    parser.add_argument("--data_type", type=str, choices=["raw", "admet", "portal", "all"], default="admet")

    # WandB parameters
    parser.add_argument("--use_wandb", action="store_true", help="WandB 로깅 사용 여부", default=False)
    parser.add_argument("--wandb_project", type=str, default="Kmelloddy-admet", help="WandB 프로젝트명")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB 엔티티명")
    parser.add_argument("--sweep", action="store_true", help="WandB sweep 사용 여부", default=False)
    parser.add_argument("--sweep_config", type=str, default="./config/sweep.json", help="Sweep 설정 파일 경로")

    return parser.parse_args()


def train_model(args, use_wandb=False):
    """통합된 학습 함수 (WandB 사용 여부 선택 가능)"""
    mode_text = "WandB" if use_wandb else "일반"
    print(f"=== {mode_text} 학습 모드 시작 ===")
    
    # WandB 초기화 (사용하는 경우에만)
    if use_wandb:
        # 간소화된 config - 주요 하이퍼파라미터만 포함
        config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "task_type": args.task_type,
            "data_type": args.data_type,
            "missing_label_strategy": args.missing_label_strategy
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config
        )
    
    trainer = ChemBERTaTrainer(
        model_name=args.model_name,
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        task_type=args.task_type,
        missing_label_strategy=args.missing_label_strategy,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=args.early_stopping_patience,
        data_type=args.data_type
    )
    
    # Logger 설정
    trainer.setup(use_wandb=use_wandb, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)

    if args.mode == "train":
        print("\n[모드: 훈련] 모델 훈련을 시작합니다...")
        test_results = trainer.train()
        trainer.save_model()
        
        print("\n====== 테스트 결과 ======")
        for result_dict in test_results:
            for metric, value in result_dict.items():
                print(f"{metric}: {value:.4f}")
                if use_wandb:
                    wandb.log({f"test_{metric}": value})
        
        print(f"\n모델이 {args.output_dir}/final_model 에 저장되었습니다.")
        if use_wandb:
            wandb.finish()

    elif args.mode == "test":
        model_path = args.model_path or os.path.join(args.output_dir, "final_model")
        print(f"[모드: 테스트] 모델 로드 중: {model_path}")
        model = trainer.load_model(model_path)
        print("테스트 데이터 평가 중...")
        _, _, test_loader = trainer.data_module.get_dataloaders()
        trainer.trainer.test(model=model, dataloaders=test_loader)
        if use_wandb:
            wandb.finish()

    elif args.mode == "predict":
        model_path = args.model_path or os.path.join(args.output_dir, "final_model")
        if args.smiles_file is None:
            print("[오류] 예측 모드에서는 --smiles_file 인자가 필요합니다.")
            return
        print(f"[모드: 예측] 모델 로드 중: {model_path}")
        model = trainer.load_model(model_path)
        with open(args.smiles_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f"총 {len(smiles_list)} 개의 SMILES에 대해 예측 중...")
        predictions = trainer.predict(smiles_list)
        output_file = os.path.join(args.output_dir, "predictions.txt")
        with open(output_file, 'w') as f:
            f.write("SMILES\t" + "\t".join(trainer.task_list) + "\n")
            for i, smiles in enumerate(smiles_list):
                results = []
                for task in trainer.task_list:
                    pred = predictions[task]['predictions'][i] if trainer.task_types[task] == 'classification' else f"{predictions[task][i]:.4f}"
                    results.append(str(pred))
                f.write(smiles + "\t" + "\t".join(results) + "\n")
        print(f"예측 결과가 {output_file} 에 저장되었습니다.")
        if use_wandb:
            wandb.finish()


def train_sweep_run(args):
    """WandB Sweep에서 실행되는 함수"""
    # WandB 초기화 (sweep 모드)
    wandb.init()
    
    # sweep에서 가져온 파라미터로 args 업데이트
    sweep_args = deepcopy(args)
    for key, value in wandb.config.items():
        if hasattr(sweep_args, key):
            setattr(sweep_args, key, value)
            print(f"Sweep 파라미터 적용: {key} = {value}")
    
    trainer = ChemBERTaTrainer(
        model_name=sweep_args.model_name,
        data_folder=sweep_args.data_folder,
        output_dir=sweep_args.output_dir,
        batch_size=sweep_args.batch_size,
        learning_rate=sweep_args.learning_rate,
        epochs=sweep_args.epochs,
        weight_decay=sweep_args.weight_decay,
        warmup_steps=sweep_args.warmup_steps,
        task_type=sweep_args.task_type,
        missing_label_strategy=sweep_args.missing_label_strategy,
        hidden_dim=sweep_args.hidden_dim,
        early_stopping_patience=sweep_args.early_stopping_patience,
        data_type=sweep_args.data_type
    )
    
    # WandB Logger 설정 (sweep 모드)
    trainer.setup(use_wandb=True, wandb_project=args.wandb_project, wandb_entity=args.wandb_entity)
    
    # 학습 실행
    test_results = trainer.train()
    trainer.save_model()
    
    # 결과를 WandB에 기록
    if test_results:
        for result_dict in test_results:
            for metric, value in result_dict.items():
                wandb.log({f"test_{metric}": value})
                print(f"메트릭 기록: test_{metric} = {value}")
    
    wandb.finish()

    

def main():
    args = parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    clear_gpu_memory()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        print(f"활성화된 GPU 수: {torch.cuda.device_count()}")

    # WandB API 키 설정
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

    # WandB Sweep 모드
    if args.sweep:
        print("=== WandB Sweep 모드 시작 ===")
        
        # sweep 설정 로드
        with open(args.sweep_config, 'r') as f:
            sweep_config = json.load(f)
        
        # sweep 초기화
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
        print(f"Sweep ID: {sweep_id}")
        
        # sweep 실행
        wandb.agent(sweep_id, function=lambda: train_sweep_run(args), count=None)
        
        print(f"Sweep 완료. WandB 대시보드에서 결과 확인하세요.")
        return

    # 일반 학습 모드
    if args.use_wandb:
        train_model(args, use_wandb=True)
    else:
        train_model(args, use_wandb=False)


if __name__ == "__main__":
    main()