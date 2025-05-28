# run.py - 메인 실행 스크립트
import os
import gc
import argparse
import numpy as np
import torch
from trainer import ChemBERTaTrainer


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

    parser.add_argument("--model_name", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--data_folder", type=str, default="/workspace/admet_recent/data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--task_type", type=str, choices=["cls", "reg"], default="cls")
    parser.add_argument("--data_type", type=str, choices=["normal", "reduce", "none"], default="none")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=["train", "test", "predict"], default="train")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--smiles_file", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0")
    return parser.parse_args()


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
        data_type=args.data_type,
        use_attention=args.use_attention,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=args.early_stopping_patience
    )

    if args.mode == "train":
        print("\n[모드: 훈련] 모델 훈련을 시작합니다...")
        test_results = trainer.train()
        trainer.save_model()
        print("\n====== 테스트 결과 ======")
        for result_dict in test_results:
            for metric, value in result_dict.items():
                print(f"{metric}: {value:.4f}")
        print(f"\n모델이 {args.output_dir}/final_model 에 저장되었습니다.")

    elif args.mode == "test":
        model_path = args.model_path or os.path.join(args.output_dir, "final_model")
        print(f"[모드: 테스트] 모델 로드 중: {model_path}")
        model = trainer.load_model(model_path)
        print("테스트 데이터 평가 중...")
        _, _, test_loader = trainer.data_module.get_dataloaders()
        trainer.trainer.test(model=model, dataloaders=test_loader)

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


if __name__ == "__main__":
    main()