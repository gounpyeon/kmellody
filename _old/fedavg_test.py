from trainer import ChemBERTaTrainer
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="results/single", help="Path to save evaluation results")
    args = parser.parse_args()

    trainer = ChemBERTaTrainer(
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        data_folder=args.data_dir,
        output_dir=args.output_dir,
        task_type="cls",
        data_type="none",
        use_attention=False,
        hidden_dim=256,
        batch_size=16,
        learning_rate=2e-5,
        epochs=1
    )

    model = trainer.load_model(args.model_path)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    _, _, test_loader = trainer.data_module.get_dataloaders()
    trainer.trainer.test(model=model, dataloaders=test_loader)