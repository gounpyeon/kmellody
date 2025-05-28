from trainer import ChemBERTaTrainer
import torch


model_path = "results/final_model/model_weights.pt"
data_folder = "/workspace/admet_recent/data"

trainer = ChemBERTaTrainer(
    model_name="seyonec/ChemBERTa-zinc-base-v1",
    data_folder=data_folder,
    output_dir="results/single",
    task_type="cls",
    data_type="none",
    use_attention=False,
    hidden_dim=256,
    batch_size=16,
    learning_rate=2e-5,
    epochs=1  # 중요하지 않음
)

model = trainer.load_model("results/final_model")
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

_, _, test_loader = trainer.data_module.get_dataloaders()
trainer.trainer.test(model=model, dataloaders=test_loader)