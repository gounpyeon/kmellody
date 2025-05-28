import torch

model_paths = [
    "results/client1_3/final_model/model_weights.pt",
    "results/client2_3/final_model/model_weights.pt",
    "results/client3_3/final_model/model_weights.pt"
]

def average_weights(model_paths):
    avg_state_dict = None
    for path in model_paths:
        state_dict = torch.load(path, map_location='cpu')
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k]
    
    for k in avg_state_dict:
        avg_state_dict[k] /= len(model_paths)

    return avg_state_dict

# 저장
avg_weights = average_weights(model_paths)
torch.save(avg_weights, "results/fedavg_merge_3/model_weights.pt")
print("✅ 평균 모델 저장 완료: results/fedavg_merge_3/model_weights.pt")