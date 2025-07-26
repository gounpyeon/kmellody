import torch
import os
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs='+', required=True, help="List of input directories containing model_weights.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the averaged model")
    args = parser.parse_args()

    model_paths = [os.path.join(d, "final_model", "model_weights.pt") for d in args.input_dirs]
    avg_weights = average_weights(model_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "model_weights.pt")
    torch.save(avg_weights, output_path)
    print(f"✅ 평균 모델 저장 완료: {output_path}")