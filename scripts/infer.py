"""
BAR-Sum: Inference Script (v2)
- Uses real sentence-transformers encoder
- Aligned with training checkpoint structure
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class ExtractiveActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x.mean(dim=0, keepdim=True))
        return logits, value

def run_inference(data_path, ckpt_path, output_path, device="cpu"):
    # 1. Load Data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[*] Inference on {len(data)} samples.")

    # 2. Load Checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    
    model = ExtractiveActorCritic(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Encoder
    print(f"[*] Loading encoder '{ckpt.get('encoder', 'all-MiniLM-L6-v2')}'...")
    encoder = SentenceTransformer(ckpt.get('encoder', 'all-MiniLM-L6-v2'), device=device)

    predictions = []

    # 4. Inference loop
    with torch.no_grad():
        for sample in tqdm(data, desc="Running Inference"):
            sentences = sample.get("sentences", [])[:20]
            if not sentences: continue
            
            # Encode sentences real-time
            sent_embs = encoder.encode(sentences, convert_to_tensor=True, device=device)
            
            logits, _ = model(sent_embs)
            probs = torch.softmax(logits, dim=-1)
            selected_indices = torch.where(probs[:, 1] > 0.5)[0].tolist()
            
            if not selected_indices:
                selected_indices = [int(torch.argmax(probs[:, 1]))]

            summary = " ".join([sentences[i] for i in selected_indices if i < len(sentences)])

            predictions.append({
                "id": sample.get("id"),
                "prediction": summary,
                "reference": sample.get("highlights", "")
            })

    # 5. Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"[✔] Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/cnn_dm_test.json")
    parser.add_argument("--ckpt", default="checkpoints/best_bar_sum_agent.pt")
    parser.add_argument("--output", default="logs_and_results/predictions.json")
    args = parser.parse_args()
    run_inference(args.data, args.ckpt, args.output)

if __name__ == "__main__":
    main()
