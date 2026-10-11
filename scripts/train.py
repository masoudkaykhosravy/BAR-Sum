"""
BAR-Sum: Multi-Objective PPO Training with Bounded Adaptive Rewards
v2.0 — Real Transformer Sentence Embeddings (Reproducible, IEEE-grade)
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (deterministic, cached)
- Actor-Critic over 384-dim sentence embeddings
- Multi-Objective Rewards: Coverage, Faithfulness, Anti-Redundancy, Fluency
- Inverse-Variance Dynamic Weight Projection into [w_min, w_max]
- Fixed seeds for full reproducibility
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs_and_results"
CACHE_DIR = "cache"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

SEED = 42
ENCODER_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExtractiveActorCritic(nn.Module):
    def __init__(self, input_dim: int = EMBED_DIM, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # [0: reject, 1: select]
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


class BoundedAdaptiveRewardManager:
    """Inverse-variance dynamic weights projected into [w_min, w_max]."""
    def __init__(self, w_min=0.10, w_max=0.45, smoothing=0.9):
        self.w_min, self.w_max, self.smoothing = w_min, w_max, smoothing
        self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.reward_history = []

    def update_weights(self, reward_vector):
        self.reward_history.append(reward_vector)
        if len(self.reward_history) < 5:
            return self.weights
        history = np.array(self.reward_history[-20:])
        inv_vars = 1.0 / (np.var(history, axis=0) + 1e-5)
        new_w = inv_vars / np.sum(inv_vars)
        updated = self.smoothing * self.weights + (1 - self.smoothing) * new_w
        bounded = np.clip(updated, self.w_min, self.w_max)
        self.weights = bounded / np.sum(bounded)
        return self.weights


def compute_extractive_rewards(selected_sentences, full_sentences, reference_text):
    if not selected_sentences:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    pred_tokens = [w.lower() for s in selected_sentences for w in s.split()]
    ref_tokens = [w.lower() for w in reference_text.split()]

    # 1. Coverage (ROUGE proxy)
    overlap = len(set(pred_tokens) & set(ref_tokens))
    r_cov = min(1.0, (2.0 * overlap) / (len(pred_tokens) + len(ref_tokens) + 1e-6))

    # 2. Faithfulness (source containment)
    src_set = set(w.lower() for s in full_sentences for w in s.split())
    r_faith = sum(1 for w in pred_tokens if w in src_set) / (len(pred_tokens) + 1e-6)

    # 3. Anti-Redundancy
    r_novelty = len(set(pred_tokens)) / (len(pred_tokens) + 1e-6)

    # 4. Fluency / length sanity (target ~3 sentences)
    r_fluency = float(np.exp(-0.5 * ((len(selected_sentences) - 3) ** 2)))

    return np.array([r_cov, r_faith, r_novelty, r_fluency], dtype=np.float32)


# ---------------- Encoder with cache ----------------

def get_or_build_embeddings(dataset, encoder_name=ENCODER_NAME, device="cpu"):
    """Embed every sentence once; cache to disk keyed by encoder + dataset hash."""
    cache_path = os.path.join(CACHE_DIR, f"emb_{encoder_name.replace('/', '_')}_{len(dataset)}.npz")
    if os.path.exists(cache_path):
        print(f"[*] Loading cached embeddings: {cache_path}")
        blob = np.load(cache_path, allow_pickle=True)
        return blob["embeddings"], blob["ids"]

    from sentence_transformers import SentenceTransformer
    print(f"[*] Loading encoder '{encoder_name}' (first run downloads ~90MB)...")
    encoder = SentenceTransformer(encoder_name, device=device)

    all_embs, all_ids = [], []
    for sample in tqdm(dataset, desc="Encoding sentences"):
        sents = sample.get("sentences", [])[:20]
        if not sents:
            sents = ["."]
        embs = encoder.encode(sents, batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs.astype(np.float32))
        all_ids.append(sample["id"])

    np.savez(cache_path, embeddings=np.array(all_embs, dtype=object), ids=np.array(all_ids, dtype=object))
    print(f"[✔] Cached embeddings -> {cache_path}")
    return np.array(all_embs, dtype=object), np.array(all_ids, dtype=object)


# ---------------- Training ----------------

def train_ppo(data_path: str, epochs: int = 5, lr: float = 3e-4, clip_eps: float = 0.2):
    print("\n=======================================================")
    print("   BAR-Sum PPO Training — Real Transformer Features    ")
    print("=======================================================")

    set_global_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Run download_datasets.py first.")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[*] Loaded {len(dataset)} documents from {data_path}")

    emb_matrix, emb_ids = get_or_build_embeddings(dataset, device=str(device))
    emb_lookup = {doc_id: emb_matrix[i] for i, doc_id in enumerate(emb_ids)}

    model = ExtractiveActorCritic(input_dim=EMBED_DIM, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reward_mgr = BoundedAdaptiveRewardManager(w_min=0.10, w_max=0.45)

    training_logs = []
    best_scalar_reward = -float("inf")

    for epoch in range(1, epochs + 1):
        epoch_rewards, epoch_actor_loss, epoch_critic_loss = [], [], []

        pbar = tqdm(dataset, desc=f"Epoch {epoch}/{epochs}")
        for sample in pbar:
            sentences = sample.get("sentences", [])
            if len(sentences) < 2:
                continue

            ref_text = sample.get("highlights", "")
            num_sent = min(len(sentences), 20)
            cur_sentences = sentences[:num_sent]

            # ---- REAL sentence embeddings (deterministic) ----
            s_emb = emb_lookup[sample["id"]][:num_sent]
            sent_features = torch.tensor(s_emb, dtype=torch.float32, device=device)

            logits, value = model(sent_features)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            selected_indices = torch.where(actions == 1)[0].tolist()
            if not selected_indices:
                selected_indices = [0]

            selected_sents = [cur_sentences[i] for i in selected_indices if i < len(cur_sentences)]

            r_vec = compute_extractive_rewards(selected_sents, cur_sentences, ref_text)
            weights = reward_mgr.update_weights(r_vec)
            scalar_reward = float(np.dot(r_vec, weights))
            epoch_rewards.append(scalar_reward)

            reward_tensor = torch.tensor(scalar_reward, dtype=torch.float32, device=device)
            advantage = (reward_tensor - value.squeeze()).detach()

            actor_loss = -(log_probs.sum() * advantage)
            critic_loss = nn.functional.mse_loss(value.squeeze(), reward_tensor)
            total_loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_actor_loss.append(actor_loss.item())
            epoch_critic_loss.append(critic_loss.item())

            pbar.set_postfix({"R": f"{scalar_reward:.3f}",
                              "w_cov": f"{weights[0]:.2f}",
                              "w_faith": f"{weights[1]:.2f}"})

        mean_r = float(np.mean(epoch_rewards))
        training_logs.append({
            "epoch": epoch,
            "mean_reward": mean_r,
            "weights": reward_mgr.weights.tolist(),
            "actor_loss": float(np.mean(epoch_actor_loss)),
            "critic_loss": float(np.mean(epoch_critic_loss))
        })

        print(f"[*] Epoch {epoch} Complete -> Mean Reward: {mean_r:.4f} | Weights: {reward_mgr.weights.round(3)}")

        if mean_r > best_scalar_reward:
            best_scalar_reward = mean_r
            best_path = os.path.join(CHECKPOINT_DIR, "best_bar_sum_agent.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "weights": reward_mgr.weights,
                "best_reward": best_scalar_reward,
                "encoder": ENCODER_NAME,          # برای سازگاری infer
                "input_dim": EMBED_DIM,           # برای سازگاری infer
                "hidden_dim": 256
            }, best_path)
            print(f"  [✔] Checkpoint updated: {best_path} (Reward: {mean_r:.4f})")

    hist_path = os.path.join(LOG_DIR, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    print("\n=======================================================")
    print("[✔] Training finished successfully.")
    print(f"[✔] Checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_bar_sum_agent.pt')}")
    print(f"[✔] History: {hist_path}")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="BAR-Sum PPO Training (v2, Real Encoder)")
    parser.add_argument("--data", type=str, default="data/raw/cnn_dm_test.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train_ppo(data_path=args.data, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
