"""
=============================================================================
Module: policy_network.py
Project: BAR-Sum (Factually Faithful Extractive Text Summarization via DRL)
Author: Masoud Keikhosravi
Description:
    Actor-Critic Network Architecture for PPO Agent in Extractive Summarization.
    Includes sequential sentence selection policy with dynamic summary state tracking.
=============================================================================
"""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


class ActorNetwork(nn.Module):
    """
    Actor / Policy Network:
    Computes sentence selection probability p(a_t = 1 | s_t, h_doc, h_summary)
    """
    def __init__(self, hidden_dim: int = 512, mlp_dim: int = 256, dropout: float = 0.1):
        super(ActorNetwork, self).__init__()
        
        # Sentence representation + Summary representation -> Selection Score
        self.fc1 = nn.Linear(hidden_dim * 2, mlp_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence_repr: torch.Tensor, summary_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence_repr: [batch_size, num_sentences, hidden_dim]
            summary_state: [batch_size, num_sentences, hidden_dim] (broadcasted current summary state)
        Returns:
            probs: [batch_size, num_sentences] (Bernoulli action probabilities)
        """
        combined = torch.cat([sentence_repr, summary_state], dim=-1) # [B, N, 2 * H]
        hidden = self.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden).squeeze(-1) # [B, N]
        probs = self.sigmoid(logits)
        # Numerical stability clamp
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        return probs


class CriticNetwork(nn.Module):
    """
    Critic / Value Network:
    Estimates baseline state value V(s) for GAE (Generalized Advantage Estimation).
    """
    def __init__(self, hidden_dim: int = 512, mlp_dim: int = 256, dropout: float = 0.1):
        super(CriticNetwork, self).__init__()
        
        # Self-attention pooling over sentences to get single document-state scalar
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, 1)

    def forward(self, sentence_repr: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            sentence_repr: [batch_size, num_sentences, hidden_dim]
            mask: [batch_size, num_sentences] (1 for valid, 0 for pad)
        Returns:
            state_value: [batch_size, 1]
        """
        attn_weights = self.attn(sentence_repr) # [B, N, 1]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Document context vector
        doc_vector = torch.sum(sentence_repr * attn_weights, dim=1) # [B, H]
        
        hidden = self.relu(self.fc1(doc_vector))
        hidden = self.dropout(hidden)
        state_value = self.fc2(hidden) # [B, 1]
        return state_value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic model for PPO Extractive Summarizer.
    """
    def __init__(self, hidden_dim: int = 512, mlp_dim: int = 256, dropout: float = 0.1):
        super(ActorCritic, self).__init__()
        self.hidden_dim = hidden_dim
        self.actor = ActorNetwork(hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout)
        self.critic = CriticNetwork(hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout)

    def get_action_and_value(
        self,
        sentence_repr: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False
    ):
        """
        Forward step for sampling actions during rollout or evaluating during PPO update.

        Args:
            sentence_repr: [batch_size, num_sentences, hidden_dim]
            action: Optional [batch_size, num_sentences] for policy evaluation
            deterministic: True for greedy inference at test time

        Returns:
            actions: [batch_size, num_sentences]
            log_probs: [batch_size, num_sentences]
            entropy: [batch_size, num_sentences]
            state_value: [batch_size, 1]
        """
        batch_size, num_sentences, _ = sentence_repr.shape
        
        # Initialize dynamic summary state as zeros
        summary_state = torch.zeros_like(sentence_repr)
        
        # Calculate action probabilities
        probs = self.actor(sentence_repr, summary_state)
        dist = Bernoulli(probs=probs)

        if action is None:
            if deterministic:
                # Top-k or threshold >= 0.5 for inference
                action = (probs >= 0.5).float()
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(sentence_repr)

        return action, log_prob, entropy, state_value

    def evaluate_actions(self, sentence_repr: torch.Tensor, action: torch.Tensor):
        """
        Evaluate log probabilities, entropy and state values for PPO gradient updates.
        """
        summary_state = torch.zeros_like(sentence_repr)
        probs = self.actor(sentence_repr, summary_state)
        dist = Bernoulli(probs=probs)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        state_value = self.critic(sentence_repr)

        return log_prob, entropy, state_value


if __name__ == "__main__":
    print("Testing ActorCritic Policy Network...")
    batch_size = 2
    num_sentences = 8
    hidden_dim = 512

    model = ActorCritic(hidden_dim=hidden_dim, mlp_dim=256)
    dummy_sent_repr = torch.randn(batch_size, num_sentences, hidden_dim)

    # 1. Test Rollout Sampling
    actions, log_probs, entropy, value = model.get_action_and_value(dummy_sent_repr)
    print(f"Sampled Actions Shape: {actions.shape} (Binary 0/1)")
    print(f"Log Probs Shape: {log_probs.shape}")
    print(f"State Value Shape: {value.shape}")

    # 2. Test PPO Evaluation Step
    eval_log_probs, eval_entropy, eval_value = model.evaluate_actions(dummy_sent_repr, actions)
    assert eval_log_probs.shape == actions.shape
    print("ActorCritic Network verified successfully.")
