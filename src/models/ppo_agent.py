"""
=============================================================================
Module: ppo_agent.py
Project: BAR-Sum (Factually Faithful Extractive Text Summarization via DRL)
Author: Masoud Keikhosravi
Affiliation: Islamic Azad University, Mashhad Branch
Description:
    Proximal Policy Optimization (PPO) Agent with Generalized Advantage 
    Estimation (GAE), Clipped Objective, and Actor-Critic optimization.
=============================================================================
"""

import os
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli


class PPOAgent:
    """
    PPO Agent orchestrating the policy updates, advantage estimation,
    and value function fitting for extractive summarization.
    """
    def __init__(
        self,
        actor_critic_model: nn.Module,
        lr_actor: float = 1e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model = actor_critic_model.to(self.device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Separate optimizers or combined optimizer with parameter grouping
        self.optimizer = optim.Adam([
            {"params": self.model.encoder.parameters(), "lr": lr_actor},
            {"params": self.model.actor_head.parameters(), "lr": lr_actor},
            {"params": self.model.critic_head.parameters(), "lr": lr_critic}
        ])

    def select_action(
        self,
        sentence_embeddings: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples binary selection actions (0 or 1) for each sentence in the document.
        
        Args:
            sentence_embeddings: Tensor of shape (batch_size, num_sentences, hidden_dim)
            mask: Bool tensor indicating valid sentences
            
        Returns:
            actions: Selected binary actions (batch_size, num_sentences)
            log_probs: Log probability of the selected actions
            state_values: Estimated state values from Critic
        """
        self.model.eval()
        with torch.no_grad():
            action_probs, state_values = self.model(sentence_embeddings, mask)
            dist = Bernoulli(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)  # Aggregate log-probs per doc
            
        return actions, log_probs, state_values.squeeze(-1)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE) and discounted returns.
        
        Args:
            rewards: Scalar composite rewards per episode/document (N,)
            values: Critic state value estimates (N,)
            dones: Termination indicators (N,)
            
        Returns:
            advantages: Normalized advantage estimates (N,)
            returns: Target values for Critic (N,)
        """
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0.0
        n_steps = len(rewards)
        
        # In episodic summarization (single-step per document selection), 
        # delta reduces to: reward - value
        for t in reversed(range(n_steps)):
            next_val = 0.0 if t == n_steps - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        # Advantage normalization for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update(
        self,
        memory_states: torch.Tensor,
        memory_actions: torch.Tensor,
        memory_old_log_probs: torch.Tensor,
        memory_returns: torch.Tensor,
        memory_advantages: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Performs PPO policy and value updates across mini-batches.
        """
        self.model.train()
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        n_updates = 0

        dataset_size = memory_states.size(0)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                b_states = memory_states[batch_indices]
                b_actions = memory_actions[batch_indices]
                b_old_log_probs = memory_old_log_probs[batch_indices]
                b_returns = memory_returns[batch_indices]
                b_advantages = memory_advantages[batch_indices]
                b_mask = mask[batch_indices] if mask is not None else None

                # Forward pass
                action_probs, state_values = self.model(b_states, b_mask)
                state_values = state_values.squeeze(-1)

                dist = Bernoulli(action_probs)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Ratio for PPO objective: r(theta) = pi_new / pi_old
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss (MSE)
                value_loss = nn.MSELoss()(state_values, b_returns)

                # Total Loss
                total_loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    - self.entropy_coef * entropy
                )

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_accum += total_loss.item()
                policy_loss_accum += policy_loss.item()
                value_loss_accum += value_loss.item()
                entropy_accum += entropy.item()
                n_updates += 1

        return {
            "loss_total": total_loss_accum / max(1, n_updates),
            "loss_policy": policy_loss_accum / max(1, n_updates),
            "loss_value": value_loss_accum / max(1, n_updates),
            "entropy": entropy_accum / max(1, n_updates)
        }

    def save_checkpoint(self, path: str):
        """Saves model weights and optimizer state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: str):
        """Loads model weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
