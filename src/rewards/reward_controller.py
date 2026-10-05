"""
=============================================================================
Module: reward_controller.py
Project: BAR-Sum (Multi-Objective Extractive Text Summarization via DRL)
Author: Masoud Keikhosravi
Affiliation: Islamic Azad University, Mashhad Branch
Description:
    Bounded Adaptive Inverse-Variance Reward Controller. Dynamically balances 
    ROUGE, Faithfulness (NLI), Redundancy, and Fluency reward signals while 
    strictly enforcing empirical bounds [w_min=0.10, w_max=0.45] with sum=1.0.
=============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple


class BoundedAdaptiveRewardController:
    """
    Adaptive Reward Controller with Inverse-Variance weighting and
    hard box-constraints [w_min, w_max] projected via iterative clipping.
    """
    def __init__(
        self,
        reward_names: List[str] = None,
        window_size: int = 100,
        w_min: float = 0.10,
        w_max: float = 0.45,
        eps: float = 1e-4
    ):
        """
        Args:
            reward_names: List of reward components (default: ROUGE, Faith, Redundancy, Fluency).
            window_size: Size of sliding window (tau) for moving variance calculation.
            w_min: Minimum allowable weight per component.
            w_max: Maximum allowable weight per component.
            eps: Numerical stability constant.
        """
        if reward_names is None:
            self.reward_names = ["rouge", "faithfulness", "redundancy", "fluency"]
        else:
            self.reward_names = reward_names

        self.num_rewards = len(self.reward_names)
        self.window_size = window_size
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.eps = float(eps)

        # Validate theoretical consistency: N * w_min <= 1 <= N * w_max
        assert self.num_rewards * self.w_min <= 1.0 <= self.num_rewards * self.w_max, (
            f"Bounds [{self.w_min}, {self.w_max}] are infeasible for {self.num_rewards} objectives."
        )

        # History buffer for moving window variance
        self.history: Dict[str, List[float]] = {name: [] for name in self.reward_names}

        # Initialize with equal weights: 1/4 = 0.25
        self.current_weights: Dict[str, float] = {
            name: 1.0 / self.num_rewards for name in self.reward_names
        }

    def update_history(self, reward_dict: Dict[str, float]):
        """
        Appends latest episodic rewards into the moving history window.
        """
        for name in self.reward_names:
            val = reward_dict.get(name, 0.0)
            self.history[name].append(float(val))
            if len(self.history[name]) > self.window_size:
                self.history[name].pop(0)

    def _project_bounds(self, weights: np.ndarray, max_iter: int = 50) -> np.ndarray:
        """
        Projects arbitrary weights onto the probability simplex bounded by [w_min, w_max]
        such that sum(weights) == 1.0 and w_min <= weights[i] <= w_max for all i.
        Uses iterative projection algorithm.
        """
        w = np.copy(weights)
        fixed = np.zeros_like(w, dtype=bool)

        for _ in range(max_iter):
            # Normalize non-fixed components
            remaining_sum = 1.0 - np.sum(w[fixed])
            num_free = np.sum(~fixed)

            if num_free == 0:
                break

            current_free_sum = np.sum(w[~fixed])
            if current_free_sum > 0:
                w[~fixed] = w[~fixed] * (remaining_sum / current_free_sum)
            else:
                w[~fixed] = remaining_sum / num_free

            # Check constraint violations
            new_fixed = False
            for i in range(len(w)):
                if not fixed[i]:
                    if w[i] < self.w_min:
                        w[i] = self.w_min
                        fixed[i] = True
                        new_fixed = True
                    elif w[i] > self.w_max:
                        w[i] = self.w_max
                        fixed[i] = True
                        new_fixed = True

            if not new_fixed:
                break

        # Safety clip to avoid float rounding discrepancies
        w = np.clip(w, self.w_min, self.w_max)
        w = w / np.sum(w)
        return w

    def compute_weights(self) -> Dict[str, float]:
        """
        Computes bounded adaptive weights based on inverse moving variance.
        If history window is not yet full, maintains balanced weights.
        """
        # If not enough history, return uniform / default weights
        min_history_len = min(len(self.history[k]) for k in self.reward_names)
        if min_history_len < 5:
            return self.current_weights

        variances = []
        for name in self.reward_names:
            var = np.var(self.history[name])
            variances.append(var)

        variances = np.array(variances, dtype=np.float64)

        # Inverse-variance: ~w_k = 1 / (sigma_k^2 + eps)
        inv_vars = 1.0 / (variances + self.eps)

        # Initial normalization: ^w_k = ~w_k / sum(~w_j)
        raw_weights = inv_vars / np.sum(inv_vars)

        # Enforce bounded constraints: [w_min, w_max] with sum = 1.0
        bounded_weights = self._project_bounds(raw_weights)

        # Update and return dictionary
        self.current_weights = {
            name: float(bounded_weights[i])
            for i, name in enumerate(self.reward_names)
        }
        return self.current_weights

    def get_composite_reward(self, reward_dict: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Updates controller statistics with new rewards, recalculates weights,
        and returns the scalar composite reward: R_total = sum(w_k * R_k).
        
        Args:
            reward_dict: Dictionary mapping component names to reward values.
            
        Returns:
            composite_reward: Weighted sum of reward components.
            weights: Dict of currently active weights applied.
        """
        self.update_history(reward_dict)
        weights = self.compute_weights()

        composite_reward = sum(
            weights[name] * reward_dict.get(name, 0.0)
            for name in self.reward_names
        )

        return float(composite_reward), weights
