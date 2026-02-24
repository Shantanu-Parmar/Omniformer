# operators/epoch_manager.py
"""
Manages one epoch (N iterations).
Collects binary scores and hidden states.
Computes poll score = true/total iterations.
Builds weighted pool of True hiddens for next epoch.
Applies randomness ramp + reward override.
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from config import Config
from logging.structured_logger import StructuredLogger


class EpochManager:
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
    ):
        self.config = config
        self.logger = logger

        self.iteration_scores: List[int] = []           # 1 or 0 per iteration
        self.iteration_hiddens: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.true_hiddens_pool: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.true_streaks: List[int] = []               # streak length ending at each True iteration

        self.current_randomness_p: float = config.initial_randomness_p
        self.ramp_decay_rate: float = config.ramp_decay_rate
        self.min_randomness_p: float = config.min_randomness_p

        self.previous_true_pool: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self.iteration_count = 0
        self.current_streak = 0

    def add_iteration_result(
        self,
        score: int,
        final_hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        self.iteration_scores.append(score)
        self.iteration_hiddens.append(final_hidden)

        if score == 1:
            self.current_streak += 1
            self.true_hiddens_pool.append(final_hidden)
            self.true_streaks.append(self.current_streak)
        else:
            self.current_streak = 0

        self.iteration_count += 1

        # Partial logging (FULL only)
        if self.logger.log_level == "FULL":
            self.logger.full("epoch", {
                "iteration": self.iteration_count,
                "score": score,
                "streak": self.current_streak,
                "hidden_norm": final_hidden[0].norm().item() if final_hidden else None,
            })

    def is_complete(self) -> bool:
        return self.iteration_count >= self.config.epoch_iterations

    def finalize_and_prepare_next(self, reward_signal: Dict[str, Any]) -> Tuple[float, List[Tuple[torch.Tensor, torch.Tensor]], float]:
        """
        End of epoch:
        - Compute poll
        - Decay randomness
        - Apply reward override
        - Build weighted pool for next epoch
        - Log
        - Return poll, next pool, current p
        """
        true_count = sum(self.iteration_scores)
        poll_score = true_count / self.iteration_count if self.iteration_count else 0.0

        # Decay randomness
        self.current_randomness_p *= self.ramp_decay_rate
        self.current_randomness_p = max(self.current_randomness_p, self.min_randomness_p)

        # Reward override (when reward computed or negative)
        reward_val = reward_signal.get("reward", 0)
        if reward_val < -0.5:
            self.current_randomness_p = max(self.current_randomness_p, 0.4)
            self.logger.work("ramp_override", {"reason": "negative_reward", "new_p": self.current_randomness_p})

        # Logging
        log_data = {
            "poll_score": poll_score,
            "true_total": f"{true_count}/{self.iteration_count}",
            "randomness_p": self.current_randomness_p,
            "true_hiddens_count": len(self.true_hiddens_pool),
        }

        if self.logger.log_level in ["WORK", "FULL"]:
            log_data["streaks"] = self.true_streaks
            if self.true_hiddens_pool:
                norms = [h[0].norm().item() for h in self.true_hiddens_pool]
                log_data["avg_true_hidden_norm"] = float(np.mean(norms))

        self.logger.minimal("epoch_end", {"poll": poll_score, "true_total": f"{true_count}/{self.iteration_count}"})
        if self.logger.log_level in ["WORK", "FULL"]:
            self.logger.work("epoch_end", log_data)

        # Build pool for next epoch
        next_pool = self.true_hiddens_pool[:]

        if not next_pool:
            # Zero True → reuse previous + noise
            next_pool = self.previous_true_pool[:]
            if next_pool:
                noise_std = 0.005
                for h, c in next_pool:
                    h.add_(torch.randn_like(h) * noise_std)
                    c.add_(torch.randn_like(c) * noise_std)

        self.previous_true_pool = next_pool  # save for next fallback

        # Clear current epoch data
        self.iteration_scores = []
        self.iteration_hiddens = []
        self.true_hiddens_pool = []
        self.true_streaks = []
        self.iteration_count = 0
        self.current_streak = 0

        return poll_score, next_pool, self.current_randomness_p

    def get_weighted_random_hidden(self, pool: List[Tuple[torch.Tensor, torch.Tensor]], streaks: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted random sample from True pool.
        Weight = streak length (longer = higher probability)
        """
        if not pool:
            raise ValueError("No True hiddens in pool")

        weights = [s + 1 for s in streaks]  # +1 to avoid zero
        total = sum(weights)
        probs = [w / total for w in weights]

        idx = np.random.choice(len(pool), p=probs)
        return pool[idx]