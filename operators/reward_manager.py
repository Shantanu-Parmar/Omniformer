# operators/reward_manager.py
"""
Highest-level controller: analyzes poll score trends across epochs,
computes reward, decides lr changes and ramp overrides.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch.optim as optim
from config import Config
from logging.structured_logger import StructuredLogger


class RewardManager:
    def __init__(
        self,
        config: Config,
        logger: StructuredLogger,
    ):
        self.config = config
        self.logger = logger

        self.poll_history: List[float] = []
        self.best_poll: float = -np.inf
        self.patience_counter: int = 0
        self.plateau_streak: int = 0               # consecutive M-epoch runs without improvement
        self.current_lr: float = config.initial_lr
        self.optimizer: Optional[optim.Optimizer] = None

        self.consecutive_no_improvement = 0        # for big plateau detection

    def register_optimizer(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer
        for pg in optimizer.param_groups:
            pg['lr'] = self.current_lr

    def add_epoch_poll(self, poll_score: float):
        self.poll_history.append(poll_score)

        if poll_score > self.best_poll + self.config.min_improvement:
            self.best_poll = poll_score
            self.patience_counter = 0
            self.consecutive_no_improvement = 0
        else:
            self.patience_counter += 1
            self.consecutive_no_improvement += 1

    def should_compute_reward(self, epoch_idx: int) -> bool:
        # Reward computed every M epochs
        return (epoch_idx + 1) % self.config.reward_every_n_epochs == 0

    def compute_reward(self) -> Dict[str, Any]:
        if len(self.poll_history) < self.config.trend_window:
            return {"reward": 0.0, "action": "hold", "lr_mult": 1.0, "ramp_boost": False}

        recent = np.array(self.poll_history[-self.config.trend_window:])
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        recent_mean = recent.mean()
        volatility = np.std(recent)

        reward = 0.0

        # Heavy weight on slope (recent improvement)
        if slope > 0.015:
            reward += 2.0
        elif slope > 0.005:
            reward += 1.0
        elif slope < -0.01:
            reward -= 1.8

        # Level bonus
        if recent_mean > 0.7:
            reward += 0.8
        elif recent_mean > self.best_poll + 0.01:
            reward += 0.6

        # Plateau penalty
        if self.patience_counter >= self.config.patience:
            reward -= 1.5

        # Volatility penalty only when ramp is active (exploration mode)
        if self.config.ramp_active and volatility > 0.12:
            reward -= 0.7

        # Determine action
        action = "hold"
        lr_mult = 1.0
        ramp_boost = False

        if reward > 0.8:
            action = "increase_lr"
            lr_mult = 1.10
        elif reward < -0.6:
            action = "decrease_lr"
            lr_mult = 0.70
            ramp_boost = True  # signal to epoch manager to boost randomness

        return {
            "reward": reward,
            "action": action,
            "lr_mult": lr_mult,
            "ramp_boost": ramp_boost,
            "slope": slope,
            "recent_mean": recent_mean,
            "volatility": volatility,
            "patience_counter": self.patience_counter,
            "plateau_streak": self.consecutive_no_improvement,
        }

    def apply_adjustment(self):
        signal = self.compute_reward()

        # Apply lr change
        self.current_lr *= signal["lr_mult"]
        if self.optimizer:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.current_lr

        # Big plateau check: recreate optimizer if too many consecutive no-improvement M-runs
        if self.consecutive_no_improvement >= self.config.big_plateau_threshold:
            if self.optimizer:
                # Recreate optimizer (resets momentum etc.)
                self.optimizer = optim.Adam(self.optimizer.param_groups[0]['params'], lr=self.current_lr)
                self.logger.work("reward", {"action": "optimizer_recreated", "reason": "big_plateau", "consecutive": self.consecutive_no_improvement})
            self.consecutive_no_improvement = 0  # reset counter

        # Logging
        log_data = {
            "reward": signal["reward"],
            "action": signal["action"],
            "new_lr": self.current_lr,
            "ramp_boost": signal["ramp_boost"],
            "slope": signal["slope"],
        }

        if self.logger.log_level in ["WORK", "FULL"]:
            log_data.update({
                "recent_mean": signal["recent_mean"],
                "volatility": signal["volatility"],
                "patience": signal["patience_counter"],
                "plateau_streak": signal["plateau_streak"],
            })

        self.logger.work("reward_adjust", log_data)

        # Return signal so epoch manager can apply ramp boost if needed
        return signal