import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable, Any


def default_timestamp_score_fn(
    pred_probs: torch.Tensor,          # (1, num_aux + 1) softmax probs
    truth_active: int                   # 0 = background, 1 = at least one aux active
) -> float:
    """
    Default scoring: binary match between predicted class and truth.
    Predicted background if argmax is on background class.
    Easy to replace with soft scoring, top-k, etc.
    """
    pred_class = pred_probs.argmax(dim=-1).item()          # 0 = background, 1.. = aux
    pred_is_background = (pred_class == 0)
    truth_is_background = (truth_active == 0)

    return 1.0 if pred_is_background == truth_is_background else 0.0


class IterationLSTM(nn.Module):
    """
    Core recurrent model — processes one timestamp at a time.
    Outputs probability distribution over aux families + background.
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_lstm_layers: int = 3,
        num_aux_families: int = 12,
        dropout: float = 0.1,
        score_fn: Callable = default_timestamp_score_fn,  # injectable scoring
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_aux_families = num_aux_families
        self.num_classes = num_aux_families + 1  # + background
        self.score_fn = score_fn

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=False,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.head = nn.Linear(hidden_dim, self.num_classes)

    def forward(
        self,
        x_t: torch.Tensor,                          # (input_dim,) or (1, input_dim)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0).unsqueeze(0)
        elif x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        out, hidden = self.lstm(x_t, hidden)
        logits = self.head(out[-1])                     # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        return probs, hidden


class EpochOperator:
    """
    Manages one epoch (~50 s = many iterations).
    Collects per-timestamp accuracies, computes poll score (mean acc),
    selects best hidden state to propagate.
    """
    def __init__(
        self,
        success_threshold: float = 0.70,                # for binary success/failure view
    ):
        self.success_threshold = success_threshold
        self.timestamp_accuracies: List[float] = []     # [0.0 or 1.0] per timestamp
        self.timestamp_hiddens: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.poll_score: Optional[float] = None         # mean accuracy over epoch
        self.best_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.best_accuracy: float = -1.0

    def add_timestamp_result(
        self,
        accuracy: float,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        self.timestamp_accuracies.append(accuracy)
        self.timestamp_hiddens.append(hidden)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_hidden = hidden

    def compute_poll(self) -> float:
        if not self.timestamp_accuracies:
            self.poll_score = 0.0
        else:
            self.poll_score = float(np.mean(self.timestamp_accuracies))
        return self.poll_score

    def get_summary(self) -> Dict[str, Any]:
        return {
            "poll_score": self.poll_score,
            "success_rate": sum(1 for a in self.timestamp_accuracies if a >= self.success_threshold) / len(self.timestamp_accuracies) if self.timestamp_accuracies else 0.0,
            "best_accuracy": self.best_accuracy,
            "num_timestamps": len(self.timestamp_accuracies),
        }


class RewardOperator:
    """
    Looks at poll scores and trends across epochs.
    Rewards stable/upward trends, punishes degradation/stagnation.
    Controls learning rate (and potentially hidden perturbation later).
    """
    def __init__(
        self,
        initial_lr: float = 3e-4,
        reward_every_n_epochs: int = 5,
        trend_window: int = 5,
        min_improvement: float = 0.01,
        patience: int = 8,
    ):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.reward_frequency = reward_every_n_epochs
        self.trend_window = trend_window
        self.min_improvement = min_improvement
        self.patience = patience
        self.patience_counter = 0
        self.best_poll = -np.inf
        self.poll_history: List[float] = []
        self.optimizer: Optional[optim.Optimizer] = None

    def register_optimizer(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer
        for pg in optimizer.param_groups:
            pg['lr'] = self.current_lr

    def add_epoch_poll(self, poll_score: float):
        self.poll_history.append(poll_score)

        if poll_score > self.best_poll + self.min_improvement:
            self.best_poll = poll_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def should_act(self, epoch_idx: int) -> bool:
        return (epoch_idx + 1) % self.reward_frequency == 0

    def compute_trend_and_reward(self) -> Dict[str, Any]:
        if len(self.poll_history) < self.trend_window:
            return {"action": "none", "lr_multiplier": 1.0}

        recent = np.array(self.poll_history[-self.trend_window:])
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        recent_mean = recent.mean()
        improvement = recent_mean - self.poll_history[0] if len(self.poll_history) > 1 else 0

        reward = 0.0
        if slope > 0.015:
            reward += 1.2
        elif slope > 0.003:
            reward += 0.4
        elif slope < -0.008:
            reward -= 1.1

        if recent_mean > self.best_poll + 0.005:
            reward += 0.8

        if self.patience_counter >= self.patience:
            reward -= 1.5

        if reward > 0.8:
            action = "increase_lr"
            mult = 1.10
        elif reward < -0.6:
            action = "decrease_lr"
            mult = 0.70
        else:
            action = "hold"
            mult = 1.0

        return {
            "reward": reward,
            "action": action,
            "lr_multiplier": mult,
            "slope": slope,
            "recent_mean": recent_mean,
        }

    def apply_adjustment(self):
        signal = self.compute_trend_and_reward()
        mult = signal["lr_multiplier"]

        self.current_lr *= mult
        if self.optimizer is not None:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.current_lr
