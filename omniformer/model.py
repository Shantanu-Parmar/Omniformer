import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

__all__ = ["Omniformer"]

# -- Gated Residual Block --
class GatedResidual(nn.Module):
    """
    Learnable gated residual connection: x + α * F(x)
    """
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, sublayer_out):
        return x + self.alpha * sublayer_out


# -- Custom Transformer Layer with per-head injection --
class CustomTransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(model_dim, model_dim * 4)
        self.linear2 = nn.Linear(model_dim * 4, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.res1 = GatedResidual(model_dim)
        self.res2 = GatedResidual(model_dim)

    def forward(self, x, external_weights=None):
        B, S, D = x.size()
        out = []

        for i in range(B):
            xi = x[i:i+1]  # [1, S, D]

            if external_weights:
                self.inject_weights(external_weights, i)

            attn_out, _ = self.attn(xi, xi, xi)
            xi = self.norm1(self.res1(xi, attn_out))

            ff_out = self.linear2(F.relu(self.linear1(xi)))
            xi = self.norm2(self.res2(xi, ff_out))

            out.append(xi)

        return torch.cat(out, dim=0)

    def inject_weights(self, weights, index):
        # This avoids autograd errors by detaching the parameter .data
        self.attn.in_proj_weight.data.copy_(weights['attn_proj_weight'][index])
        self.attn.in_proj_bias.data.copy_(weights['attn_proj_bias'][index])
        self.linear1.weight.data.copy_(weights['ff1_weight'][index])
        self.linear1.bias.data.copy_(weights['ff1_bias'][index])
        self.linear2.weight.data.copy_(weights['ff2_weight'][index])
        self.linear2.bias.data.copy_(weights['ff2_bias'][index])


# -- HyperNet: Context → Transformer Layer Weights --
class HyperNet(nn.Module):
    """
    Generates per-sample weights for a Transformer layer based on a context vector.
    """
    def __init__(self, context_dim, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        total_dim = (
            3 * model_dim * model_dim +  # QKV weights
            3 * model_dim +              # QKV bias
            4 * model_dim * model_dim +  # FF1 weight
            4 * model_dim +              # FF1 bias
            model_dim * 4 * model_dim +  # FF2 weight
            model_dim                    # FF2 bias
        )
        self.fc = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.ReLU(),
            nn.Linear(512, total_dim)
        )

    def forward(self, context_vector):
        batch_size = context_vector.shape[0]
        flat = self.fc(context_vector)
        return self.unflatten(flat, batch_size)

    def unflatten(self, flat, batch_size):
        idx = 0
        out = {}

        def take(name, shape):
            nonlocal idx
            numel = torch.prod(torch.tensor(shape)).item()
            out[name] = flat[:, idx:idx+numel].view(batch_size, *shape)
            idx += numel

        d = self.model_dim
        take('attn_proj_weight', [3 * d, d])
        take('attn_proj_bias', [3 * d])
        take('ff1_weight', [4 * d, d])
        take('ff1_bias', [4 * d])
        take('ff2_weight', [d, 4 * d])
        take('ff2_bias', [d])

        return out


# -- Learned Positional Encoding --
class LearnedPositionalEncoding(nn.Module):
    """
    Learnable positional encoding matrix added to input sequence.
    """
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, model_dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# -- Meta-Optimizer Stub --
class MetaOptimizer(nn.Module):
    """
    Placeholder for meta-optimization (future: optimize HyperNet based on loss or gradients).
    """
    def __init__(self):
        super().__init__()

    def forward(self, hypernets, loss):
        return


# -- Omniformer Model --
class Omniformer(nn.Module):
    """
    Full Omniformer model with:
    - Per-layer transformer blocks
    - Per-layer HyperNets (context-aware dynamic weights)
    - Learnable positional encoding
    - Optional TensorBoard logging
    """
    def __init__(self, input_dim, context_dim, model_dim=128, num_layers=6, num_heads=4, seq_len=100, enable_logging=True, device="cpu"):
        super().__init__()
        self.model_dim = model_dim
        self.device = device
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_enc = LearnedPositionalEncoding(seq_len, model_dim)

        self.layers = nn.ModuleList()
        self.hypernets = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(CustomTransformerLayer(model_dim, num_heads))
            self.hypernets.append(HyperNet(context_dim, model_dim, num_heads))

        self.meta_optimizer = MetaOptimizer()
        self.output_head = nn.Linear(model_dim, 1)

        self.global_step = 0
        if enable_logging:
            logdir = "runs/Omniformer"
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

    def forward(self, x, context_vector):
        x = self.input_proj(x)
        x = self.pos_enc(x)

        for i, (layer, hypernet) in enumerate(zip(self.layers, self.hypernets)):
            weights = hypernet(context_vector)
            x = layer(x, external_weights=weights)

            if self.writer is not None and self.writer.log_dir:
                self.writer.add_scalar(f'layer_{i}/mean_activation', x.mean().item(), self.global_step)

        output = self.output_head(x[:, -1, :])
        return output

    def log_loss(self, loss):
        if self.writer:
            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.global_step += 1

    def save_checkpoint(self, path="checkpoint.pt"):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path="checkpoint.pt", device=None):
        device = device or self.device
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Omniformer] Checkpoint not found: {path}")
        try:
            self.load_state_dict(torch.load(path, map_location=device))
        except Exception as e:
            raise RuntimeError(f"[Omniformer] Failed to load checkpoint: {e}")

    def forward_single(self, x_seq, context_vector):
        """
        Inference on a single (1, seq_len, input_dim) sample.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x_seq.unsqueeze(0), context_vector.unsqueeze(0)).squeeze()
