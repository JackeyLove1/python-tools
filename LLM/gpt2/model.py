import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class Config:
    block_size: int = field(default=1024)
    vocab_size: int = field(
        default=50304)  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = field(default=12)
    n_head: int = field(default=12)
    n_embd: int = field(default=768)
    dropout: float = field(default=0.0)
    bias: bool = field(default=True)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = 1e-5

    def forward(self, input: Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "config.n_embd % config.n_head != 0"
        # q, k, v
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regulation
        self.attn_dropout = nn.Dropout(p=config.dropout)
        self.resid_dropout = nn.Dropout(p=config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            raise EnvironmentError("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: Tensor):
        B, T, C = x.size()
