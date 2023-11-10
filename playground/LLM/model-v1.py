import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from typing_extensions import Self
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    max_seq_length = 512

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "test": dict(n_layer=4, n_head=4, n_embd=512),
    "tiny": dict(n_layer=8, n_head=8, n_embd=1024),
    "small": dict(n_layer=16, n_head=16, n_embd=2048),
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class RMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope(seq_len: int, dim: int, dtype: torch.dtype = torch.float, device: torch.device = device,
               base: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, seq_len, 2, dtype=dtype, device=device) / dim))  # [dim//2]
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)  # [seq_len]
    freqs_idx = torch.outer(seq_idx, freqs).float()  # [seq_len, dim//2]
    sin_cache = torch.sin(freqs_idx)  # [seq_len, dim//2]
    cos_cache = torch.cos(freqs_idx)  # [seq_len, dim//2]
    rope_cache = torch.stack([sin_cache, cos_cache], dim=-1)  # [seq_len, dim//2, 2]
    return rope_cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    bs, length, head, dim = x.shape
    sin_cache = rope_cache[:, :, 0]  # [seq_len, dim//2]
    cos_cache = rope_cache[:, :, 1]  # [seq_len, dim//2]
    assert length == sin_cache.shape[0]
    assert (dim // 2) == sin_cache.shape[1]
    sin_cache = sin_cache.reshape(1, length, 1, -1)  # [1, seq_len, 1, dim//2]
    cos_cache = cos_cache.reshape(1, length, 1, -1)  # [1, seq_len, 1, dim//2]
    x = x.reshape(bs, length, head, -1, 2)  # [bs, seq_len, head, dim//2, 2]
    q0, q1 = x[..., 0], x[..., 1]  # [bs, seq_len, head, dim//2]
    x_out = torch.stack(
        [q0 * cos_cache - q1 * sin_cache, q1 * cos_cache + q0 * sin_cache],
        dim=-1,
    )  # [bs, seq_len, head, dim//2, 2]
    x_out = x_out.reshape(bs, length, head, dim)  # [bs, seq_len, head, dim]
    return x_out.type_as(x)


rope = None
mask = None


def build_mask_cache(config: LLaMAConfig) -> MaskCache:
    ones = torch.ones(config.block_size, dtype=torch.bool, device=device)  # [block_size, block_size]
    mask_cache = torch.tril(ones, diagonal=0).unsqueeze(0).unsqueeze(0)  # [1, 1, block_size, block_size]
    return mask_cache


class MLP(torch.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        assert config.block_size // config.n_embd == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        bs, length, emb = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        head_size = emb // self.n_head
        k = k.view(bs, length, self.n_head, head_size)  # [bs, length, n_head, head_size]
        q = q.view(bs, length, self.n_head, head_size)  # [bs, length, n_head, head_size]
        v = v.view(bs, length, self.n_head, head_size)  # [bs, length, n_head, head_size]

        k = apply_rope(k, rope)
        q = apply_rope(q, rope)

        k = k.transpose(-2, -3)  # [bs, n_head, length , head_size]
        q = q.transpose(-2, -3)  # [bs, n_head, length , head_size]
        v = v.transpose(-2, -3)  # [bs, n_head, length , head_size]

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask[:, :, :length, :length] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(bs, length, self.n_embd)
        y = self.c_proj(y)  # [bs, length, n_embd]
        return y


class Block(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res1 = x
        x = self.rms_1(x)
        x = self.attn(x) + x_res1
        x_res2 = x
        x = self.rms_2(x)
        x = self.mlp(x) + x_res2
        return x


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx : torch.Tensor):
        pass