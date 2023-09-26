from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self
from dataclasses import dataclass


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LLaMA(nn.Module):
    pass


class Block(nn.Module):
    pass


class CausalSelfAttention(nn.Module):
    pass


class MLP(nn.Module):
    pass


class RMSNorm(nn.Module):
    pass


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype = torch.float, device: torch.device = "cpu",
                     base: int = 1000) -> RoPECache:
    #
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)
    idx_theta = torch.outer(seq_idx, theta).float() # (seq_len, theta)
    cache =torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # (seq_len, theta, 2)
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache : RoPECache) -> torch.Tensor:
    # x : (B, T, self.n_head, head_size)
    # rope_cache : (T, )
    T = x.size()
    rope_cache = rope_cache[:T]
    # cast
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)