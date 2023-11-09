import torch
import torch.nn as nn
from typing import Union, List, TypedDict, Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    pass

# 1. RMSNorm Without Bias
class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_norm = x * torch.rsqrt(x_mean + self.eps)
        return x_norm * self.scale

# 2.