# implement of RoPE
import torch
from torch import nn
from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention

def ptensor(name:str, t : torch.Tensor):
    print("name:", name, " value:", t, " size:", t.size())

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
       super().__init__()
       self.base = base
       self.d = d
       self.cos_cached = None
       self.sin_cached = None

    def build_cache(self, x : torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        ptensor("x", x)
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        ptensor("theta", theta)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        ptensor("seq_idx:", seq_idx)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        ptensor("idx_theta:", idx_theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta],dim=1)
        ptensor("idx_theta2", idx_theta2)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        ptensor("cos:", self.cos_cached)
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
        ptensor("sin:", self.sin_cached)

    def _neg_half(self,x : torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:,:,:,d_2:], x[:,:,:,:d_2]], dim=1)

    def forward(self, x:torch.Tensor):
        self.build_cache(x)
        x_rope, x_pass = x[...,:self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[:x.shape[0]]])
        return torch.cat((x_rope, x_pass), dim=-1)


test = torch.arange(3)
r = RotaryPositionalEmbeddings(1)
r.build_cache(test)
