import torch
import torch.nn as nn
from typing import Tuple

torch.manual_seed(42)

freq_g = None


def precompute_freqs_cis_v1(dim: int, length: int, constant: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[:dim // 2] / dim))  # [dim//2]
    t = torch.arange(length, dtype=freqs.dtype, device=freqs.device)  # [length]
    # freqs = torch.outer(t, freqs).float()  # [length, dim//2]
    freqs = torch.einsum('i,j->ij', t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # [length, dim//2]
    return freqs_cis.reshape(1, length, 1, dim // 2)  # [1, length, 1, dim//2]


def apply_rotary_emb_v1(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    :param xq: [bs, length, head, dim]
    :param xk: [bs, length, head, dim]
    :param freqs_cis: [1, length, 1, dim//2]
    :return: [bs, length, dim]
    """
    bs, length, head, dim = xq.shape
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [bs, length,head, dim//2, 2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xq.shape[:-1], -1, 2))  # [bs, length,head, dim//2, 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis)  # [bs, length, head, dim//2, 2, 2]
    xq_out = xq_out.reshape(bs, length, head, -1)  # [bs, length, head, dim]
    xk_out = torch.view_as_real(xk_ * freqs_cis)  # [bs, length, head, dim//2, 2, 2]
    xk_out = xk_out.reshape(bs, length, head, -1)  # [bs, length, head, dim]
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, constant: float = 10000.0):
    '''
    计算cos和sin的值，cos值在实部，sin值在虚部，类似于 cosx+j*sinx
    :param dim: q,k,v的最后一维，一般为emb_dim/head_num
    :param end: 句长length
    :param constant： 这里指10000
    :return:
    复数计算 torch.polar(a, t)输出， a*(cos(t)+j*sin(t))
    '''
    # freqs: 计算 1/(10000^(2i/d) )，将结果作为参数theta
    # 形式化为 [theta_0, theta_1, ..., theta_(d/2-1)]
    freqs = 1.0 / (constant ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [d/2]

    # 计算m
    t = torch.arange(end, device=freqs.device)  # [length]
    # 计算m*theta
    freqs = torch.outer(t, freqs).float()  # [length, d/2]
    # freqs形式化为 [m*theta_0, m*theta_1, ..., m*theta_(d/2-1)],其中 m=0,1,...,length-1

    # 计算cos(m*theta)+j*sin(m*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # freqs_cis: [cos(m*theta_0)+j*sin(m*theta_0),  cos(m*theta_1)+j*sin(m*theta_1),), ..., cos(m*theta_(d/2-1))+j*sin(m*theta_(d/2-1))]
    # 其中j为虚数单位， m=0,1,...,length-1
    return freqs_cis  # [length, d/2]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]  # (1, length, 1, d/2)
    return freqs_cis.view(*shape)  # [1, length, 1, d/2]


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, ):
    # 先将xq维度变为[bs, length, head,  d/2, 2], 利用torch.view_as_complex转变为复数
    # xq:[q0, q1, .., q(d-1)] 转变为 xq_: [q0+j*q1, q2+j*q3, ..., q(d-2)+j*q(d-1)]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [bs, length, head, d/2]
    # 同样的，xk_:[k0+j*k1, k2+j*k3, ..., k(d-2)+j*k(d-1)]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    print("xq_ shape:", xq_.shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # [1, length, 1, d/2]
    global freq_g
    freq_g = freqs_cis
    print("freqs_cis shape:", freqs_cis.shape)
    # 下式xq_ * freqs_cis形式化输出，以第一个为例, 如下
    # (q0+j*q1)(cos(m*theta_0)+j*sin(m*theta_0)) = q0*cos(m*theta_0)-q1*sin(m*theta_0) + j*(q1*cos(m*theta_0)+q0*sin(m*theta_0))
    # 上式的实部为q0*cos(m*theta_0)-q1*sin(m*theta_0)，虚部为q1*cos(m*theta_0)+q0*sin(m*theta_0)
    # 然后通过torch.view_as_real函数，取出实部和虚部，维度由[bs, length, head, d/2]变为[bs, length, head, d/2, 2]，最后一维放实部与虚部
    # 最后经flatten函数将维度拉平，即[bs, length, head, d]
    # 此时xq_out形式化为 [实部0，虚部0，实部1，虚部1，..., 实部(d/2-1), 虚部(d/2-1)]
    print("torch.view_as_real(xq_ * freqs_cis) shape:", torch.view_as_real(xq_ * freqs_cis).shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # [bs, length, head, d]
    print("xq_out shape:", xq_out.shape)
    # 即为新生成的q

    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def build_rope_cache_v1(length: int, dim: int, constant: float = 10000.0) -> torch.Tensor:
    freqs = 1 / (constant ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))  # [dim//2]
    idx = torch.arange(length, dtype=freqs.dtype, device=freqs.device)  # [length]
    freqs_idx = torch.einsum('i,j->ij', idx, freqs)  # [length, dim//2]
    sin_cache = torch.sin(freqs_idx)  # [length, dim//2]
    cos_cache = torch.cos(freqs_idx)  # [length, dim//2]
    return torch.stack([sin_cache, cos_cache], dim=-1)  # [length, dim//2, 2]


def apply_rope_v1(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    :param x: [bs, length, head, dim]
    :param rope_cache: [length, dim//2, 2]
    :return:
    """
    bs, length, head, dim = x.shape
    sin_cache = rope_cache[:, :, 0]  # [length, dim//2]
    cos_cache = rope_cache[:, :, 1]  # [length, dim//2]
    assert (dim // 2) == sin_cache.shape[-1]
    sin_cache = sin_cache.reshape(1, length, 1, -1)  # [1, length, 1, dim//2]
    cos_cache = cos_cache.reshape(1, length, 1, -1)  # [1, length, 1, dim//2]
    x = x.reshape(bs, length, head, -1, 2)  # [bs, length, head, dim//2, 2]
    q0, q1 = x[..., 0], x[..., 1]  # [bs, length, head, dim//2]
    x_out = torch.stack(
        [q0 * cos_cache - q1 * sin_cache,
         q1 * cos_cache + q0 * sin_cache],
        dim=-1,
    )  # [bs, length, head, dim//2, 2]
    x_out = x_out.reshape(bs, length, head, dim)  # [bs, length, head, dim]
    return x_out.type_as(x)


if __name__ == '__main__':
    # (bs, length, head, d)
    bs, length, head, d = 2, 10, 12, 32
    q = torch.randn((2, 10, 12, 32))
    k = torch.randn((2, 10, 12, 32))
    v = torch.randn((2, 10, 12, 32))
    freqs_cis_v1 = precompute_freqs_cis_v1(dim=32, length=10)
    q_new_v1, k_new_v1 = apply_rotary_emb_v1(xq=q, xk=k, freqs_cis=freqs_cis_v1)
    freqs_cis = precompute_freqs_cis(dim=32, end=10, constant=10000.0)
    freqs_cis_v2 = freqs_cis.view(1, 10, 1, -1)
    q_new, k_new = apply_rotary_emb(xq=q, xk=k, freqs_cis=freqs_cis)
    eps = 1e-6
    print(torch.allclose(freqs_cis_v1, freqs_cis_v2, atol=eps))
    print(torch.allclose(q_new, q_new_v1, atol=eps))
    print(torch.allclose(k_new, k_new_v1, atol=eps))

    rope_cache = build_rope_cache_v1(length, d)
    xq_out, xk_out = apply_rope_v1(q, rope_cache), apply_rope_v1(k, rope_cache)
    print(torch.allclose(xq_out, q_new_v1, atol=eps))
    print(torch.allclose(xk_out, k_new_v1, atol=eps))
