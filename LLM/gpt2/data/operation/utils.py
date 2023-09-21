import torch
from torch import Tensor


def remove_tokens(ids: Tensor, token: int):
    indices = torch.nonzero(ids == token)
    return ids[ids != token]
