import torch
import torch.nn as nn
from d2l import torch as d2l
class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
