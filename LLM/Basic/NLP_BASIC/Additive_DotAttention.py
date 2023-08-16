import math

import d2l.torch
import torch
import torch.nn as nn
from torch import Tensor


def sequence_mask(X: Tensor, valid_len: Tensor, value: int = 0):
    '''
    :param X: 3D tensor (batch_size, seq_len, emb_size)
    :param valid_len: 1D or 2D tensor (batch_size,)
    :param value: mask_value, seq2seq = 0, softmax = 1e-9
    :return: (batch_size, seq_len)
    '''
    max_len = X.size()[1]
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)  # (seq_len, )
    mask = mask.unsqueeze(0)  # (batch_size, seq_len)
    valid_len = valid_len.unsqueeze(1)  # (batch_size, seq_len)
    mask = mask < valid_len  # (batch_size, seq_len)
    X[~mask] = value
    return X
# test sequence mask
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
valid_len = torch.tensor([1, 2])
print(sequence_mask(X, valid_len))

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_seq2seq_decoder`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred: Tensor, label : Tensor, valid_len : Tensor):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def masked_softmask(X:Tensor, valid_lens:Tensor):
    '''
    :param X: 3D (batch_size, seq_len, emb_size)
    :param valid_lens: 1D or 2D (batch_size)
    :return:
    '''
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.view(-1)
        X = d2l.torch.sequence_mask(X.view(-1, shape[-1]), valid_len, value=1e-6)
        return nn.functional.softmax(X.reshape(-1), dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, value_size, num_hiddens, dropout, **kwargs):
        super().__init__()
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(value_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.w_q(queries), self.w_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmask(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductionAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.size()[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmask(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)