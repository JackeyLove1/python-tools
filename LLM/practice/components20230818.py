import torch
import torch.nn as nn
import math
import copy
import random
from torch import Tensor
import torch.nn.functional as F
from d2l import torch as d2l


def sequence_mask(X: Tensor, valid_len: Tensor, value: float = 0):
    '''
    :param X: (batch_size, n_step, n_hiddens)
    :param valid_len: (batch_size,)
    :return:(batch_size, n_step, n_hiddens)
    '''
    max_len = X.size(-1)
    valid_len = valid_len.view(-1, 1, 1)
    mask = torch.arange((max_len), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mask = ~(mask < valid_len)  # broadcast
    return X.masked_fill(mask, value)


def masked_softmax(X: Tensor, valid_len: Tensor):
    '''
    :param X: (batch_size, n_step, n_hiddens)
    :param valid_len: (batch_size,)
    :return:(batch_size, n_step, n_hiddens)
    example:
        masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
        tensor([[[0.5980, 0.4020, 0.0000, 0.0000],[0.5548, 0.4452, 0.0000, 0.0000]],
        [[0.3716, 0.3926, 0.2358, 0.0000],[0.3455, 0.3337, 0.3208, 0.0000]]])
    '''
    if valid_len is None:
        return F.softmax(X, dim=-1)
    x_masked = sequence_mask(X, valid_len, value=-1e6)
    return nn.functional.softmax(x_masked, dim=-1)


'''
def test_masked_softmax():
    input_tensor = torch.ones(2, 2, 4)
    mask_tensor = torch.tensor([2, 3])
    result = masked_softmax(input_tensor, mask_tensor)
    expected = d2l.masked_softmax(input_tensor, mask_tensor)
    assert torch.allclose(result, expected)
'''


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, valid_lens=None):
        d_k = queries.size()[0]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d_k)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X: Tensor, num_headers: int):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,num_hiddens/num_heads)
    batch_size, kvs, num_hiddens = X.size()  # (batch_size，查询或者“键－值”对的个数，num_hiddens)
    assert num_hiddens % num_headers == 0
    X = X.reshape(batch_size, kvs, num_headers, -1)  # (batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)  # (batch_size， num_heads, 查询或者“键－值”对的个数，num_hiddens/num_heads)
    return X.reshape(-1, X.size()[2], X.size()[3])  # (batch_size * num_heads, 查询或者“键－值”对的个数，num_hiddens/num_heads)


def transpose_out(X: Tensor, num_headers: int):
    """逆转transpose_qkv函数的操作"""
    X = X.view(-1, num_headers, X.size()[1], X.size()[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.size()[0], X.size()[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_headers = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    # queries，keys，values的形状:
    # (batch_size，查询或者“键－值”对的个数，num_hiddens)
    # valid_lens　的形状:
    # (batch_size，)或(batch_size，查询的个数)
    # 经过变换后，输出的queries，keys，values　的形状:
    # (batch_size*num_heads，查询或者“键－值”对的个数，
    # num_hiddens/num_heads)
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.w_q(queries), self.num_headers)
        keys = transpose_qkv(self.w_k(keys), self.num_headers)
        values = transpose_qkv(self.w_v(values), self.num_headers)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_headers, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_out(output, self.num_headers)
        return self.w_o(output_concat)


num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
print(attention)
batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens).size())

class PositionalEncoding(nn.Module):
    """绝对位置编码"""
    # i -> max_len, j -> num_hiddens
    # p(i, 2j) = sin( i / (10000 ** (2j / d)))
    # p(i, 2j + 1) = cos(i / (10000 ** ((2j + 1) / d)))
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2) / num_hiddens )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, X.size()[1], :]
        X = self.dropout(X)
        return X
'''
# test code
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P1 = pos_encoding.P[:, :X.shape[1], :]
pos_encoding2 = d2l.PositionalEncoding(encoding_dim, 0)
P2 = pos_encoding2.P[:, :X.shape[1], :]
print(torch.allclose(P1, P2))
'''

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
'''
ffn = PositionWiseFFN(4, 8, 4)
ffn.eval()
print(ffn(torch.ones((2, 3, 4)))[0])
'''

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
'''
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
print(add_norm(torch.ones(2,3,4), torch.ones(2,3,4)).shape)
'''

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))

X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)