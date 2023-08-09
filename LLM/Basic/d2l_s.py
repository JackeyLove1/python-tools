import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)))

class Car:
    def __init__(self, color, brand):
        self.color = color
        self.brand = brand
car = Car("blue", "Toyota")
car_color = getattr(car, "color")
print(car_color)
car_brand = getattr(car, "brand")
print(car_brand)

from d2l import torch as d2l
import collections
import re
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open("../data/timemachine.txt", 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token="word"):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise RuntimeError('错误：未知词元类型：' + token)
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
class Vocab:
    @staticmethod
    def count_corpus(tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    def __init__(self, tokens=None, min_freq=0, reversed_tokens=None):
        if tokens is None:
            tokens = []
        if reversed_tokens is None:
            reversed_tokens = []
        counter = Vocab.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x : x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reversed_tokens # ['<unk>']
        self.token_to_idx = {token : idx for idx, token in enumerate(self.idx_to_token)} # [(0, '<unk>')]
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

vocab = Vocab(tokens)
# for i in range(10):
#     print("token:", tokens[i])
#     print("idx:", vocab[tokens[i]])
print(vocab.token_freqs[:10])

# two grams
# corpus = [token for token in tokens]
# vocab = Vocab(corpus)
# bigram_tokens = [pair for pair in zip(corpus[:], corpus[1:])]
# bigram_vocab = Vocab(bigram_tokens)
# print(bigram_vocab.token_freqs[:10])

def subsample(sentences, vocab):
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))
    return [[token for token in line if keep(token)] for line in sentences]

def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, 1 - window_size), min(len(line), i + 1  + window_size)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

