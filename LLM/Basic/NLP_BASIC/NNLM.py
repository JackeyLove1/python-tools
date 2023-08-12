import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from torch import Tensor

sentences = ["i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()
word_set = list(set(word_list))
word2idx = {w: i for i, w in enumerate(word_set)}
idx2word = {i: w for i, w in enumerate(word_set)}
vocab_size = len(word_set)
n_hidden = 20
n_step = 2


def match_batch(sentences: List[str]) -> Tuple[Tensor, Tensor]:
    input_batch = []
    target_batch = []
    for sentence in sentences:
        words = sentence.split()
        input = [word2idx[word] for word in words[:-1]]
        target = [word2idx[words[-1]]]

        input_batch.append(input)
        target_batch.append(target)
    return torch.LongTensor(input_batch), torch.LongTensor(target_batch)


class NNLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_hidden: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.H = nn.Linear(embedding_dim * n_step, n_hidden)
        self.tanh = nn.Tanh()
        self.U = nn.Linear(n_hidden, vocab_size, bias=False)
        self.W = nn.Linear(n_step * self.embedding_dim, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))

    def forward(self, x):
        x = self.C(x)  # (batch_size, n_step, embedding_dim)
        x = x.view(-1, n_step * self.embedding_dim)  # (batch_size, n_step * embedding_dim)
        hidden = self.tanh(self.H(x))  # (batch_size, n_hidden)
        output = self.b + self.W(x) + self.U(hidden)  # (batch_size, vocab_size)
        return output


if __name__ == "__main__":
    model = NNLM(vocab_size=vocab_size, embedding_dim=2, n_hidden=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    input_batch, target_batch = match_batch(sentences)

    for epoch in range(5000):
        output = model(input_batch)  # (batch_size, vocab_size)
        loss = criterion(output, target_batch.squeeze())
        optimizer.zero_grad()
        loss.backward()
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss.item()))
        optimizer.step()

    pred = model(input_batch).max(dim=-1)[1]
    print(pred)
    print([idx2word[idx.item()] for idx in pred])
