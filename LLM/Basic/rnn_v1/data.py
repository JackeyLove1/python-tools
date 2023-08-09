import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
# print(corpus)
# print(vocab.token_freqs)
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_substeps = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_substeps * num_steps, num_steps))
    random.shuffle(initial_indices)
    def data(pos):
        return corpus[pos:pos + num_steps]
    num_batches = num_steps // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
myseq = list(range(35))
for X, Y in seq_data_iter_random(myseq, batch_size=2, num_steps=5):
    print(f"X:{X}, Y:{Y}")


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
print("vocab len:", len(vocab))
import torch.nn as nn
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
print(f"Y.shape:{Y.shape}, state_new.shape:{state_new.shape}")
