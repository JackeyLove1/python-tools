import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.utils.data as Data
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# define params
batch_size = 2  # mini-batch size
embedding_dim = 8  # embedding size
vocab_size = 100
context_size = 2
learning_rate = 0.001
sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
             "dog cat animal", "cat monkey animal", "monkey dog animal"]

word_sequences = " ".join(sentences).split()
word_list = list(set(word_sequences))
word2idx = {w: i for i, w in enumerate(word_list)}
idx2word = {i: w for w, i in enumerate(word_list)}
vocab_size = len(word_list)
# global param
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device:", device)

# define utils

# TODO: use torch one hot coding to enhance make_data
def make_data(skip_grams, vocab_size):
    input_data = []
    output_data = []
    for i in range(len(skip_grams)):
        input_word, output_word = skip_grams[i]
        input_idx, output_idx = word2idx[input_word], word2idx[output_word]
        input_data.append(np.eye(vocab_size)[input_idx])
        output_data.append(output_idx)
    dataset = Data.TensorDataset(torch.Tensor(input_data), torch.LongTensor(output_data))
    data_loader = Data.DataLoader(dataset, batch_size, True)
    return data_loader


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding = nn.Linear(self.vocab_size, self.embedding_dim, bias=False)
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)

    def forward(self, x):
        '''
        :param x: (batch_size, vocab_size)
        :return:  (batch_size, vocab_size)
        '''
        x = self.embedding(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    # get skip gram
    skip_gram = []
    for i in range(context_size, len(word_sequences) - context_size):
        target = word_sequences[i]
        context = word_sequences[i - context_size:i] + word_sequences[i + 1:i + context_size + 1]
        for w in context:
            skip_gram.append([target, w])

    model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = make_data(skip_gram, vocab_size)
    for epoch in range(5000):
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y_hat = model(batch_x)
            optimizer.zero_grad()
            loss = criterion(batch_y_hat, batch_y)
            loss.backward()
            if (epoch + 1) % 500 == 0 and idx == 0:
                print(f"Epoch:{epoch + 1}, loss:{loss.item()}")
            optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = float(W[i][0]), float(W[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
