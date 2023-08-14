import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from typing import List
from torch import Tensor

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)

word_list = sentence.split()
vocab = list(set(word_list))
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
n_class = len(vocab)

# TextRNN Parameter
embedding_size = 5
batch_size = 5
n_step = len(word_list)
n_hidden = 10
n_direction = 2

def make_data(sentences: str):
    inputs = []
    targets = []
    word_list = sentences.split()
    for idx in range(len(word_list)-1):
        input = [word2idx[word] for word in word_list[:idx+1]]
        input = input + [0] * (n_step - len(input))
        target = word2idx[word_list[idx+1]]
        inputs.append(input)
        targets.append(target)
    return torch.LongTensor(inputs), torch.LongTensor(targets)


input_batch, target_batch = make_data(sentence)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=n_hidden, bidirectional=True)
        self.linear = nn.Linear(n_hidden * n_direction, n_class)

    def forward(self, x: Tensor):
        '''
        seq_len is n_step
        :param x: (batch_size, seq_len)
        :return: (batch_size, n_classes)
        '''
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.transpose(0, 1) # (seq_len, batch_size, embedding_dim)
        output, (hn, cn) = self.lstm(x)  # out: (seq_len, batch_size, hidden_dim), hn: (num_layers, batch_size, hidden_dim)
        output = output[-1]  # take the last step (batch_size, hidden_dim)
        output = self.linear(output)  # (batch_size, n_class)
        return output


model = BiLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(500):
    for idx, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        hidden = torch.zeros(1, batch_x.size()[0], n_hidden)
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y)
        optimizer.zero_grad()
        loss.backward()
        if (epoch + 1) % 100 == 0 and idx == 0:
            print(f"Epoch:{epoch + 1}, loss:{loss.item()}")
        optimizer.step()
