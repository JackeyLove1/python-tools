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

# 3 words sentences (=sequence_length is 3)
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

# TextCNN Parameter
embedding_size = 2
sequence_length = len(sentences[0]) # every sentences contains sequence_length(=3) words
num_classes = len(set(labels))  # num_classes=2
batch_size = 3
output_channels = 3
filter_width = 2
filter_length = embedding_size

word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

def make_data(sentences : List[str], labels: List[int]):
    inputs = []
    for sentence in sentences:
        inputs.append([word2idx[w] for w in sentence.split()])
    targets = []
    for label in labels:
        targets.append(label)
    return torch.LongTensor(inputs), torch.LongTensor(targets)

input_batch, target_batch = make_data(sentences, labels)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=output_channels, kernel_size=(filter_width, filter_length)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.linear = nn.Linear(output_channels, num_classes)

    def forward(self, x : Tensor):
        '''
        :param x: (batch_size, seq_len)
        :return:  (batch_size, num_classes)
        '''
        batch_size = x.shape[0]
        x = self.embedding(x) # (batch_size, seq_len, embedding_size)
        x = x.unsqueeze(1) # (batch_size, input_channel, seq_len, embedding_size)
        x = self.conv(x) # (batch_size, output_channels * 1 * 1)
        x = x.view(batch_size, -1)
        x = self.linear(x) # (batch_size, num_classes)
        return x

model = TextCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(500):
    for idx, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_hat = model(batch_x)
        loss = criterion(y_hat, batch_y)
        optimizer.zero_grad()
        loss.backward()
        if (epoch + 1) % 100 == 0 and idx == 0:
            print(f"Epoch:{epoch + 1}, loss:{loss.item()}")
        optimizer.step()

test_text = 'i hate me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)

# Predict
model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")