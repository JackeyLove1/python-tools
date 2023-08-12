import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

corpus = [
    ['I', 'love', 'natural', 'language', 'processing'],
    ['NLP', 'is', 'fascinating'],
    ['Machine', 'learning', 'is', 'fun'],
    ['I', 'enjoy', 'working', 'with', 'Python'],
    ['Deep', 'learning', 'is', 'powerful'],
    ['NLP', 'is', 'part', 'of', 'AI'],
]

word2idx = {}
for sentence in corpus:
    for word in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

vocab_size = len(word2idx)
idx2word = {idx: word for word, idx in word2idx.items()}

def generate_training_sample(sentence, num_neg_sample):
    positive_sample = [word2idx[word] for word in sentence]

    # 生成负样本
    # TODO: we should choose the sample is not similry to sentence
    negative_samples = []
    for _ in range(num_neg_sample):
        negative_sentences = random.choice(corpus)
        negative_sample = [word2idx[word] for word in negative_sentences]
        negative_samples.append(negative_sample)
    return positive_sample, negative_samples

class NLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(NLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        logits = self.linear(embedded)
        outputs = self.sigmoid(logits)
        return outputs

embedding_dim = 10
num_neg_samples = 5
batch_size = 2
learning_rate = 0.001
num_epochs = 10

model = NLPModel(vocab_size, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
positive_sample, negative_samples = generate_training_sample(corpus[0], num_neg_samples)
dataset = [(positive_sample, negative_samples) for _ in range(batch_size)]
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for positive_samples, negative_samples in dataloader:
        # positive sample
        print(positive_samples, negative_samples)
        positive_samples = positive_samples
        positive_outputs = model(positive_samples)
        positive_loss = -torch.log(positive_outputs).mean()

        # negative sample
        negative_samples = negative_samples
        negative_outputs = model(negative_samples)
        negative_loss = -torch.log(1 - negative_outputs).mean()

        loss = positive_loss + negative_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")