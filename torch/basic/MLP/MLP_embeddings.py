import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import optim
class BowDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets
class MLP(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        hidden = self.linear1(embeddings)
        activation = self.activation(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs, dim=1)
        return probs

class CNN(nn.Module):
    def __init__(self,vocab_size,embedding_dim, filter_size,num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.activate(embedding)
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embedidng = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)
    def forward(self, inputs, outputs):
        embeddings = self.embedidng(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs

# define const
batch_size = 10
num_epoch = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load datasets
def LoadDatasets():
    return torch.tensor([]), torch.tensor([])

train_dataset, test_dataset = LoadDatasets()
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

# model train
model = Model()
#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #使用Adam优化器

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

# model test
#测试过程
acc = 0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1) == targets).sum().item()

#输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")

net = MLP(vocab_size=8, embedding_dim=3, hidden_dim=5, num_class=2)
inputs = torch.tensor([[0,1,2,1],[4,6,6,7]],dtype=torch.long)
outputs = net(inputs)
print(outputs)