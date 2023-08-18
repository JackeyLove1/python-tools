from torch.utils.data import Dataset, IterableDataset
import json
import random
import os
import numpy as np


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "rt") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

# train_data = AFQMC("../data/afqmc_public/train.json")
# valid_data = AFQMC("../data/afqmc_public/test.json")
#
# print(f"train size:{len(train_data)}, valid size:{len(valid_data)}") # train size:34334, valid size:3861

class IteratorableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, "rt") as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IteratorableAFQMC('../data/afqmc_public/train.json')
print(next(iter(train_data)))
'''
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
'''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW, get_scheduler

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples ):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append((sample['sentence2']))
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    y = torch.tensor(batch_label)
    return X, y

train_data_loader = DataLoader(train_data, batch_size=4, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_data_loader))
# print(f"batch_X size:{len(batch_X.tokens())}, batch_y size:{batch_y.shape}")
# print(batch_X)
# print(batch_y)
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)

import torch.nn as nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1
n_hiddens = 768
n_output = 2
seed_everything(42)
learning_rate = 1e-5
batch_size = 4
epoch_num = 3

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_hiddens, n_output)
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
print(model)

outputs = model(batch_X)
print(outputs.shape)

from tqdm.auto import tqdm
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_step_num = (epoch-1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

loss_fn = nn.CrossEntropyLoss()

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_data_loader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_data_loader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")