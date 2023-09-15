import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed
from typing import List

set_seed(0)
words = '<PAD>,<BOS>,<EOS>,1,2,3,4,5,6,7,8,9,0,+,='
vocab = {word: i for i, word in enumerate(words.split(','))}
vocab_r = {idx: word for word, idx in vocab.items()}
print(vocab)
print(vocab_r)


def get_data(min_length=10, max_length=20):
    # 定义词集合
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 每个词被选中的概率
    p = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    p = p / p.sum()

    # 随机采样s1 s2
    n1 = random.randint(min_length, max_length)
    s1 = np.random.choice(words, size=n1, replace=True, p=p)
    s1 = s1.tolist()

    n2 = random.randint(min_length, max_length)
    s2 = np.random.choice(words, size=n2, replace=True, p=p)
    s2 = s2.tolist()

    # x
    x = s1 + ['+'] + s2 + ['=']

    # y
    y = int(''.join(s1)) + int(''.join(s2))
    y = list(str(y))

    # add BOS and EOS
    x = ['<BOS>'] + x
    y = y + ['<EOS>']

    return x, y


# define dataset
class TwoSumDataset(torch.utils.data.Dataset):
    def __init__(self, size=100000, min_length=10, max_length=20):
        super().__init__()
        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        x, y = self.get(item)

        # tokenize
        context_ids = [vocab[i] for i in x]
        target_ids = [vocab[i] for i in y]
        input_ids = context_ids + target_ids

        labels = [-100] * len(context_ids) + target_ids
        masks = [0 if t == vocab['<PAD>'] else 1 for t in input_ids]
        example = {"input_ids": input_ids, "labels": labels, "attention_mask": masks}
        return example

    def get(self, item):
        return get_data(self.min_length, self.max_length)

    def show_examples(self, example):
        input_ids, labels = example['input_ids'], example['labels']
        x = ''.join([vocab_r[a] for a, b in zip(input_ids, labels) if b == -100])
        y = ''.join([vocab_r[a] for a, b in zip(input_ids, labels) if b != -100])
        print(x + y)


ds_train = TwoSumDataset(size=100000, min_length=10, max_length=20)
ds_val = TwoSumDataset(size=10000, min_length=10, max_length=20)
example = ds_train[0]
ds_train.show_examples(example)


def data_collator(examples: List):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids)  # 用最长的长度进行padding

    input_ids = []
    labels_list = []
    masks_list = []

    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]
        masks = example["attention_mask"]

        ids = [vocab['<PAD>']] * (longest - length) + ids
        labs = [-100] * (longest - length) + labs
        mask = [0] * (longest - length) + masks

        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
        masks_list.append(torch.LongTensor(mask))

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(masks_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

dl_train = DataLoader(
    dataset=ds_train,
    batch_size=200,
    drop_last=True,
    shuffle=True,
    collate_fn=data_collator,
)

dl_val = DataLoader(
    dataset=ds_val,
    batch_size=200,
    drop_last=True,
    shuffle=False,
    collate_fn=data_collator,
)

for batch in dl_train:
    print(batch)
    break