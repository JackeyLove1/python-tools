
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pack_sequence
from tqdm.auto import tqdm
class MyData(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

def collate_fn1(data):
    data.sort(key=lambda x:len(x), reverse=False)
    data = pad_sequence(data, batch_first=True, padding_value=0)
    return data

def collate_fn2(data):
    data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data] # 获取数据真实的长度
    data = pad_sequence(data, batch_first=True)
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)

    data = pack_sequence(data)
    # seq_len = [s.size(0) for s in data]
    # data = pad_sequence(data, batch_first=True)
    # data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data

a = torch.tensor([1,2,3,4])
b = torch.tensor([5,6,7])
c = torch.tensor([7,8])
d = torch.tensor([9])
train_x = torch.cat([a, b, c, d])

data = MyData(train_x)
data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)

r"""Unpacks PackedSequence into a list of variable length Tensors

    ``packed_sequences`` should be a PackedSequence object.


    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1,2,3])
        >>> b = torch.tensor([4,5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        >>> print(sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
        >>> packed_sequences = pack_sequence(sequences)
        >>> print(packed_sequences)
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> print(unpacked_sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]


    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """