from torch.utils.data import IterableDataset, DataLoader

class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        assert end > start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


ds = MyIterableDataset(0, 10)
print(list(DataLoader(ds, num_workers=0)))
# print(list(DataLoader(ds, num_workers=2)))

