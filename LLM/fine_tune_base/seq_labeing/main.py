from torch.utils.data import DataLoader, Dataset

categories = set()

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    @staticmethod
    def load_data(data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]])  # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence,
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_file = "../data/china-people-daily-ner-corpus/example.train"
print(PeopleDaily.load_data(data_file))