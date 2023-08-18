from torch.utils.data import Dataset
import json


test_file_path = "test.json"
with open(test_file_path, "rt") as f:
    for idx, line in enumerate(f):
        print(f"idx:{idx}, line:{line.strip()}")
        sample = json.loads(line.strip())
        print(f"Sample:{sample}, type:{type(sample)}")