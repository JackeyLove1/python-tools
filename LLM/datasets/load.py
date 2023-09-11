from datasets import load_dataset

# 1. load dataset from url
# !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
# !unzip drugsCom_raw.zip
# data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
# drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
# drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
# drug_sample[:3]

# 2. load train
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")
print(dataset)
print(len(dataset['text']))
print(dataset['text'][0])

def split_content(item):
    return {"text" : item["text"].split("###")[-1]}
dataset = dataset.map(split_content)
print(dataset["text"][0])

# 3. load local CSV
'''
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
'''
# stream_dataset = load_dataset(dataset_name, split="train", streaming=True)
# print(stream_dataset)
# print(next(iter(stream_dataset)))

# self-define dataset
import requests
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
import pandas as pd
assert response.status_code == 200, "Failed to fetch from github"
df = pd.DataFrame.from_records(response.json())
df.to_json(f"issues.jsonl", orient="records", lines=True)
issues_dataset = load_dataset("json", data_files="issues.jsonl", split="train")
print(issues_dataset)
