from datasets import load_dataset

# 1. load dataset from url
# data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
# pubmed_dataset = load_dataset("json", data_files=data_files, split="train")

# 2. load frin
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name, split="train")
print(dataset)
print(dataset['train'])
# 3. load local CSV
'''
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
'''
