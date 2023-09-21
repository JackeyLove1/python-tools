"""
Prepare the calc numbers datasets.
Use a specialized tokenization that includes only necessary numbers and symbols.
sft_train.txt: prompt + response, typically used for SFT (Supervised Fine-Tuning)
A dataset, sft_train.bin, will be made.
"""
import os
import pickle
import requests
import numpy as np
from tokenizer_calcn import CalcNTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', default='small/small_sft_train.txt')
parser.add_argument('--output', default='small/small_sft_train.bin')
args = parser.parse_args()

# data
f_input = args.input
f_output = args.output
# common config
max_prompt_len = 32
max_resp_len = 136
sep_len = 1
debug = False
# the tokenizer
tokenizer = CalcNTokenizer()

# the calc-n sft dataset
input_file_path = os.path.join(os.path.dirname(__file__), f_input)
with open(input_file_path, 'r') as f:
    lines = f.readlines()
print(f"{len(lines)} lines loaded")

# preprocessing the data line by line: left padding prompt, right padding response, equal length sequence
train_ids = []
sep_str = tokenizer.decode(tokenizer.sep_token_id)
pad_str = tokenizer.decode(tokenizer.pad_token_id)
eos_str = tokenizer.decode(tokenizer.eos_token_id)
sequence_len = max_prompt_len + sep_len + max_resp_len  # should be 169
for line in lines:
    prompt, response = line.strip().split(sep_str)
    left_padding_len = max(max_prompt_len - len(prompt), 0)
    prompt = ''.join([pad_str] * left_padding_len) + prompt
    prompt = prompt[0: max_prompt_len]  # double-safe for the prompt length
    response = response + eos_str
    right_padding_len = max(max_resp_len - len(response), 0)
    response = response + ''.join([pad_str] * right_padding_len)
    response = response[0: max_resp_len]  # double-safe for the response length
    sequence = prompt + sep_str + response
    sequence_ids = tokenizer.encode(sequence)
    assert len(sequence_ids) == sequence_len, f'sequence_ids len {len(sequence_ids)} != sequence_len {sequence_len}'
    train_ids.extend(sequence_ids)

# append an extra sentinel data
train_ids.extend(train_ids[-1-sequence_len:-1])

# convert to np array and writ to file
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), f_output))
assert (len(lines) + 1)*sequence_len == train_ids.size
print(f'wrote ({len(lines)}+1)*{sequence_len}={train_ids.size} data to file.')

