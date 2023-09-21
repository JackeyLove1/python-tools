import os
import numpy as np
from tokenizer_operation import OperationTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='small_sft_train.txt')
parser.add_argument('--output', default='small_sft_train.bin')
args = parser.parse_args()

# data
f_input = args.input
f_output = args.output
# common config
max_prompt_len = 20
max_resp_len = 120
sep_len = 1
sequence_len = max_prompt_len + sep_len + max_resp_len  # 141

tokenizer = OperationTokenizer()

equal_str = tokenizer.decode(tokenizer.equal_token_id)
bos_str = tokenizer.decode(tokenizer.bos_token_id)
eos_str = tokenizer.decode(tokenizer.eos_token_id)
pad_str = tokenizer.decode(tokenizer.pad_token_id)

with open(f_input, "r") as f:
    lines = f.readlines()
print(f"{len(lines)} has been loaded in memory")

train_ids = []
for line in lines:
    prompt, resp = line.strip().split(equal_str)
    left_padding_len = max(max_prompt_len - len(prompt), 0)
    prompt = pad_str * left_padding_len + prompt
    prompt = prompt[:max_prompt_len]
    right_padding_len = max(max_resp_len - len(resp), 0)
    resp = resp + pad_str * right_padding_len
    resp = resp[:max_resp_len]
    sequence = prompt + equal_str + resp
    inputs_id = tokenizer.encode(sequence)
    assert len(inputs_id) == sequence_len, f'sequence_ids len {len(inputs_id)} != sequence_len {sequence_len}'
    train_ids.append(inputs_id)

# convert to np array and writ to file
train_ids = np.array(train_ids, dtype=np.uint16)
train_ids.tofile(f_output)
print(f'wrote ({len(lines)}) lines data to file.')
