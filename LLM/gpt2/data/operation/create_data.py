import numpy as np
import random
from dataclasses import dataclass
import os
from tokenizer_operation import OperationTokenizer

random.seed(0)
np.random.seed(0)
tokenizer = OperationTokenizer()
bos = tokenizer.decode(tokenizer.bos_token_id)
eos = tokenizer.decode(tokenizer.eos_token_id)


@dataclass
class GenerateDataConfig:
    min_length = 1
    max_length = 10
    words = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    freq = np.array([7, 5, 5, 7, 6, 5, 7, 6, 5, 7])
    prob = freq / freq.sum()
    ops = ["+", "-"]
    nums = 10000  # 2e5


gconfig = GenerateDataConfig()


def get_data():
    l1 = random.randint(gconfig.min_length, gconfig.max_length)
    s1 = np.random.choice(gconfig.words, size=l1, replace=True, p=gconfig.prob)
    s1 = s1.tolist()

    l2 = random.randint(gconfig.min_length, gconfig.max_length)
    s2 = np.random.choice(gconfig.words, size=l2, replace=True, p=gconfig.prob)
    s2 = s2.tolist()

    op = np.random.choice(gconfig.ops)
    x = s1 + [op] + s2 + ["="]
    num1, num2 = int("".join(s1)), int("".join(s2))

    if op == "+":
        y = str(num1 + num2)
    elif op == "-":
        y = str(num1 - num2)
    elif op == '*':
        y = str(num1 * num2)
    elif op == '/':
        if num2 == 0:
            y = "n"
        else:
            y = str(num1 / num2)
    else:
        raise RuntimeError("Unsupported operation")
    return x, y


def create_one():
    x, y = get_data()
    return bos + "".join(x) + "".join(y) + eos


def write2txt(nums: int):
    process_path = os.getcwd()
    if "data" in os.getcwd():
        path = os.path.join(process_path, "small_sft_train.txt")
    else:
        path = os.path.join(process_path, "data/calcn/small_sft_train.txt")
    with open(path, "w") as f:
        for _ in range(nums):
            line = create_one()
            f.write(line + "\n")
    print(f"Write {nums} data in {path}")


if __name__ == "__main__":
    # TODO: set data_nums
    write2txt(gconfig.nums)
