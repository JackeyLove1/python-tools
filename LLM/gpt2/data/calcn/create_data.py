import os.path
import random
from typing import List
import argparse

random.seed(0)
data_nums = 100000
min_nums, max_nums = 5, 9
min_number, max_number = 1, 15
sep_token = "S"


def perform_operation(num1: int, num2: int, operator: str):
    result = 0
    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        result = int(num1 / num2)
    else:
        raise RuntimeError(f"Unsupported op:{operator}")
    cal = str(num1) + operator + str(num2) + "=" + str(result)
    return result, cal


def reduce_list(numbers: List[int]):
    operators = ["+", "-", "*", "/"]
    cal_list = []
    while len(numbers) > 1:
        index1 = random.randint(0, len(numbers) - 1)
        index2 = random.randint(0, len(numbers) - 1)
        while index2 == index1:
            index2 = random.randint(0, len(numbers) - 1)
        nums1 = numbers[index1]
        nums2 = numbers[index2]
        op = random.choice(operators)
        if nums2 == 0 and op == "/":
            continue
        result, cal = perform_operation(nums1, nums2, op)
        numbers.remove(nums1)
        numbers.remove(nums2)
        numbers.append(result)
        cal_list.append(cal)
    return numbers[0], cal_list


def create_one():
    nums = random.randint(min_nums, max_nums)
    numbers = [random.randint(min_number, max_number) for _ in range(nums)]
    numbers = sorted(numbers)
    inputs = ",".join([str(num) for num in numbers])
    result, cal_list = reduce_list(numbers)
    output = inputs + ":" + str(result) + sep_token + ",".join(cal_list)
    return output


def write2txt(nums: int):
    path = os.path.join(os.getcwd(), "data/calcn/small/small_sft_train.txt")
    with open(path, "w") as f:
        for _ in range(nums):
            line = create_one()
            f.write(line + "\n")
    print(f"Write {nums} data in {path}")


if __name__ == "__main__":
    # TODO: set data_nums
    write2txt(data_nums)
