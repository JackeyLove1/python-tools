import time
from typing import List
from collections import defaultdict
import multiprocessing

items_nums = 3000
single_len = 3
file_path = "rl_train_dedup.txt"
core_nums = 6
pool_timeout = 5

def calculate_operations(nums):
    memo = defaultdict(list)
    results = []

    def helper(nums, path):
        if len(nums) == 1:
            results.append(path + str(nums[0]))
            return

        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                remaining_nums = nums[:i] + nums[i + 1:j] + nums[j + 1:]

                # Addition
                helper([a + b] + remaining_nums, f"{path},{a}+{b}={a + b},")

                # Subtraction
                helper([a - b] + remaining_nums, f"{path},{a}-{b}={a - b},")
                helper([b - a] + remaining_nums, f"{path},{b}-{a}={b - a},")

                # Multiplication
                helper([a * b] + remaining_nums, f"{path},{a}*{b}={a * b},")

                # Division
                if b != 0:
                    helper([a // b] + remaining_nums, f"{path},{a}/{b}={a // b},")
                if a != 0:
                    helper([b // a] + remaining_nums, f"{path},{b}/{a}={b // a},")

    helper(nums, "")
    for res in results:
        tmp_res = res.split(",")
        target = int(tmp_res.pop())
        memo[target].append(",".join(tmp_res))
    return memo


def print_memo(memo):
    for k, v in memo.items():
        print(f"k:{k}, v:{str(v)}")


def merge(vec1: List[str], target1: int, vec2: List[str], target2: int, op: str, end: int) -> List[str]:
    result = []
    for cal1 in vec1:
        for cal2 in vec2:
            result.append(cal1 + "," + cal2 + "," + f"{target1}{op}{target2}={end}")
    # print("result:", result)
    return result


def handle_result(result, line):
    for idx in range(len(result)):
        if result[idx].startswith(","):
            result[idx] = result[idx].lstrip(",")
        result[idx] = result[idx].replace(",,", ",")
        result[idx] =  line + "S" + result[idx]
    return result[:single_len]


def binary_enumeration(arr: List[int], target: int, line: str) -> List[str]:
    n = len(arr)
    if n >= 7:
        return []
    result = []
    found = False
    for i in range(1, 2 ** n):
        if found:
            return handle_result(result, line)
        subset1, subset2 = [], []
        for j in range(n):
            if (i >> j) & 1:
                subset1.append(arr[j])
            else:
                subset2.append(arr[j])
        memo1 = calculate_operations(subset1)
        memo2 = calculate_operations(subset2)
        for k1, vec1 in memo1.items():
            if found:
                break
            for k2, vec2 in memo2.items():
                if k1 + k2 == target:
                    res1 = merge(vec1, k1, vec2, k2, "+", target)
                    res2 = merge(vec2, k2, vec1, k1, "+", target)
                    result.extend(res1)
                    result.extend(res2)
                    found = True
                    break
                if k1 - k2 == target:
                    res = merge(vec1, k1, vec2, k2, "-", target)
                    result.extend(res)
                    found = True
                    break
                if k2 - k1 == target:
                    res = merge(vec2, k2, vec1, k1, "-", target)
                    result.extend(res)
                    found = True
                    break
                if k1 * k2 == target:
                    res1 = merge(vec1, k1, vec2, k2, "*", target)
                    res2 = merge(vec2, k2, vec1, k1, "*", target)
                    result.extend(res1)
                    result.extend(res2)
                    found = True
                    break
                if k2 != 0 and k1 // k2 == target:
                    res = merge(vec1, k1, vec2, k2, "/", target)
                    result.extend(res)
                    found = True
                    break
                if k1 != 0 and k2 // k1 == target:
                    res = merge(vec2, k2, vec1, k1, "/", target)
                    result.extend(res)
                    found = True
                    break
    return handle_result(result, line)


def handle_line(line: str):
    numbers, target = line.split(":")
    target = int(target)
    nums = list(map(lambda num: int(num), numbers.split(",")))
    return nums, target


def write_files(file_path, target_path):
    start = time.time()
    with multiprocessing.Pool(processes=core_nums) as pool:
        result = []
        pool_results = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[:(items_nums // 3) + 1]:
                line = line.rstrip("\n")
                line = line.rstrip("S")
                nums, target = handle_line(line)
                pool_results.append(pool.apply_async(binary_enumeration, args=(nums, target, line,)))
            for pool_result in pool_results:
                try:
                    lines = pool_result.get(timeout=pool_timeout)
                    for res in lines:
                        if len(res) > (136 + 32):
                            continue
                        result.append(res)
                except TimeoutError:
                    continue

    with open(target_path, "a") as f:
        for line in result:
            f.write(line + "\n")
        print("write lines:", len(result))
    print(f"Cost:{time.time() - start}")


if __name__ == "__main__":
    write_files("rl_train_dedup.txt", "sft_train.txt")
# 测试
# binary_enumeration(array)
'''
numbers = [6, 9, 9, 14, 15]
target = -17
result = binary_enumeration(numbers, target)
print(result)

numbers = [1, 2, 5, 8, 10, 11]
target = 0
result = binary_enumeration(numbers, target)
print(result)

numbers = [1, 3, 4, 5, 9, 12, 13]
target = 3
result = binary_enumeration(numbers, target)
print(result)
'''

from trl import PPOTrainer