from itertools import permutations, product


def evaluate_expression(exp):
    try:
        return eval(exp)
    except ZeroDivisionError:
        return None


def op(nums1, nums2, op):
    if op == "+":
        return nums1 + nums2
    elif op == "-":
        return nums1 - nums2
    elif op == "*":
        return nums1 * nums2
    elif op == "/":
        return nums1 // nums2
    else:
        raise RuntimeError(f"Unsupported operation {op}")


def find_target_sequence(numbers, target):
    operations = ['+', '-', '*', '//']
    for perm in permutations(numbers):
        for ops in product(operations, repeat=len(numbers) - 1):
            expr = str(perm[0])
            for i in range(1, len(perm)):
                expr += ops[i - 1] + str(perm[i])
            try:
                if eval(expr) == target:
                    expr = expr.replace("//", "/", -1)
                    return expr
            except ZeroDivisionError:
                continue
    return ""


numbers = [1, 5, 7, 7, 8, 9, 13, 15]
target = 0
result = find_target_sequence(numbers, target)
print("Expression:", result)

import re


def step_by_step_calculation(expression):
    tokens = re.split('([+\-*/])', expression.replace(' ', ''))
    used = [False] * len(tokens)
    st = []
    steps = []
    while len(st) != 0:
        for idx in range(len(tokens)):
            token = tokens[idx]
            if token not in "*/":
                st.append(token)
            else:
                if token is "*":
                    num1 = int(st.pop())
                    used[idx-1] = True
                    


    return steps


# Given expression
expression = "1+5+7+7-8*9/13-15"
# Evaluate the expression to get the final result for reference
final_result = eval(expression)
# Perform the step-by-step calculation
step_by_step_result = step_by_step_calculation(expression)
print(step_by_step_result, final_result)
