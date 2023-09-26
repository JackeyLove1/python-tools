from copy import deepcopy


def dp_find_target(numbers, target, memo):
    # Check memoization table
    if frozenset(numbers) in memo:
        return memo[frozenset(numbers)]

    # Base Case: Check if target is in numbers
    for num, expr in numbers:
        if num == target:
            return expr

    # Recursive Case: Try all pairs of numbers
    for (a, expr_a) in numbers:
        for (b, expr_b) in numbers:
            if a == b and expr_a == expr_b:
                continue

            # Try all operations
            new_numbers = deepcopy(numbers)
            new_numbers.remove((a, expr_a))
            new_numbers.remove((b, expr_b))

            new_exprs = [
                (a + b, f"({expr_a}+{expr_b})"),
                (a - b, f"({expr_a}-{expr_b})"),
                (a * b, f"({expr_a}*{expr_b})")
            ]

            # Avoid division by zero
            if b != 0:
                new_exprs.append((a / b, f"({expr_a}/{expr_b})"))

            for new_num, new_expr in new_exprs:
                new_numbers.add((new_num, new_expr))
                result = dp_find_target(new_numbers, target, memo)
                if result:
                    return result
                new_numbers.remove((new_num, new_expr))

            new_numbers.add((a, expr_a))
            new_numbers.add((b, expr_b))

    # Update memoization table
    memo[frozenset(numbers)] = None
    return None


# Initialize memoization table
memo = {}

# Initialize set of numbers with their expression strings
# numbers = {(27, '27'), (9, '9'), (41, '41'), (9, '9'), (58, '58'), (14, '14')}
numbers = {(55, "55"),(27, "27"),(2, "2"),(37, "37"),(18, "18")}
# Find a sequence of operations to reach the target value (0)
result = dp_find_target(numbers, -53, memo)


def expression_to_steps(expr):
    steps = []
    operands = []
    operators = []

    num = ''
    for char in expr:
        if char.isdigit() or char == '.':
            num += char
        else:
            if num:
                operands.append(float(num))
                num = ''
            if char in "+-*/":
                operators.append(char)

    # Add the last number if it exists
    if num:
        operands.append(float(num))

    while operators:
        op1 = operands.pop(0)
        op2 = operands.pop(0)
        operator = operators.pop(0)

        operation_str = f"{op1}{operator}{op2}"
        result = eval(operation_str)

        steps.append(f"{operation_str}={result}")

        operands.insert(0, result)

    return ", ".join(steps)


# Convert the expression to a series of operations
print(result)
print(expression_to_steps(result))