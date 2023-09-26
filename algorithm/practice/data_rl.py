import re


# expr = "1,5,7,7,8,9,13,15:0S8*15=120,9-5=4,7+7=14,14-120=-106,4+1=5,5/-106=0,0/13=0"
def transfer(text):
    pattarn = r"[+*=]"
    if "+" in text:
        op = "+"
    else:
        op = "*"
    parts = re.split(pattarn, text)
    parts[0], parts[1] = parts[1], parts[0]
    parts.insert(1, op)
    parts.insert(-1, "=")
    return "".join(parts)


def data_enhance(expr):
    try:
        part1, part2 = expr.split("S")
        parts = part2.split(",")
        for idx, calculation in enumerate(parts):
            if "+" in calculation or "*" in calculation:
                parts[idx] = transfer(calculation)
        return part1 + "S" + ",".join(parts)
    except Exception as e:
        return ""


def write_enhance_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        print(line)
        line = data_enhance(line)
        print(line)
        if len(lines) > 0:
            new_lines.append(line)
    with open(file_path, "a") as f:
        f.write("\n")
        for line in new_lines:
            f.write(line)

write_enhance_data("sft_train.txt")
