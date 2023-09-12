'''
{
    "instruction": "你是个优秀的小说作家，请续写下面的内容：",
    "input": "xxxx",
    "output": "xxxx"
}
len(input) + output(input) <= 500
'''


def handle_txt():
    file_path = "jinyong.txt"
    minl = 30
    outputs = []
    with open(file_path, "r") as f:
        for line in f:
            line.strip()
            if len(line) > minl:
                outputs.append(line)
    with open("new.txt", "w") as f:
        for line in outputs:
            f.write(line)


import json

reserve_length = 1000
chunk_size = 100


def format_story_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    datas = []
    for idx in range(0, len(content), chunk_size * 2):
        input_text = content[idx:idx + chunk_size]
        output_text = content[idx + chunk_size:min(len(content), idx + 4 * chunk_size)]
        data = {
            "instruction": "你是个优秀的小说作家，请续写下面的内容：",
            "input": input_text,
            "output": output_text
        }
        datas.append(data)
    print("length:", len(datas))
    if reserve_length > 0:
        datas = datas[:reserve_length]
    return datas


def save_to_json_file(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Test the functions with a sample story file
file_path = "new.txt"
output_path = "new.json"
formatted_data = format_story_to_json(file_path)
save_to_json_file(formatted_data, output_path)
print(f"JSON data saved to {output_path}")
