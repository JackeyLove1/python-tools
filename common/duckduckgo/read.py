file_path1="/mnt/bytenas/python_filesystem/93MO24"
file_path2="/data00/fanhuanjie/bytenas/testing/chaos_cases_v1/src/python_filesystem/93MO24"
offset=4096
def read(file_path: str) -> bytes:
    with open(file_path1, "rb") as file:
        file.seek(offset)
        data = file.read()
        print(data)
        return data
if __name__ == "__main__":
    data1 = read(file_path1)
    data2 = read(file_path2)
    if data1 == data2:
        print("data is equal")
    else:
        print("data is not equal")