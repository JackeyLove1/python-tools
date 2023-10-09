import os
import random
import hashlib
import multiprocessing
import time
from typing import List, Union
from collections import defaultdict


def calculate_md5_range(mode_strings: List[str]):
    md5_hashes = []
    for current_string in mode_strings:
        md5_hash = hashlib.md5(current_string.encode("utf-8")).hexdigest()
        md5_hashes.append(md5_hash)
    return md5_hashes


class BigFileTestArgs:
    def __init__(self):
        random.seed(int(time.time()))
        self._file_size = 8 * 1024 * 1024 * 1024 * 1024  # default 8T
        self.num_process = 1
        self.mount_path = "/mnt"
        self.file_path = os.path.join(self.mount_path, "bigfile")
        self.file_name = None
        self.chunk_size = 4096  # ensure file_size / chunk_size == 0
        self.write_nums = 0
        self.chunk_mode = [mode_char * self.chunk_size for mode_char in
                           list(map(str, range(1, 10)))]  # ['1' * 4096 ... ]
        self.md5_mode = calculate_md5_range(self.chunk_mode)
        self.max_test_write_length = 1 * 1024 * 1024 * 1024  # 1G
        self.max_test_truncate_size = 8 * 1024 * 1024 * 1024 * 1024  # 8T
        self.debug = False
        self.write_list = ['0'] * self.write_nums  # "0" should be unwritten

    @property
    def file_size(self):
        return self._file_size

    @file_size.setter
    def file_size(self, value: int):
        assert value > 0, "value should be larger than 0"
        self._file_size = value

    @staticmethod
    def generate_name():
        random.seed(int(time.time()))
        return "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(16)])

    def prepare_work(self):
        if not self.debug and not os.path.ismount(self.mount_path):
            raise RuntimeError("{} is not a mounted dir".format(self.mount_path))
        self.file_path = os.path.join(self.mount_path, "bigfile")
        if not os.path.exists(self.file_path):
            os.system("mkdir -p {}".format(self.file_path))
        if self.file_name is None:
            self.file_name = os.path.join(self.file_path, BigFileTestArgs.generate_name())
        self.write_nums = self.file_size // self.chunk_size
        self.write_list = ['0'] * self.write_nums


def write_fixed_mode(offset: int, index: int, args: BigFileTestArgs):
    assert index < len(args.chunk_mode)
    if not os.path.exists(args.file_name):
        os.system("touch {}".format(args.file_name))
    with open(args.file_name, 'rb+') as file:
        file.seek(offset)
        file.write(args.chunk_mode[index].encode("utf-8"))
        args.write_list[offset // args.chunk_size] = args.chunk_mode[index][0]


def calculate_md5_stream(file_path: str):
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_md5_str(data: str):
    md5_hash = hashlib.md5()
    md5_hash.update(data.encode('utf-8'))
    return md5_hash.hexdigest()


def write_data_with_offset(args: BigFileTestArgs, data: Union[bytes, str], offset: int):
    """data should be the same like chunk_mode and size should be chunk_size"""
    assert len(data) == args.chunk_size, "write data length should be equal to chunk_size"
    tmp = data if isinstance(data, str) else data.decode("utf-8")
    args.write_list[offset // args.chunk_size] = tmp[0]
    with open(args.file_name, 'rb+') as file:
        file.seek(offset)
        if isinstance(data, str):
            data = data.encode('utf-8')
        file.write(data)


def read_data_with_offset(file_path: str, offset: int, length: int):
    with open(file_path, 'rb') as file:
        file.seek(offset)
        data = file.read(length)
    return data.decode("utf-8")


def write_chunk_cycle(args: BigFileTestArgs):
    if not os.path.exists(args.file_name):
        os.system("touch {}".format(args.file_name))
    for idx in range(args.write_nums):
        offset = idx * args.chunk_size
        data = args.chunk_mode[idx % len(args.chunk_mode)]
        write_data_with_offset(args, data, offset)


def calculate_md5_cycle(args: BigFileTestArgs):
    md5_hash = hashlib.md5()
    for idx in range(args.write_nums):
        chunk = args.chunk_mode[idx % len(args.chunk_mode)]
        md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_md5_without_read(args: BigFileTestArgs, offset1: int, offset2: int):
    ''' use write mode list'''
    md5_hash = hashlib.md5()
    offset2 = min(offset2, os.path.getsize(args.file_name))
    print(f"off1:{offset1}, offset2:{offset2}, filesize:{os.path.getsize(args.file_name)}")
    while offset1 < offset2:
        idx = offset1 // args.chunk_size
        if args.write_list[idx] == "0":
            continue
        chunk = args.write_list[idx] * args.chunk_size
        md5_hash.update(chunk.encode("utf-8"))
        offset1 += args.chunk_size
    return md5_hash.hexdigest()


def calculate_md5_mode_list(args: BigFileTestArgs, mode_list: List[str]):
    md5_hash = hashlib.md5()
    for mode in mode_list:
        chunk = args.chunk_size * mode
        md5_hash.update(chunk.encode("utf-8"))
    return md5_hash.hexdigest()


def calculate_md5_partial(file_path: str, offset1: int, offset2: int):
    assert offset1 < offset2
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        file.seek(offset1)  # Move file pointer to the start offset
        while file.tell() < offset2:
            chunk_size = min(4096, offset2 - file.tell())
            chunk = file.read(chunk_size)
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_md5_parallel(file_path: str, num_processes: int):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_processes
    processes = []

    for i in range(num_processes):
        offset1 = i * chunk_size
        offset2 = offset1 + chunk_size if i < num_processes - 1 else file_size
        process = multiprocessing.Process(
            target=calculate_md5_partial,
            args=(file_path, offset1, offset2)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    md5_hash = hashlib.md5()
    for process in processes:
        md5_hash.update(process.exitcode.to_bytes(4, 'big'))

    return md5_hash.hexdigest()


def write_bigfile(args: BigFileTestArgs):
    return write_chunk_cycle(args)


def write_data_offset(args: BigFileTestArgs, offset: int):
    index = random.randint(0, len(args.chunk_mode) - 1)
    with open(args.file_name, "rb+") as file:
        file.seek(offset)
        file.write(args.chunk_mode[index].encode("utf-8"))
        args.write_list[offset // args.chunk_size] = args.chunk_mode[index][0]


def time_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        cost = time.time() - start
        print("Cost:{}s".format(str(int(cost))))

    return wrapper


@time_cost
def write_data_length(args: BigFileTestArgs, offset: int, length: int):
    if not os.path.exists(args.file_name):
        os.system("touch {}".format(args.file_name))
    assert length % args.chunk_size == 0, "random write data length should be multiple of chunk_size"
    if offset + length > os.path.getsize(args.file_name):
        args.write_list.extend(["0"] * (((offset + length) - os.path.getsize(args.file_name)) // args.chunk_size))
    write_nums = length // args.chunk_size
    for _ in range(write_nums):
        write_data_offset(args, offset)
        offset += args.chunk_size


@time_cost
def check_total_file(args: BigFileTestArgs):
    actual_md5 = calculate_md5_stream(args.file_name)
    expected_md5 = calculate_md5_mode_list(args, args.write_list)
    assert actual_md5 == expected_md5, "Failed to check total file md5, expected:{}, actual:{}".format(expected_md5,
                                                                                                       args)


def random_write(args: BigFileTestArgs):
    '''
    1. check overlap range md5
    2. check total file md5
    3. use write mode list to calculate expect md5 and use iterator calculate actual md5
    '''
    offset = args.chunk_size * random.randint(1, args.write_nums - 1)
    assert offset < os.path.getsize(args.file_name), "offset should be less than file size"
    length = args.chunk_size * random.randint(1, args.max_test_write_length // args.chunk_size)
    write_data_length(args, offset, length)

    # check write range md5
    begin_index = offset // args.chunk_size
    end_index = begin_index + length // args.chunk_size
    write_mode_list = args.write_list[begin_index:end_index]
    read_data = read_data_with_offset(args.file_name, offset, length)
    actual_md5 = calculate_md5_str(read_data)
    expected_md5 = calculate_md5_mode_list(args, write_mode_list)
    assert actual_md5 == expected_md5, "Failed to check range md5, expected:{}, actual:{}".format(expected_md5,
                                                                                                  actual_md5)
    # check total md5
    check_total_file(args)


def truncate_file(args: BigFileTestArgs, size: int):
    with open(args.file_name, 'r+') as file:
        file.truncate(size)
    num_bits = size // args.chunk_size
    if num_bits > len(args.write_list):
        args.write_list.extend(["0"] * (num_bits - len(args.write_list)))
    elif num_bits < len(args.write_list):
        args.write_list = args.write_list[:num_bits]
    else:
        return


def random_truncate(args: BigFileTestArgs):
    truncate_size = random.randint(1, args.max_test_truncate_size // args.chunk_size) * args.chunk_size
    truncate_file(args, truncate_size)
    check_total_file()
