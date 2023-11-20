# TODO(fhj): use merkel tree to accelerate writing and verifying
import os
import random
import hashlib
import multiprocessing
from multiprocessing import Process
import time
import argparse
from typing import List, Union
from collections import defaultdict
import logging
import datetime
import sys
from logging.handlers import RotatingFileHandler


def calculate_md5_range(mode_strings: List[str]):
    md5_hashes = []
    for current_string in mode_strings:
        md5_hash = hashlib.md5(current_string.encode("utf-8")).hexdigest()
        md5_hashes.append(md5_hash)
    return md5_hashes


class BigFileTestArgs:
    FILE_SIZE_8T = 8 * 1024  # default 8T
    FILE_SIZE_16T = 2 * FILE_SIZE_8T
    FILE_SIZE_64T = 8 * FILE_SIZE_8T
    MOUNT_PATH = os.getcwd()
    LOG_NAME = "LargeFileTest.log"
    WORK_NAME = "LargeFile"

    def __init__(self):
        random.seed(int(time.time()))
        self._file_size = 100 * 1024
        self._mount_path = os.getcwd()
        self._log_dir = os.getcwd()
        self.logger = None
        self.num_process = 1
        self.file_path = None
        self.file_name = None
        self.chunk_size = 1024  # ensure file_size / chunk_size == 0
        self.write_nums = 0
        self.chunk_mode = [mode_char * self.chunk_size for mode_char in
                           list(map(str, range(1, 10)))]  # ['1' * self.chunk_size ... ]
        self.md5_mode = calculate_md5_range(self.chunk_mode)
        self.max_test_write_length = 100 * 1024  # 100G
        self.max_test_truncate_size = 200 * 1024  # 8T
        self.write_list = ["\0"] * self.write_nums  # "\0" should be unwritten
        self.local_mode = True  # for testing

    def __str__(self):
        return "name:{},size:{}, mount:{}, log:{}".format(self.file_name, self._file_size, self._mount_path,
                                                          self._log_dir)

    @property
    def file_size(self):
        return self._file_size

    @file_size.setter
    def file_size(self, value: int):
        assert value > 0, "value should be larger than 0"
        self._file_size = value

    @property
    def mount_path(self):
        return self._mount_path

    @mount_path.setter
    def mount_path(self, mount_path: str):
        self._mount_path = mount_path

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        self._log_dir = log_dir

    @staticmethod
    def generate_name():
        random.seed(int(time.time()))
        return "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(16)])

    def setUp(self):
        '''we should set mount_dir log_dir file_size before use it'''
        if self.local_mode:
            self._mount_path = os.getcwd()
            self._log_dir = os.getcwd()
        self.file_path = os.path.join(self._mount_path, "bigfile")
        if not os.path.exists(self.file_path):
            os.system("mkdir -p {}".format(self.file_path))
        if self.file_name is None:
            self.file_name = os.path.join(self.file_path, BigFileTestArgs.generate_name())
        self.write_nums = self.file_size // self.chunk_size
        self.write_list = ["\0"] * self.write_nums
        self._log_dir = os.path.join(self._log_dir, "log")
        if not os.path.exists(self._log_dir):
            os.system("mkdir -p {}".format(self._log_dir))
        self.chunk_mode = [mode_char * self.chunk_size for mode_char in
                           list(map(str, range(1, 10)))]
        if self.logger is None:
            max_bytes = 100 * 1024 * 1024  # 100M
            backups = 100
            logger = logging.getLogger(BigFileTestArgs.WORK_NAME)
            logger.setLevel(logging.INFO)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            log_file = os.path.join(self._log_dir, '{}_{}.log'.format(BigFileTestArgs.WORK_NAME, timestamp))
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s (%(funcName)s:%(lineno)d)] %(levelname)s in %(module)s: %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            self.logger = logger
        self.logger.info("Succeed to init big file handle")
        self.logger.info(
            "file_name:{}, log dir:{}, local_mode:{}".format(self.file_name, self._log_dir, self.local_mode))

    def tearDown(self):
        self.logger.info("Tear Down BigFileTest, remove file:{}".format(self.file_name))
        if not self.local_mode:
            os.system("rm -rf {}".format(self.file_name))


def time_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        cost = time.time() - start
        print("Func:{}, Cost:{}s".format(func.__name__, str(int(cost))))

    return wrapper


def get_bigfile_handler(mount_dir: str, base_dir: str, file_size: int, local: bool = False):
    file = BigFileTestArgs()
    if not local:
        file.mount_path = mount_dir
        file.log_dir = base_dir
    else:
        file.mount_path = ""
        file.log_dir = ""
    file.file_size = file_size
    file.local_mode = local
    file.setUp()
    return file


def write_fixed_mode(offset: int, index: int, args: BigFileTestArgs):
    assert index < len(args.chunk_mode)
    if not os.path.exists(args.file_name):
        os.system("touch {}".format(args.file_name))
    with open(args.file_name, 'rb+') as file:
        file.seek(offset)
        file.write(args.chunk_mode[index].encode("utf-8"))
        args.write_list[offset // args.chunk_size] = args.chunk_mode[index][0]
        args.logger.info(
            "file_name:{}, offset:{}, write:{}, write_size:{}".format(args.file_name, offset, args.chunk_mode[index][0],
                                                                      args.chunk_size))


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
    args.logger.info("file_name:{}, file size:{}, chunk_size:{}, write_nums:{}".format(args.file_name, args.file_size,
                                                                                       args.chunk_size,
                                                                                       args.file_size // args.chunk_size))
    assert args.file_size % args.chunk_size == 0
    args.write_nums = args.file_size // args.chunk_size
    for idx in range(int(args.file_size / args.chunk_size)):
        offset = idx * args.chunk_size
        data = args.chunk_mode[idx % len(args.chunk_mode)]
        write_data_with_offset(args, data, offset)
        args.logger.info(
            "file_name:{}, offset:{}, write:{}, write_size:{}".format(args.file_name, offset, data[0],
                                                                      args.chunk_size))


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
    args.logger.info("off1:{}, offset2:{}, filesize:{}".format(offset1, offset2, os.path.getsize(args.file_name)))
    while offset1 < offset2:
        idx = offset1 // args.chunk_size
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


@time_cost
def write_bigfile(args: BigFileTestArgs):
    return write_chunk_cycle(args)


def write_data_offset(args: BigFileTestArgs, offset: int):
    index = random.randint(0, len(args.chunk_mode) - 1)
    with open(args.file_name, "rb+") as file:
        file.seek(offset)
        file.write(args.chunk_mode[index].encode("utf-8"))
        args.write_list[offset // args.chunk_size] = args.chunk_mode[index][0]
        args.logger.info(
            "file_name:{}, offset:{}, write:{}, write_size:{}".format(args.file_name, offset, args.chunk_mode[index][0],
                                                                      args.chunk_size))


@time_cost
def write_data_length(args: BigFileTestArgs, offset: int, length: int):
    if not os.path.exists(args.file_name):
        os.system("touch {}".format(args.file_name))
    assert length % args.chunk_size == 0, "random write data length should be multiple of chunk_size"
    if offset + length > os.path.getsize(args.file_name):
        args.write_list.extend(["\0"] * (((offset + length) - os.path.getsize(args.file_name)) // args.chunk_size))
    write_nums = length // args.chunk_size
    for _ in range(write_nums):
        write_data_offset(args, offset)
        offset += args.chunk_size


@time_cost
def check_total_file(args: BigFileTestArgs):
    actual_md5 = calculate_md5_stream(args.file_name)
    expected_md5 = calculate_md5_mode_list(args, args.write_list)
    if expected_md5 == actual_md5:
        args.logger.info(
            "Succeed to check md5, file_name:{}, expected md5:{}, actual md5:{}".format(args.file_name, expected_md5,
                                                                                        actual_md5))
    else:
        args.logger.error(
            "Failed to check md5, file_name:{}, expected md5:{}, actual md5:{}".format(args.file_name, expected_md5,
                                                                                       actual_md5))
    assert actual_md5 == expected_md5, "Failed to check total file md5, expected:{}, actual:{}".format(expected_md5,
                                                                                                       actual_md5)


@time_cost
def compare_total_file(args: BigFileTestArgs):
    compare_offset = 0
    chunk_size = args.chunk_size
    with open(args.file_name, "r") as file:
        file.seek(compare_offset)
        actual_chunk = file.read(chunk_size)
        compare_offset += chunk_size
        print("actual:", actual_chunk, " chunk_size:", chunk_size)
        idx = compare_offset // args.chunk_size
        assert idx < len(args.chunk_mode)
        expected_chunk = args.chunk_mode[idx] * args.chunk_size
        print("idx:", idx, " expect_chunk:", expected_chunk)
        if expected_chunk != actual_chunk:
            # args.logger.error("Failed to verify data, offset:", chunk_size,  "expect:", expected_chunk, " actual:", actual_chunk)
            assert False, "Failed to check"
        compare_offset += chunk_size


@time_cost
def random_write(args: BigFileTestArgs):
    '''
    1. check overlap range md5
    2. check total file md5
    3. use write mode list to calculate expect md5 and use iterator calculate actual md5
    '''
    file_size = os.path.getsize(args.file_name)
    offset = args.chunk_size * random.randint(1, file_size // args.chunk_size)
    assert offset <= os.path.getsize(args.file_name), "offset should be less than file size"
    length = args.chunk_size * random.randint(1, min(args.max_test_write_length // args.chunk_size,
                                                     args.file_size // args.chunk_size))
    args.logger.info(
        "file:{}, size:{}, random write offset:{}, length:{}".format(args.file_name, file_size, offset, length))
    write_data_length(args, offset, length)

    # check write range md5
    begin_index = offset // args.chunk_size
    end_index = begin_index + length // args.chunk_size
    write_mode_list = args.write_list[begin_index:end_index]
    read_data = read_data_with_offset(args.file_name, offset, length)
    actual_md5 = calculate_md5_str(read_data)
    expected_md5 = calculate_md5_mode_list(args, write_mode_list)
    msg = "file:{}, expected md5:{}, actual md5:{}".format(args.file_name, expected_md5, actual_md5)
    if actual_md5 == expected_md5:
        args.logger.info(msg)
    else:
        args.logger.error("Failed to check md5, {}".format(msg))
    assert actual_md5 == expected_md5, "Failed to check range md5, {}".format(msg)
    # check total md5
    check_total_file(args)


@time_cost
def truncate_file(args: BigFileTestArgs, size: int):
    origin_size = args.file_size
    with open(args.file_name, 'r+') as file:
        file.truncate(size)
        args.file_size = size
        args.logger.info("file_name:{}, origin_size:{}, truncate_size:{}".format(args.file_name, origin_size, size))
    num_bits = size // args.chunk_size
    if num_bits > len(args.write_list):
        args.write_list.extend(["\0"] * (num_bits - len(args.write_list)))
    elif num_bits < len(args.write_list):
        args.write_list = args.write_list[:num_bits]
    else:
        return


@time_cost
def random_truncate(args: BigFileTestArgs):
    assert os.path.exists(args.file_name), "{} is not exist".format(args.file_name)
    file_size = os.path.getsize(args.file_name)
    max_truncate_nums = args.max_test_truncate_size // args.chunk_size
    truncate_size = random.randint(1, min((file_size // args.chunk_size) * 2, max_truncate_nums)) * args.chunk_size
    args.logger.info(
        "file_name:{},origin_size:{}, truncate size:{}".format(args.file_name, args.file_size, truncate_size))
    truncate_file(args, truncate_size)
    check_total_file(args)


def BigFileTest(func):
    def wrapper(file: BigFileTestArgs, *args, **kwargs):
        start_time = time.time()
        file.setUp()
        file.logger.info(
            "Succeed to setup bigfile handler, file_name:{}, write_size:{}".format(file.file_name, file.file_size))
        func(file, *args, **kwargs)
        file.tearDown()
        print("Func: {}, Cost:{}s".format(func.__name__, int(time.time() - start_time)))

    return wrapper


@BigFileTest
def base_write(file: BigFileTestArgs):
    '''写入文件校验md5'''
    write_bigfile(file)
    check_total_file(file)


@BigFileTest
def base_random_write(file: BigFileTestArgs, operation_nums: int = int(sys.maxsize)):
    '''随机覆盖写然后校验md5'''
    assert os.path.exists(file.file_name)
    for _ in range(operation_nums):
        random_write(file)


@BigFileTest
def base_random_truncate(file: BigFileTestArgs, operation_nums: int = int(sys.maxsize)):
    '''随机截断文件然后校验md5'''
    assert os.path.exists(file.file_name)
    for _ in range(operation_nums):
        random_truncate(file)


KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB


def parse_arguments():
    parser = argparse.ArgumentParser()
    FILE_SIZE = 6 * TB
    MOUNT_PATH = os.getcwd()
    LOCAL = False
    NUMS = 1000
    parser.add_argument("--size", dest="size", type=int, required=False, default=FILE_SIZE,
                        help="写入文件大小")
    parser.add_argument("--mount_dir", dest="mount_dir", type=str, required=False, default=MOUNT_PATH,
                        help="挂载目录")
    parser.add_argument("--work_dir", dest="work_dir", type=str, required=False, default=os.getcwd(),
                        help="工作目录")
    parser.add_argument("--local", dest="local", type=bool, required=False, default=LOCAL,
                        help="本地测试模式")
    parser.add_argument("--nums", dest="nums", type=int, required=False, default=NUMS,
                        help="随机读写次数")
    args = parser.parse_args()
    return args


def random_execute(file: BigFileTestArgs, nums: int = 1):
    """random write and truncate"""
    # write and check md5
    write_bigfile(file)
    check_total_file(file)
    # random write and truncate many times
    for _ in range(nums):
        random_write(file)
        random_truncate(file)


def dump_execute(file: BigFileTestArgs, sleep_time: int = 60 * 60):
    """wait for dump and periodic verification"""
    write_bigfile(file)
    while True:
        check_total_file(file)
        time.sleep(sleep_time)


def local_test():
    test_size = 100 * MB
    file = get_bigfile_handler("", "", test_size, True)
    random_execute(file)


def mount_test():
    test_size = 1 * GB
    mount_path = "/mnt/bytenas"
    file = get_bigfile_handler(mount_path, "", test_size, False)
    random_execute(file)


def main():
    """write_verify - random_write - random_truncate"""
    args = parse_arguments()
    random_file = get_bigfile_handler(args.mount_dir, args.work_dir, args.size, args.local)
    random_execute(random_file, args.nums)
    '''
    dump_file = get_bigfile_handler(args.mount_dir, args.work_dir, args.size, args.local)
    p1 = Process(random_execute, args=(random_file, args.nums, ))
    p2 = Process(dump_execute, args=(dump_file, ))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    '''


if __name__ == "__main__":
    main()
