# python3 run_multifile_writer.py --mount_dir=/mnt/bytenas [--prefix=A --work_dir=${your_local_dir}]
import os
import random
import time
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque
from queue import Queue
import datetime
import hashlib
import argparse
from enum import Enum
import requests
import json
import multiprocessing


random.seed(int(time.time()))
msg_queue = Queue(-1)
WORK_NAME = "MultipleFile"
DEFAULT_WORK_DIR = os.getcwd()
DEFAULT_MOUNT_DIR = "/mnt/bytenas"
FILE_NUMS = 5000
OP_NUMS = 20
FILE_NAME_LENGTH = 20
MAX_TRUNCATE_SIZE = 1024 * 50  # 50K
MAX_WRITE_LENGTH = 1024  # 1k
SEPARATOR = "#"
# meta file: file_name + SEPARATOR + md5 + SEPARATOR + file_size + SEPARATOR + ctime + SEPARATOR + mtime
META_FILE = WORK_NAME + ".meta"
LOCAL_TEST = False


class Mode(Enum):
    OnlyWriteData = 1
    WriteAndCheckData = 2
    OnlyCheckData = 3


def get_logger(log_dir: str):
    logger = logging.getLogger(WORK_NAME)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, '{}_{}.log'.format(WORK_NAME, timestamp))
    file_handler = RotatingFileHandler(log_file, maxBytes=100 * 1024 * 1024, backupCount=10)
    formatter = logging.Formatter(
        "[%(asctime)s (%(funcName)s:%(lineno)d)] %(levelname)s in %(module)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def lark_failed(msg: str):
    from lark_alarm import send_lark_failed
    send_lark_failed("", WORK_NAME, msg)
    raise RuntimeError("Failed to {}, process should stop".format(msg))


def lark_success(msg: str):
    from lark_alarm import send_lark_success
    send_lark_success("", WORK_NAME, msg)
    # from run_lark_alarm import report_chaos_server
    # return report_chaos_server(WORK_NAME)


def md5_hash(file_path: str):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


class MultipleFileWorker:
    """"""

    def __init__(self, mount_dir: str, work_dir: str, file_nums: int = FILE_NUMS):
        self.mount_dir = mount_dir
        if not LOCAL_TEST:
            assert os.path.ismount(self.mount_dir), "{} is not a mounted dir".format(self.mount_dir)
        self.mount_dir = os.path.join(self.mount_dir, WORK_NAME)
        self.work_dir = os.path.join(work_dir, WORK_NAME)
        os.system("mkdir -p {}".format(self.mount_dir))
        os.system("mkdir -p {}".format(self.work_dir))
        self.file_nums = file_nums
        log_dir = os.path.join(work_dir, "log")
        os.system("mkdir -p {}".format(log_dir))
        self.logger = get_logger(log_dir)
        self.logger.info("Succeed to init MultipleFileWorker, mount_dir:{}, work_dir:{}".format(mount_dir, work_dir))

    @staticmethod
    def generate_name():
        return "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(FILE_NAME_LENGTH)])

    @staticmethod
    def generate_content(size: int):
        assert size > 0, "size should be larger than 0"
        return "".join([random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(size)])

    def check_file(self, file_name: str) -> (bool, str):
        mount_file_path = os.path.join(self.mount_dir, file_name)
        local_file_path = os.path.join(self.work_dir, file_name)
        mount_file_md5 = md5_hash(mount_file_path)
        local_file_md5 = md5_hash(local_file_path)
        mount_file_size = str(os.path.getsize(mount_file_path))
        local_file_size = str(os.path.getsize(local_file_path))
        msg = None
        if mount_file_md5 != local_file_md5:
            msg = "file {} md5 not match, mount:{}, local:{}".format(file_name, mount_file_md5, local_file_md5)
            self.logger.error(msg)
            return False, msg
        if mount_file_size != local_file_size:
            msg = "file {} size not match, mount:{}, local:{}".format(file_name, mount_file_size, local_file_size)
            self.logger.error(msg)
            return False, msg
        # only record mount_file meta
        ctime = str(os.path.getctime(mount_file_path))
        mtime = str(os.path.getctime(mount_file_path))
        msg = local_file_md5 + SEPARATOR + str(local_file_size) + SEPARATOR + ctime + SEPARATOR + mtime
        return True, msg

    def random_write(self, file_name: str):
        size = os.path.getsize(os.path.join(self.work_dir, file_name))
        seek_off = random.randint(0, size)  # assure random write_size <= file_size
        write_size = random.randint(1, MAX_WRITE_LENGTH)
        content = self.generate_content(write_size)
        self.logger.info(
            "Before random write, file_name:{}, write seek_off:{}, file size:{}, write size:{}".format(file_name,
                                                                                                       seek_off, size,
                                                                                                       write_size))
        with open(os.path.join(self.work_dir, file_name), "r+") as file:
            file.seek(seek_off)
            file.write(content)
        with open(os.path.join(self.mount_dir, file_name), "r+") as file:
            file.seek(seek_off)
            file.write(content)
        self.logger.info(
            "Success write file_name:{}, seek_off:{}, write_size:{}".format(file_name, seek_off, write_size))

    def random_truncate(self, file_name: str):
        origin_size = os.path.getsize(os.path.join(self.work_dir, file_name))
        truncate_size = random.randint(1, origin_size) # old 1.3 cluster not support truncate large
        # truncate_size = random.randint(1, MAX_TRUNCATE_SIZE)
        self.logger.info(
            "Before random truncate, file_name:{}, origin size:{}, truncate size:{}".format(file_name, origin_size,
                                                                                            truncate_size))
        with open(os.path.join(self.mount_dir, file_name), "r+") as file:
            file.truncate(truncate_size)
        with open(os.path.join(self.work_dir, file_name), "r+") as file:
            file.truncate(truncate_size)
        self.logger.info(
            "Success truncate file_name:{}, origin size:{}, truncate size:{}".format(file_name, origin_size,
                                                                                     truncate_size))

    def handle_one_file(self):
        file_name = self.generate_name()
        self.logger.info("Start handle file_name:{}".format(file_name))
        os.system("touch {}".format(os.path.join(self.work_dir, file_name)))
        os.system("touch {}".format(os.path.join(self.mount_dir, file_name)))
        msg = None
        for _ in range(random.randint(1, OP_NUMS)):
            self.random_write(file_name)
            self.random_write(file_name)
            result, msg = self.check_file(file_name)
            if not result:
                lark_failed(msg)
                return
            self.random_truncate(file_name)
            result, msg = self.check_file(file_name)
            if not result:
                lark_failed(msg)
                return
        _, msg = self.check_file(file_name)
        key = file_name + SEPARATOR + msg
        msg_queue.put(key)
        os.system("rm -rf {}".format(os.path.join(self.work_dir, file_name)))
        self.logger.info("Success handle file_name:{}, msg:{}".format(file_name, msg))

    def dump_meta(self):
        """only keep up a meta file"""
        meta_file_path = os.path.join(self.work_dir, META_FILE)
        with open(meta_file_path, "w") as file:
            for k in msg_queue.queue:
                file.write(k + "\n")
        self.logger.info("Dump meta file:{}".format(meta_file_path))

    def run(self):
        for _ in range(self.file_nums):
            self.handle_one_file()
        self.dump_meta()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount_dir", dest="mount_dir", type=str, required=False, default=DEFAULT_MOUNT_DIR,
                        help="挂载目录")
    parser.add_argument("--work_dir", dest="work_dir", type=str, required=False, default=DEFAULT_WORK_DIR,
                        help="工作目录")
    parser.add_argument("--local", dest="local", type=bool, required=False, default=LOCAL_TEST,
                        help="本地测试模式")
    parser.add_argument("--mode", dest="mode", type=int, required=False, default=Mode.WriteAndCheckData,
                        help="运行模式")
    parser.add_argument("--prefix", dest="prefix", type=str, required=False, default="prefix",
                        help="文件标志")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    LOCAL_TEST = args.local
    global WORK_NAME
    WORK_NAME = WORK_NAME +"_" +  args.prefix
    if args.local:
        args.mount_dir = os.path.join(os.getcwd(), WORK_NAME, "data")
        os.system("mkdir -p {}".format(args.mount_dir))
    writer_worker = MultipleFileWorker(args.mount_dir, args.work_dir)
    writer_worker.run()


if __name__ == "__main__":
    main()
