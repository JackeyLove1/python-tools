import os
import random

from BigFileTest import *
import unittest


class TestBigFile(unittest.TestCase):
    def setUp(self) -> None:
        self.args = BigFileTestArgs()
        self.args.mount_path = os.getcwd()
        self.args.file_size = 20 * 1024 * 1024
        self.args.chunk_size = 1024 * 4
        self.args.debug = True
        self.args.max_test_write_length = 50 * 1024 * 1024
        self.args.max_test_truncate_size= 100 * 1024 * 1024
        self.args.local_mode = True
        self.args.setUp()

    def tearDown(self) -> None:
        # os.system("rm -rf {}".format(self.args.file_name))
        pass

    def test_write_file(self):
        offset = 0
        for index in range(len(self.args.chunk_mode)):
            write_fixed_mode(offset, index, self.args)
            data = read_data_with_offset(self.args.file_name, offset, self.args.chunk_size)
            offset += self.args.chunk_size
            print("Expected:{}, Actual:{}".format(self.args.chunk_mode[index], data))
            self.assertEqual(self.args.chunk_mode[index], data)
            print("write list:", self.args.write_list)
            self.assertEqual(self.args.write_list[index], data[0])

    def test_calculate_md5_partial(self):
        for idx in range(100):
            offset1 = idx * self.args.chunk_size
            offset2 = (idx + 1) * self.args.chunk_size
            index = idx % len(self.args.chunk_mode)
            write_fixed_mode(offset1, index, self.args)
            md51 = calculate_md5_partial(self.args.file_name, offset1, offset2)
            md52 = self.args.md5_mode[index % len(self.args.md5_mode)]
            print("Expected: {}, Actual:{}".format(md52, md51))
            self.assertEqual(md51, md52)

    def test_calculate_md5_cycle(self):
        write_bigfile(self.args)
        for idx in range(self.args.write_nums):
            offset = self.args.chunk_size * idx
            length = self.args.chunk_size * random.randint(1, 10)
            data = read_data_with_offset(self.args.file_name, offset, length)
            expected_md5 = calculate_md5_str(data)
            actual_md5 = calculate_md5_without_read(self.args, offset, offset + length)
            print("expected:{}, actual:{}".format(expected_md5, actual_md5))
            self.assertEqual(expected_md5, actual_md5)

    def test_random_write_simple(self):
        write_bigfile(self.args)
        for _ in range(200):
            offset = self.args.chunk_size * random.randint(0, self.args.write_nums - 1)
            length = self.args.chunk_size * random.randint(1, self.args.max_test_write_length // self.args.chunk_size)
            write_data_length(self.args, offset, length)
            # check write range md5
            begin_index = offset // self.args.chunk_size
            end_index = begin_index + length // self.args.chunk_size
            write_mode_list = self.args.write_list[begin_index:end_index]
            read_data = read_data_with_offset(self.args.file_name, offset, length)
            actual_md5 = calculate_md5_str(read_data)
            expected_md5 = calculate_md5_mode_list(self.args, write_mode_list)
            self.assertEqual(actual_md5, expected_md5)
        check_total_file(self.args)

    def test_random_write(self):
        write_bigfile(self.args)
        for _ in range(100):
            random_write(self.args)

    def test_random_truncate(self):
        write_bigfile(self.args)
        for _ in range(100):
            random_truncate(self.args)

    def test_random_integration(self):
        write_bigfile(self.args)
        for _ in range(50):
            random_write(self.args)
            random_truncate(self.args)

    def test_write_specific_file(self):
        for _ in range(100):
            expected_size = random.randint(1, 1000) * self.args.chunk_size
            self.args.file_size =expected_size
            write_bigfile(self.args)
            actual_size = os.path.getsize(self.args.file_name)
            print("file_name:{}, expected_size: {}, actual size:{}".format(self.args.file_name, expected_size, actual_size))
            self.assertEquals(actual_size, self.args.file_size)
            os.system("rm -rf {}".format(self.args.file_name))

    def test_truncate_specific_size(self):
        write_bigfile(self.args)
        for _ in range(100):
            truncate_size = random.randint(1, 100) * self.args.chunk_size
            truncate_file(self.args, truncate_size)
            self.assertEqual(os.path.getsize(self.args.file_name), truncate_size)