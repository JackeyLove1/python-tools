'''
排查update_blob
'''
import argparse
import os
from collections import defaultdict


class Blob:
    def __init__(self, name: str, size: str):
        '''
        :param name: '200_3395_115_2/log_118'
        :param size: '4224'
        '''
        self.name = name
        self.size = int(size)
        parts = self.name.split("/")
        assert len(parts) == 2, "Failed to parse {} into two parts".format(str(parts))
        part1, part2 = parts
        self.type = part2.split("_")[0]
        self.index = int(part2.split("_")[-1])
        self.volume_id = int(part1.split("_")[1])
        self.segment_id = int(part1.split("_")[2])

    @property
    def Index(self):
        return self.index

    @property
    def Size(self):
        return self.size

    @property
    def Type(self):
        return self.type

    @property
    def Segment(self):
        return self.segment_id

    @property
    def Volume(self):
        return self.volume_id

    def __repr__(self):
        return "name:{}, volume:{}, segment:{},type:{}, size:{}" \
            .format(self.name, self.volume_id, self.segment_id,
                    self.type, self.size)

    def IsEntry(self):
        return "entry" in self.type

    def IsUpdate(self):
        return "update" in self.type


def main():
    parser = argparse.ArgumentParser(description="协助排查BS可能加载的dupdate blob超过一定阈值的工具")
    parser.add_argument("--file", dest="path", type=str, default="blob.size", help="bytestore blob stat list")
    parser.add_argument("--size", dest="size", type=int, default=1024, help="默认查找大于1k的有效update blob")
    args = parser.parse_args()
    file_path = args.path
    file_size = args.size
    if not os.path.isfile(file_path):
        raise FileNotFoundError("{} 文件不存在".format(file_path))
    segment_entry_index = {}  # segment_id => min_entry_index
    segment_update_blob_map = defaultdict(dict)  # segment_id => update_map {update_blob_index : [update_blob_name]}
    with open(file_path, "r") as f:
        for line in f:
            # print(line.strip().split())
            parts = line.strip().split()
            if "tmp" in parts[0]:
                continue
            if "entry" not in parts[0] and "update" not in parts[0]:
                continue
            blob = Blob(parts[0], parts[1])
            segment_id = blob.Segment
            if blob.IsEntry():
                if segment_entry_index.get(segment_id) is None:
                    segment_entry_index[segment_id] = blob.Index
                else:
                    segment_entry_index[segment_id] = min(segment_entry_index[segment_id], blob.Index)
            if blob.IsUpdate():
                update_map = segment_update_blob_map[segment_id]
                if update_map.get(blob.Index) is None:
                    update_map[blob.Index] = []
                update_map[blob.Index].append(blob)
    # filter segment_update_blob_map by segment_entry_index
    for segment_id, entry_idx in segment_entry_index.items():
        if segment_update_blob_map.get(segment_id) is not None:
            update_map = segment_update_blob_map[segment_id]
            for update_index, update_list in update_map.items():
                if update_index < entry_idx:
                    update_list = []  # clear not delete
    # merge and print update blob
    count = 0
    for _, update_map in segment_update_blob_map.items():
        for _, update_list in update_map.items():
            for update_blob in update_list:
                if update_blob.Size > file_size:
                    print(update_blob)
                    count += 1
    print("Count:", count)


if __name__ == "__main__":
    main()
