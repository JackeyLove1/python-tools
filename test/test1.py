import argparse

parser = argparse.ArgumentParser(description="help")
from loguru import logger
logger.info("logging test")

class Student:
    name = "unkown"
    def __int__(self):
        self.age = 20

    @classmethod
    def ToString(cls):
        return "Student Class Attrutes: name={}".format(cls.name)

logger.info(Student.ToString())

import re
ch_pattern = re.compile(r"[\u4e00-\u9fa5]+", re.M|re.I)
html = r'<p class="board-content">榜单规则：将猫眼电影库中的经典影片，按照评分和评分人数从高到低综合排序取前100名，每天上午10点更新。相关数据来源于“猫眼电影库”。</p>'
match = ch_pattern.findall(html)
logger.info("".join(match))