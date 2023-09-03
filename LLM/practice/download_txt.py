from loguru import logger
import urllib.request
from retrying import retry
import time
import random
import os

os.system("mkdir -p data")
random.seed(int(time.time()))
id = 1
target = 10000
while id <= target:
    filename = f"./data/{id}.txt"
    url = f"https://www.txt99.cc/api/txt_down.php?articleid={id}"
    id += 1
    try:
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Succeed to Download {url}")
        time.sleep(random.randint(0, 5))
    except Exception as e:
        logger.error(f"Failed to download url{url}, Error:{e}")
