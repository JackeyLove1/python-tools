import threading

MAX_WORKERS = 10
import time
import random
from concurrent import futures
def print_sleep(n : int):
    time.sleep(1)
    print(n)
def run():
    with futures.ThreadPoolExecutor(MAX_WORKERS) as executor:
        for _ in range(MAX_WORKERS):
            executor.submit(print_sleep, random.random())
run()
from tqdm import tqdm
for _ in tqdm(range(10)):
    time.sleep(0.01)

import asyncio
async def asyc1():
    print("Hello ... ")
    await asyncio.sleep(1)
    print("..., World")
asyncio.run(asyc1())