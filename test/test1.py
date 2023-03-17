import os
import time
import functools
from functools import reduce

def char2Int(s):
    return reduce(lambda x, y: 10 * x + y, map(lambda ch:int(ch), s))
print(char2Int("12345"))

def log(func):
    def wrapper(*args, **kwargs):
        print("call: {}".format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

@log
def run():
    print(time.time())
run()

def log_text(text):
    def decorator(func):
        functools.wraps(func)
        def call(*args, **kwargs):
            print("text:{}, call:{}".format(text, func.__name__))
            return func(*args, **kwargs)
        return call
    return decorator

@log_text("run2")
def run2():
    print(time.time())
run2()

int2 = functools.partial(int, base=2)
int10 = functools.partial(int, base=10)
print(int2("1000"))
print(int10("1000"))

class Student(object):
    def __init__(self):
        self.score = 0
    @property
    def score(self):
        return self.score

    @score.setter
    def score(self, score):
        self.score = score

class Fib():
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 1000:
            raise StopIteration
        return self.a

    def __getitem__(self, item):
        a, b = 0, 1
        for i in range(item):
            a, b = b, a + b
        return a
for n in Fib():
    print(n)

f = Fib()
print(f[20])

print([x for x in os.listdir('.') if os.path.isdir(x)])

import multiprocessing
import random
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random())
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

import base64
base64.b64encode("binary\x00string")


