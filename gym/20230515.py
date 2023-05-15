import functools

fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
print(sorted(fruits, key=lambda word: word[::-1]))
import random
from typing import List


class BingoCage:
    def __init__(self, items: List):
        self._items = items
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError("pick from empty BingoCage")

    def __call__(self, *args, **kwargs):
        return self.pick()


bingo = BingoCage(list(range(5)))
print(bingo.pick())
print(bingo())
callable(bingo)
# print(dir(bingo))

from functools import reduce


def fact(n: int):
    return reduce(lambda a, b: a * b, range(1, n + 1))


print(fact(10))
import operator


def fact2(n: int):
    return reduce(operator.mul, range(1, n + 1))


print(fact2(10))

uppercase = operator.methodcaller("upper")
print(uppercase("hello, world"))

triple = functools.partial(operator.mul, 3)
print(list(map(triple, list(range(1, 10)))))

from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])
p = Point(1, 2)
print(p[0], p[1])
print(p.x, p.y)
print(p)
print(p._asdict())
print(p._make([2, 3]))
print(p._fields)


def make_averager():
    count, total = 0, 0

    def inner_avg(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return inner_avg


avg = make_averager()
print(avg(1))
print(avg(2))
print(avg(3))

import time


def count_time(func):
    def wrapper():
        start = time.time()
        func()
        print(f"cost:{time.time() - start}s")

    return wrapper


@count_time
def print_hello():
    time.sleep(0.05)
    print("hello, world")


print_hello()

import functools


def clock(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        name = func.__name__
        arg_lst = []
        if args:
            arg_lst.append(', '.join(repr(arg) for arg in args))
        if kwargs:
            pairs = ['%s=%r' % (k, w) for k, w in sorted(kwargs.items())]
            arg_lst.append(', '.join(pairs))
        arg_str = ', '.join(arg_lst)
        print('[%0.8fs]%s(%s)->%r ' % (elapsed, name, arg_str, result))
        return result

    return clocked


@clock
def calculate(a: int, l: List, d: dict):
    time.sleep(0.05)
    print(a)
    print(l)
    print(d)


calculate(1, [2, 3], {4: 5})
mod = 1e7 + 9


@clock
def fabonacci(n: int):
    if n < 2:
        return n
    return fabonacci(n - 1) + fabonacci(n - 2)


fabonacci(10)


@clock
def factorial1(n: int):
    return n * fabonacci(n - 1) if n else 1


@clock
@functools.cache
def factorial2(n: int):
    return n * fabonacci(n - 1) if n else 1


fab_n = 2
factorial1(fab_n)
factorial2(fab_n)

from array import array
import math


class Vector2d:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __iter__(self):
        return (i for i in (self.x, self.y))

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, self.x, self.y)

    def __str__(self):
        return str(tuple(self))

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

    def __getattr__(self, item):
        pass

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __mul__(self, other):
        pass


from datetime import datetime

print(datetime.now())
print(format(datetime.now(), "%H:%M:%S"))

for item in zip([1, 2, 3], ["apple", "banana"]):
    print(item)

my_list = [[1, 2, 3], [40, 50, 60], [9, 8, 7]]
print(functools.reduce(lambda a, b: a + b, [sub[1] for sub in my_list]))
print(sum(sub[1] for sub in my_list))
print(sum(sum(sub) for sub in my_list))

import re
import reprlib

RE_WORD = re.compile('\w+')


class Sentence:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return "Sentence(%s)" % reprlib.repr(self.text)

    def __iter__(self):
        return SentenceIterator(self.words)


class SentenceIterator:
    def __init__(self, words):
        self.words = words
        self.index = 0

    def __next__(self):
        try:
            word = self.words[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return word

    def __iter__(self):
        return self


s = Sentence('Hello World')
for word in s:
    print(word)
print("s[1]:", s[1])
it = iter(s)
print(next(it))
print(next(it))

def gen_123():
    yield 1
    yield 2
    yield 3
print(gen_123)
for i in gen_123():
    print(i)
g = gen_123()
print(next(g))
print(next(g))
print(next(g))

def gen_AB():
    print("start")
    yield 'A'
    print("continue")
    yield 'B'
    print('end')

for i in gen_AB():
    print(i)
res2 = (i for i in gen_AB())
for i in res2:
    print(i)

def fibnacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
f = fibnacci()
print([next(f) for _ in range(10)])
print(*[next(f) for _ in range(10)])
print(''.join(str(next(f)) for _ in range(10)))

def simple_coroutine():
    print("-> coroutine start")
    x = yield
    print("-> coroutine received:", x)
my_co = simple_coroutine()
next(my_co)

def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count
co_avger = averager()
print(next(co_avger))
print(co_avger.send(10))
print(co_avger.send(20))
print(co_avger.send(30))
