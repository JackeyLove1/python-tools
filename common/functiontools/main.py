import time
from functools import cache, lru_cache


@cache
def fab(n: int):
    return n * fab(n - 1) if n > 1 else 1


start = time.time()
print(fab(10))
print(fab(5))


@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return 1
    return fib(n - 1) + fib(n - 2)
print([fib(n) for n in range(10)])
print(fib.cache_info())