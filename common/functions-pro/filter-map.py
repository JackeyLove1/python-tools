item1 = list(map(lambda x : x ** 2, filter(lambda x : x % 2, range(1, 10))))
print(item1)

item2 = [x ** 2 for x in range(1, 10) if x % 2]
print(item2)

from functools import reduce
from operator import add
print(reduce(add, range(1, 101)))
print(sum(range(1, 101)))

import random
print(sorted([random.randint(0, 100) for _ in range(10)], key=lambda x : -x))

