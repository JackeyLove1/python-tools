from collections import OrderedDict

d = OrderedDict()
d["1"] = 1
d["2"] = 2
d["3"] = 3
d["4"] = 4
print(d)
for k in d.keys():
    print(k)
d["one"] = 1
print(d)


class Foo:
    pass


print(hash(Foo))

import copy

items = [1, 2, 3, 4]
item2 = copy.copy(items)
item3 = copy.deepcopy(items)
print(item2)
print(items == item2)
print(items is item2)
print(item2 == item3)

from collections import defaultdict
int_dict = defaultdict(int)
int_dict["foo"] += 1
print(int_dict)

def generate(max_number:int):
    for i in range(max_number):
        if i % 2 == 0:
            yield i

for i in generate(10):
    print(i)

import random
nums = [random.randint(0, 10) for _ in range(10) ]
print(nums)
print(set(nums))
print(OrderedDict.fromkeys(nums).keys())

from typing import List
def remove_even(numbers : List[int]):
    for number in numbers:
        if number % 2 == 0:
            numbers.remove(number)
nums = [1,2,4,5,6,8,11]
remove_even(nums)
print(nums)