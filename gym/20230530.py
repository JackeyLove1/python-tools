from enum import Enum
class UserType(int, Enum):
    VIP = 3
    BANNED = 13

print(UserType.VIP)
print(UserType.BANNED)

import timeit
print(timeit.timeit('-'.join(str(n) for n in range(1000)), number=10000))


