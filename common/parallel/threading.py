def timeit(func):
    def result():
        start_time = time.time()
        func()
        end_time = time.time()
        print("cost:%.2fs" % (end_time - start_time))
    return result

@timeit
def shuffles():
    time.sleep(1)
shuffles()

def max_number(*args):
    if len(args) == 0:
        return 0
    max_n = args[0]
    for num in args:
        max_n = max(max_n, num)
    return max_n
print(max_number(1,2,3,4,5,6,7,8,9,10))

def max_number2(*args):
    return max(list(args))
print(max_number2(1,2,3,4,5))

class People(object):
    country = "CN"
    @staticmethod
    def getCountry():
        return People.country
    @classmethod
    def setCountry(cls, country):
        cls.country = country

p = People()
print(p.getCountry())
print(People.getCountry())

import threading
import time
time_cost = 0.1
def sing():
    for i in range(3):
        print("sing...\n")
        time.sleep(time_cost)

def dance():
    for i in range(3):
        print("dance...\n")
        time.sleep(time_cost)

def task(count):
    for i in range(count):
        print("task doing...")
        time.sleep(time_cost)
    print("done")

if __name__ == "__main__":
    sing_thread = threading.Thread(target=sing)
    dance_thread = threading.Thread(target=dance)
    sing_thread.start()
    dance_thread.start()

    # pass args param
    sub_thread = threading.Thread(target=task, args=(4,))
    sub_thread.start()

    # pass kwargs param
    sub_thread2 = threading.Thread(target=task, kwargs={"count":3})
    sub_thread2.start()