import threading
import time

g_num = 0
num = 1000
lock = threading.Lock()

def add_num1():
    lock.acquire()
    for i in range(num):
        global g_num
        g_num += 1
        time.sleep(0.001)
    print("sum1:", g_num)
    lock.release()

def add_num2():
    lock.acquire()
    for i in range(num):
        global g_num
        g_num += 1
    print("sum2:", g_num)
    lock.release()

if __name__ == "__main__":
    first = threading.Thread(target=add_num1)
    second = threading.Thread(target=add_num2)
    first.start()
    second.start()