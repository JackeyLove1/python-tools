import time
import threading
from concurrent.futures import ThreadPoolExecutor

class Account(object):
    def __init__(self):
        self.balance = 0.0
        self.lock = threading.Lock()
    def deposit(self, money):
        with self.lock:
            new_balance = self.balance + money
            time.sleep(0.01)
            self.balance = new_balance

def main():
    account = Account()
    pool = ThreadPoolExecutor(max_workers=10)
    futures = []
    for _ in range(100):
        future = pool.submit(account.deposit, 1)
        futures.append(future)
    pool.shutdown()
    for future in futures:
        future.result()
    print(account.balance)

if __name__ == '__main__':
    main()