import threading
import time
import datetime
from concurrent.futures import ThreadPoolExecutor
from random import randint
from time import sleep
import typing


class Account:
    def __init__(self, balance=0):
        self.balance = balance
        lock = threading.RLock()
        self.condition = threading.Condition(lock)

    def withdraw(self, money):
        with self.condition:
            while money > self.balance:
                self.condition.wait()
            new_balance = self.balance - money
            sleep(0.01)
            self.balance = new_balance

    def deposit(self, money):
        with self.condition:
            new_balance = self.balance + money
            sleep(0.01)
            self.balance = new_balance
            self.condition.notify_all()


cost: int = 3


def add_money(account: Account):
    start_time = datetime.datetime.now()
    while (datetime.datetime.now() - start_time).total_seconds() < cost:
        money = randint(5, 10)
        account.deposit(money)
        print(threading.current_thread().name, ":", money, "=====>", account.balance)
        sleep(0.5)


def sub_money(account: Account):
    start_time = datetime.datetime.now()
    while (datetime.datetime.now() - start_time).total_seconds() < cost:
        money = randint(10, 30)
        account.withdraw(money)
        print(threading.current_thread().name,
              ':', money, '<====', account.balance)
        sleep(1)


def main():
    account = Account()
    with ThreadPoolExecutor(max_workers=15) as pool:
        for i in range(10):
            pool.submit(add_money, account)
        for i in range(5):
            pool.submit(sub_money, account)
    print("total left:", account.balance)


if __name__ == "__main__":
    main()
