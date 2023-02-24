import multiprocessing
import time
time_cost = 0.1
def dance():
    for i in range(3):
        print("dancing...\n")
        time.sleep(time_cost)
def sing():
    for i in range(3):
        print("sing...\n")
        time.sleep(time_cost)

def task(count):
    for i in range(count):
        print("task...")
        time.sleep(time_cost)
    else:
        print("done...")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # pass no param
    dance_process = multiprocessing.Process(target=dance)
    sing_process = multiprocessing.Process(target=sing)
    dance_process.start()
    sing_process.start()

    # pass args param
    task_process = multiprocessing.Process(target=task, args=(5,))
    task_process.start()

    # pass kwargs param
    task_process2 = multiprocessing.Process(target=task, kwargs={"count": 10})
    task_process2.start()