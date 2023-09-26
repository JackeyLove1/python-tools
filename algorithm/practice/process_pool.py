import multiprocessing
import random
import time


def worker(task):
    # 执行任务的函数
    # 这里可以根据具体需求定义任务的逻辑
    # 这里只是简单返回任务的平方
    time.sleep(random.randint(1, 5))
    print("get:", task**2)
    return task**2

if __name__ == "__main__":
    # 创建进程池，指定进程池中的进程数量
    pool = multiprocessing.Pool(processes=4)
    pool_results = []
    results = []
    # 模拟一组任务
    for num in range(25):
        pool_results.append(pool.apply_async(worker, args=(num,)))

    # 使用进程池执行任务，并获取返回值
    # 关闭进程池，不再接受新的任务
    pool.close()

    # 等待所有任务完成，并获取结果
    output = [result.get() for result in pool_results]

    print(output)