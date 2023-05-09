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

# process map
def main(url):
    html = request_douban(url)
    soup = BeautifulSoup(html, 'lxml')
    save_content(soup)

if __name__ == '__main__':
    start = time.time()
    urls = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for i in range(0, 10):
        url = 'https://movie.douban.com/top250?start=' + str(i * 25) + '&filter='
        urls.append(url)
    pool.map(main, urls)
    pool.close()
    pool.join()