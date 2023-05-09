import random
from time import time
import requests
from threading import Thread


class DownLoader(Thread):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        file_name = self.url[self.url.rfind('/') + 1:]
        resp = requests.get(url=self.url)
        rstr = str(int(random.random() * 10))
        file_path = './' + rstr + file_name
        with open(file_path, mode='wb') as f:
            f.write(resp.content)


def main():
    url = 'https://www.baidu.com'
    threads = []
    for i in range(10):
        t = DownLoader(url)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
