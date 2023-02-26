# The asynchronous execution can be performed with threads, using ThreadPoolExecutor,
# or separate processes, using ProcessPoolExecutor
# ThreadPoolExecutor Example
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))

# ProcessPoolExecutor Example
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()

import time
def make_request():
    pass
# single thread
def main():
    for num in range(1, 101):
        make_request(num)

# concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait
def main():
    futures = []
    with ThreadPoolExecutor() as executor:
        for num in range(1, 101):
            futures.append(executor.submit(make_request, num))
    wait(futures)

# asyncIO
import asyncio
import time
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *[make_request_async(num, client) for num in range(1, 101)]
        )
loop = asyncio.get_event_loop()
loop.run_until_complete(main())



if __name__ == "__main__":
    start_time = time.perf_counter()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed run time: {elapsed_time} seconds")
if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Elapsed run time: {end_time - start_time} seconds.")