# pip install retrying
import random
import time

from retrying import retry

random.seed(int(time.time()))

# retry forever without waiting.
@retry
def do_something_unreliable():
    if random.randint(0, 10) > 5:
        raise IOError("Broken sauce, everything is hosed!!!111one")
    else:
        return "Awesome sauce!"

print (do_something_unreliable())
print (do_something_unreliable())
print (do_something_unreliable())

'''
others:
@retry(stop_max_attempt_number=7)
@retry(stop_max_delay=10000)
@retry(wait_fixed=2000)
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)
'''