def counter(maximum=100):
    i= 0
    while i < maximum:
        yield i
        i += 1

it = counter()
for _ in range(10):
    print(next(it))


def count(firstval=0, step=1):
    x = firstval
    while 1:
        yield x
        x += step
it = count()
for _ in range(10):
    print(next(it))

