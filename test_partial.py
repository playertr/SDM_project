from functools import partial

def add(x, a, b):
    return x + 10 * a + 100 * b

add = partial(add, 2, 3)
print(add(4))