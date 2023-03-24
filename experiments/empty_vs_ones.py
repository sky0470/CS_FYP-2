import numpy as np
import time
import torch

def timing(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()

        print(f"function: {func.__name__}({args}, {kwargs}) ran for {t2 - t1} sec")
    return wrapper

@timing
def np_empty(trial=10000):
    for i in range(trial):
        a = np.empty((5, 3, 3, 5))

@timing
def np_ones(trial=10000):
    for i in range(trial):
        a = np.ones((5, 3, 3, 5))

np_empty()
np_ones()

