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



obs = np.random.rand(1024, 7, 5, 5, 3)
obs_tensor = torch.from_numpy(obs)
# obs = batch x agent x obs
# extract obs shape = (1024, 1, 5, 5, 3)

@timing
def expand(trial=10000):
    b = []
    for i in range(trial):
        a = np.expand_dims(obs[:, i%7], axis=1)
        b.append(a)

    return b

@timing
def np_newaxis(trial=10000):
    b = []
    for i in range(trial):
        a = obs[:, np.newaxis, i%7]
        b.append(a)

    return b

@timing
def np_none(trial=10000):
    b = []
    for i in range(trial):
        a = obs[:, None, i%7]
        b.append(a)

    return b

@timing
def torch_unsqueeze(trial=10000):
    b = []
    for i in range(trial):
        a = torch.unsqueeze(obs_tensor[:, None, i%7], 1)
        b.append(a)

    return b

@timing
def direct(trial=10000):
    b = []
    for i in range(trial):
        a = obs[:, (i%7,)]
        b.append(a)

    return b

ans_expand = expand()
ans_np_newaxis = np_newaxis()
ans_np_none = np_none()
ans_direct = direct()
ans_torch_unsequeeze = torch_unsqueeze()
print(ans_expand == ans_torch_unsequeeze)
