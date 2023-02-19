import timeit
import time
from timeit import Timer
import numpy as np

num_actions = 5
num_agents = 7


# generate bases
def method1():
    nvec = np.array([num_actions] * num_agents)
    bases = np.ones_like(nvec)
    for i in range(1, len(bases)):
        bases[i] = bases[i - 1] * nvec[-i]
    
    return np.flip(bases)

def method2():
    bases = num_actions ** np.arange(num_agents)
    bases = np.flip(bases)
    return bases

def method3():
    bases = num_actions ** np.arange(num_agents)[::-1]
    return bases

def method4():
    bases = num_actions ** np.arange(num_agents - 1, -1, -1)
    return bases


bases = method4()

trial = int(1e5)

print(f"---generate bases---")
print(f"run for {trial} times")
timer = Timer("method1()", "from __main__ import method1")
t_ms = timer.timeit(number=trial)
print(f"method1: {t_ms} ms")

timer = Timer("method2()", "from __main__ import method2")
t_ms = timer.timeit(number=trial)
print(f"method2: {t_ms} ms")

timer = Timer("method3()", "from __main__ import method3")
t_ms = timer.timeit(number=trial)
print(f"method3: {t_ms} ms")

timer = Timer("method4()", "from __main__ import method4")
t_ms = timer.timeit(number=trial)
print(f"method4: {t_ms} ms")

#################
# int to base

def int2base(x, base, bases):
    digits = (x.reshape(x.shape + (1,)) // bases) % base
    
    return digits

def int2base2(act: np.ndarray) -> np.ndarray:
    # action() in my_dqn
    converted_act = []
    for b in bases:
    # for b in np.flip(bases):
        converted_act.append(act // b)
        act = act % b
    res = np.array(converted_act, dtype=int).transpose()
    return res

trial = 20000
acts = np.array([np.random.randint(num_actions, size=num_agents) for _ in range(trial)])
buffer_act = np.array([np.sum(bases * acts[i]) for i in range(trial)])

print(f"---int to bases---")
print(f"run for {trial} times")
t_start = time.time()
res = int2base(buffer_act, num_actions, bases)
t_end = time.time()

print(f"correctness: {(res == acts).all()}")
print(f"int2base(): {t_end - t_start} ms")

t_start = time.time()
res = int2base2(buffer_act)
t_end = time.time()

print(f"correctness: {(res == acts).all()}")
print(f"int2base2(): {t_end - t_start} ms")

