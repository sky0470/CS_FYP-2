import time
import numpy as np

num_actions = 5
num_agents = 7


d = {}
shape = (100, 100)
n = np.prod(shape)
for x in range(1000):
    d[str(x)] = np.arange(n).reshape(shape)

start = time.time()
a = np.array([list(d.values())]*50 )
print(a.shape)
end = time.time()
print(f'{end - start}')

start = time.time()
b = np.tile(np.array(list(d.values())), (50,)+(1,)*3)
print(b.shape)
end = time.time()
print(f'{end - start}')

print((a==b).all())
