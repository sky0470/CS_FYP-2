import numpy as np
import time

obs = np.random.rand(1024, 7, 5, 5, 3)
# obs = batch x agent x obs
# extract obs shape = (1024, 1, 5, 5, 3)

start = time.time()
for i in range(1000):
    a = np.expand_dims(obs[:, i%7], axis=1)
end = time.time()
print(f"expand takes {end-start}")


start = time.time()
for i in range(1000):
    a = obs[:, (i%7,)]
end = time.time()
print(f"direct takes {end-start}")
