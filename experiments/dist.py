import numpy as np
import matplotlib.pyplot as plt


def dist1d(a, b):
    return (a - b) ** 2


def dist2d(data):
    # return ((data[0, 0] - data[1, 0]) ** 2 + (data[0, 1] - data[1, 1]) ** 2) **.5
    return abs(data[0, 0] - data[1, 0])+ abs(data[0, 1] - data[1, 1])


n = int(1e4)
size = 10
data2d = np.random.randint(size + 1, size=(n, 2, 2))

# data2d[:, 0] = np.array([size // 2, size // 2])
data2d[:, 0] = np.array([size, size])
y = np.array([dist2d(d) for d in data2d])

print(f'dist mean is {y.mean()}')
print(f'dist std is {y.std()}')

rwd = lambda step: step * 0.05 + (40-step) * 0.5 *5

r = rwd(y)
print(f'rwd mean is {r.mean()}')
print(f'rwd std is {r.std()}')


# dmean, dstd = 10, 4.48
dmean, dstd = 20, 5.48

r = np.zeros((n,3))
for i in range(n):
    d = np.random.normal(dmean, dstd, (3))
    for j in range(40):
        arr = np.sum(d < j)
        r[i] = r[i] + np.where(d<j, 0.5 * arr, -0.05)

print(r.mean())
print(r.std())


import sys
sys.exit()

plt.subplot(1, 2, 1)
plt.hist(y, bins=10, rwidth=.7)

plt.subplot(1, 2, 2)
plt.yscale('log')
plt.hist(y, bins=10, rwidth=.7)

plt.show()


# plt.hist(y, bins=10, rwidth=0.7)
# plt.yscale('log')
# plt.show()
