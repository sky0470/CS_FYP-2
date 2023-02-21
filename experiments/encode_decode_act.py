import numpy as np

nvec = np.array([5, 5, 5, 5, 5, 5, 5])
bases = np.ones_like(nvec)
for i in range(1, len(bases)):
    bases[i] = bases[i - 1] * nvec[-i]

def action(act: np.ndarray) -> np.ndarray:
    converted_act = []
    for b in np.flip(bases):
        converted_act.append(act // b)
        act = act % b
    return np.array(converted_act).transpose()

def reverse(act):
    act = act.flatten()
    ret = 0
    for i, b in enumerate(np.flip(act)):
        ret = ret + b * bases[i]
    return ret

print(reverse(np.array([1,1,1,1,1,1,1])))
x = np.array([19530])
print(action(x))
print(reverse(action(x)))
