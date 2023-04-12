import numpy as np
shape1 = (3, 4, 2)
shape2 = (5, 3, 3, 2)
a = np.arange(np.prod(shape1)).reshape(shape1)
b = np.arange(np.prod(shape2)).reshape(shape2)
print(a)

def apply_obs_noise_norm(obs_noise):
    axes = tuple(range(1, obs_noise.ndim - 1))
    print(axes)
    mean, std = obs_noise.mean(axis=axes, keepdims=True), obs_noise.std(axis=axes, keepdims=True)
        
    print(dict(mean=mean, std=std, mean_shape=mean.shape, std_shape=std.shape))
    obs_noise_norm = (obs_noise - mean)/(std + 1e-8)
    return obs_noise_norm

a_norm = apply_obs_noise_norm(a)
print(dict(a_norm=a_norm, shape=a_norm.shape))

b_norm = apply_obs_noise_norm(b)
print(dict(b_norm=b_norm, shape=b_norm.shape))