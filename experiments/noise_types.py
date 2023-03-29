import numpy as np
import torch
from torch.distributions import Normal

num_agents = 6
obs_range = 3
logit_output_dim = 5
obs_shape = (num_agents, obs_range, obs_range, logit_output_dim)

prev_obs = np.arange(np.prod(obs_shape)).astype('float')

num_noise_type = 2
num_noise_per_type = 1
num_noise_per_agent = num_noise_type * num_noise_per_type

noise = np.arange(num_agents * num_noise_per_agent) / 100 + 0.01

# reshape prev obs and noise
prev_obs = prev_obs.reshape(obs_shape) # originally hardcoded as (5, 3, 3, 5) (n_agents, obs_range, obs_range, obs_dims)
noise = noise.reshape(num_agents, -1) # noise: (num_agent, num_noise_per_agent)
num_noise_per_agent = noise.shape[1]
print(dict(prev_obs=prev_obs,
           prev_obs_shape=prev_obs.shape,
           noise=noise, 
           noise_shape=noise.shape
           ))

# apply noise to predator and prey dim
obs_noise = prev_obs
for i in range(num_agents):
    obs_noise[i, :, :, :2] += noise[i, :]
np.set_printoptions(suppress = True)
print(dict(obs_noise=obs_noise))
print(f"agent0:\n {obs_noise[0]}")
    
mu = torch.tensor([5., -2.])
sig = torch.tensor([0.3, 0.3])
# sample value
normal = Normal(loc=mu, scale=sig)
samples = normal.sample(torch.tensor([9]))
print(f"sample from Normal(mu={mu}, sig={sig}), get: {samples}, prob: {normal.log_prob(samples).exp()}")

noise_flat = noise.flatten()
noise_recover = noise_flat.reshape((num_agents, num_noise_per_agent))
print(dict(
    noise_flat=noise_flat,
    noise_recover=noise_recover,
    result=(noise == noise_recover).all(),
))
