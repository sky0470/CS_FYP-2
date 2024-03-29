from collections import defaultdict
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

# from pettingzoo.sisl import pursuit_v4
from pursuit_msg.my_sisl import pursuit_v4
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import ParallelEnv
from tianshou.env import MultiDiscreteToDiscrete

# from gymnasium.vector.utils import batch_space
from pursuit_msg.my_gym_vector_utils.spaces import batch_space
from pursuit_msg.my_gym_vector_utils.wrapper import MultiDiscreteToDiscreteNoise

import numpy as np

from .my_pursuit import aec_to_parallel_wrapper

import wandb
import random

SEED = 42

ActionType = Optional[int]
AgentID = str
ActionDict = Dict[AgentID, ActionType]
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")

AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]


# from utils.conversion.py
def my_parallel_wrapper_fn_noise(env_fn, seed=None):
    def par_fn(**kwargs):
        has_noise = kwargs.pop("has_noise")
        noise_shape = kwargs.pop("noise_shape")
        apply_noise = kwargs.pop("apply_noise", 1)
        obs_noise_norm = kwargs.pop("obs_noise_norm", 0)
        assert has_noise, "this is noise env"

        env = env_fn(**kwargs)
        env = aec_to_parallel_wrapper_noise(env, seed,
                                            noise_shape=noise_shape,
                                            apply_noise=apply_noise, 
                                            obs_noise_norm=obs_noise_norm)
        env = MultiDiscreteToDiscreteNoise(env,
                                           has_noise=has_noise,
                                           noise_shape=noise_shape
        )
        return env

    return par_fn


# from utils.conversion.py
class aec_to_parallel_wrapper_noise(aec_to_parallel_wrapper):
    def __init__(self, aec_env, seed=None, noise_shape=0, apply_noise=1, obs_noise_norm=0):
        aec_env.reset()
        self.observation_space_ = batch_space(
            batch_space(
                aec_env.unwrapped.observation_space_ic3("pursuer_0"), aec_env.num_agents
            ),
            aec_env.num_agents,
        )
        super().__init__(aec_env, seed)
        self.noise_shape = noise_shape
        self.apply_noise = apply_noise
        self.obs_noise_norm = obs_noise_norm

    @property
    def observation_space(self):
        return self.observation_space_

    def cal_dist(self, a,b):
        obs_range = a.shape[0]
        idx = obs_range // 2
        xa, ya = a[idx, idx, 3], a[idx, idx, 4]
        xb, yb = b[idx, idx, 3], b[idx, idx, 4]
        # (([xa], [ya])) = np.where(a[:,:,0]==1)
        # (([xb], [yb])) = np.where(b[:,:,0]==1)
        return (xa-xb)**2 + (ya-yb)**2

    def reset(self, seed=None, return_info=False, options=None):
        self.aec_env.reset(seed=seed, return_info=return_info, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.unwrapped.observe_ic3(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }
        obs = np.array(list(observations.values()))
        # obs_mean = obs.mean(axis=0)
        # obs_mean = np.repeat(obs_mean[np.newaxis, :], obs.shape[0], axis=0)
        # observations = np.swapaxes(np.stack([obs, obs_mean]), 0, 1)

        dist = np.array([[-1 if i==j else self.cal_dist(o, obs[i]) for (j, o) in enumerate(obs)] for i in range(obs.shape[0])])
        order = dist.argsort()
        observations = np.array([obs[order[i]] for i in range(obs.shape[0])])

        if not return_info:
            return observations
        else:
            infos = dict(**self.aec_env.infos)
            return observations, infos

    def step(self, actions):
        (actions, noise, act_noise, prev_obs) = actions
        actions = actions.astype(int)
        # prev_obs = prev_obs.reshape([5,3,3,5])
        actions = dict(zip(self.aec_env.agents, actions))

        rewards = defaultdict(int)
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        for agent in self.aec_env.agents:
            if agent != self.aec_env.agent_selection:
                if self.aec_env.terminations[agent] or self.aec_env.truncations[agent]:
                    raise AssertionError(
                        f"expected agent {agent} got termination or truncation agent {self.aec_env.agent_selection}. Parallel environment wrapper expects all agent death (setting an agent's self.terminations or self.truncations entry to True) to happen only at the end of a cycle."
                    )
                else:
                    raise AssertionError(
                        f"expected agent {agent} got agent {self.aec_env.agent_selection}, Parallel environment wrapper expects agents to step in a cycle."
                    )
            obs, rew, termination, truncation, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        terminations = dict(**self.aec_env.terminations)
        truncations = dict(**self.aec_env.truncations)
        infos = dict(**self.aec_env.infos)
        observations = {
            agent: self.aec_env.unwrapped.observe_ic3(agent)
            for agent in self.aec_env.agents
        }
        while self.aec_env.agents and (
            self.aec_env.terminations[self.aec_env.agent_selection]
            or self.aec_env.truncations[self.aec_env.agent_selection]
        ):
            self.aec_env.step(None)

        self.agents = self.aec_env.agents

        obs = np.array(list(observations.values()))
        num_agents = obs.shape[0]

        # reshape prev obs and noise
        prev_obs = prev_obs.reshape(obs.shape) # originally hardcoded as (5, 3, 3, 5) (n_agents, obs_range, obs_range, obs_dims)
        noise = noise.reshape(num_agents, -1) # noise: (num_agent, num_noise_per_agent)
        # noise.shape[1] should be same as abs(np.prod(self.noise_shape))
        
        obs_noise = prev_obs
        # apply noise to predator and prey dim
        if self.apply_noise:
            def apply_obs_noise_norm(obs_noise):
                axes = tuple(range(1, obs_noise.ndim - 1))
                mean, std = obs_noise.mean(axis=axes, keepdims=True), obs_noise.std(axis=axes, keepdims=True)
                obs_noise_norm = (obs_noise - mean)/(std + 1e-8)
                return obs_noise_norm

            if self.noise_shape == (-1, 1):
                if True:
                    obs_noise += noise[:, None, None, :] 
                    if self.obs_noise_norm:
                        obs_noise = apply_obs_noise_norm(obs_noise)
                else: # override here to change to toggle noise
                    for i in range(5):
                        if noise[i, 0] < 0:
                            obs_noise[i] = 0
            elif self.noise_shape == (2, 1):
                # obs_noise[:, :, :, :2] += noise[:, None, None, :]
                obs_noise_slice = obs_noise[:, :, :, :2] + noise[:, None, None, :]
                if self.obs_noise_norm:
                    obs_noise_slice = apply_obs_noise_norm(obs_noise_slice)
                obs_noise[:, :, :, :2] = obs_noise_slice
            elif self.noise_shape == (2, 9):
                noise = noise.reshape(obs.shape[:-1]+(-1,)) # 5, 18 -> 5 3 3 2
                # obs_noise[:, :, :, :2] += noise
                obs_noise_slice = obs_noise[:, :, :, :2] + noise
                if self.obs_noise_norm:
                    obs_noise_slice = apply_obs_noise_norm(obs_noise_slice)
                obs_noise[:, :, :, :2] = obs_noise_slice
            elif self.noise_shape == (-1, 9):
                noise = noise.reshape(obs.shape[:-1]+(-1,)) # 5, 9 -> 5 3 3 2
                obs_noise += noise
                if self.obs_noise_norm:
                    obs_noise = apply_obs_noise_norm(obs_noise)

        dist = np.array([[-1 if i==j else self.cal_dist(o, obs[i]) for (j, o) in enumerate(obs)] for i in range(num_agents)])
        order = dist.argsort()
        # observations = np.array([np.vstack((obs[i][None,:], obs[order[i][1:]])) for i in range(num_agents)])
        observations = np.array([np.vstack((obs[i][None,:], obs_noise[order[i][1:]])) for i in range(num_agents)])

        rewards = np.array(list(rewards.values()))  # for CTDE
        # rewards = np.array(list(rewards.values())).sum() # for centralized
        terminations = any(terminations.values())
        truncations = any(truncations.values())
        # assert all([info == list(infos.values())[0] for info in list(infos.values())]), f"{infos} are not all same"
        infos = list(infos.values())[0]

        return observations, rewards, terminations, truncations, infos


# from sisl.pursuit.pursuit.py
# my_parallel_env = my_parallel_wrapper_fn(pursuit_v4.env, seed=SEED)
my_parallel_env_noise = my_parallel_wrapper_fn_noise(pursuit_v4.env)
