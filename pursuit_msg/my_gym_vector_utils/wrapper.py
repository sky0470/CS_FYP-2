from typing import Any, Dict, List, Tuple, Union, Sequence

import gymnasium as gym
import numpy as np
from packaging import version

class MultiDiscreteToDiscreteMsg(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        nvec = env.action_space.nvec
        assert nvec.ndim == 1
        self.bases = np.ones_like(nvec)
        for i in range(1, len(self.bases)):
            self.bases[i] = self.bases[i - 1] * nvec[-i]
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: np.ndarray) -> np.ndarray:
        # noise = act[1:6]
        prev_obs = act[1:]
        act = act[0]
        converted_act = []
        for b in np.flip(self.bases):
            converted_act.append(act // b)
            act = act % b
        # ret = np.concatenate((np.array(converted_act)[:, None], noise[:, None], env[:, None]),1)
        return (np.array(converted_act), prev_obs)
        # return np.array(converted_act).transpose()



class MultiDiscreteToDiscreteNoise(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, env: gym.Env, 
                 has_noise: bool = False,
                 noise_shape: Sequence[int] = 0) -> None:
                #  num_noise_type: int = 0, 
                #  num_noise_per_type: int = 0, 
                #  num_noise_per_agent: int = 0) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        nvec = env.action_space.nvec # 1D vec with len=num_agents, value=num_actions
        assert nvec.ndim == 1
        self.num_agents = len(nvec)
        self.num_actions = nvec[0]
        self.bases_arr = self.num_actions ** np.arange(self.num_agents - 1, -1, -1)

        self.has_noise = has_noise
        self.noise_shape = noise_shape
        self.num_noise_per_agent = abs(np.prod(noise_shape))
        # self.num_noise_type = num_noise_type
        # self.num_noise_per_type = num_noise_per_type
        # self.num_noise_per_agent = num_noise_per_agent
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: np.ndarray) -> np.ndarray:
        noise = act[1:1 + self.num_agents * self.num_noise_per_agent]
        prev_obs = act[1 + self.num_agents * self.num_noise_per_agent:]

        def action(act: np.ndarray) -> np.ndarray:
            # decode int to n-base representation
            converted_act = []
            for b in self.bases_arr:
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()
        
        converted_act = action(act[0])

        return (converted_act, noise, prev_obs)
