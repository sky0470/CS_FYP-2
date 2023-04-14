from typing import Any, Dict, List, Tuple, Union, Sequence

import gymnasium as gym
import numpy as np
from packaging import version

class MultiDiscreteToDiscreteMsg(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, 
                 env: gym.Env,
                 num_actions=5,
                 ) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        nvec = env.action_space.nvec # 1D vec with len=num_agents, value=num_actions
        assert nvec.ndim == 1
        self.num_agents = len(nvec)
        self.num_actions = num_actions
        self.bases_arr = self.num_actions ** np.arange(self.num_agents - 1, -1, -1)
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: np.ndarray) -> np.ndarray:
        prev_obs = act[1:]

        def action(act: np.ndarray) -> np.ndarray:
            # decode int to n-base representation
            converted_act = []
            for b in self.bases_arr:
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()
        
        converted_act = action(act[0])

        return (converted_act, prev_obs)



class MultiDiscreteToDiscreteNoise(gym.ActionWrapper):
    """Gym environment wrapper to take discrete action in multidiscrete environment.

    :param gym.Env env: gym environment with multidiscrete action space.
    """

    def __init__(self, env: gym.Env, 
                 num_actions = 5,
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
        self.num_actions = num_actions,
        self.bases_arr = self.num_actions ** np.arange(self.num_agents - 1, -1, -1)

        self.has_noise = has_noise
        self.noise_shape = noise_shape
        self.num_noise_per_agent = abs(np.prod(noise_shape))
        self.num_norm = abs(noise_shape[0])
        self.action_space = gym.spaces.Discrete(np.prod(nvec))

    def action(self, act: np.ndarray) -> np.ndarray:
        idx = [1, self.num_agents * self.num_noise_per_agent, self.num_norm * 2 * self.num_agents]
        split = np.split(act, np.cumsum(idx))
        noise = split[1]
        act_noise = split[2]
        prev_obs = split[3]
        # noise = act[1:1 + self.num_agents * self.num_noise_per_agent]
        # prev_obs = act[1 + self.num_agents * self.num_noise_per_agent:]

        def action(act: np.ndarray) -> np.ndarray:
            # decode int to n-base representation
            converted_act = []
            for b in self.bases_arr:
                converted_act.append(act // b)
                act = act % b
            return np.array(converted_act, dtype=int).transpose()
        
        converted_act = action(act[0])

        return (converted_act, noise, act_noise, prev_obs)
