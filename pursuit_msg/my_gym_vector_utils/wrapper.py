from typing import Any, Dict, List, Tuple, Union

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
        noise = act[1:6]
        prev_obs = act[1:]
        act = act[0]
        converted_act = []
        for b in np.flip(self.bases):
            converted_act.append(act // b)
            act = act % b
        # ret = np.concatenate((np.array(converted_act)[:, None], noise[:, None], env[:, None]),1)
        return (np.array(converted_act), noise, prev_obs)
        # return np.array(converted_act).transpose()
