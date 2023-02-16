"""
proof of concept of how to change the parallel part of the pursuit env
code copied from lib to here have not been modified
this code will (and should) not work after the modification
"""
from pettingzoo.sisl import pursuit_v4

from collections import defaultdict
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

from pettingzoo.utils.env import ParallelEnv

ActionType = Optional[int]
AgentID = str
ActionDict = Dict[AgentID, ActionType]
ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")

AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]


# from utils.conversion.py
def my_parallel_wrapper_fn(env_fn):
    def par_fn(**kwargs):
        env = env_fn(**kwargs)
        env = aec_to_parallel_wrapper(env)
        return env

    return par_fn

# from utils.conversion.py
class aec_to_parallel_wrapper(ParallelEnv):
    def __init__(self, aec_env):
        assert aec_env.metadata.get("is_parallelizable", False), (
            "Converting from an AEC environment to a parallel environment "
            "with the to_parallel wrapper is not generally safe "
            "(the AEC environment should only update once at the end "
            "of each cycle). If you have confirmed that your AEC environment "
            "can be converted in this way, then please set the `is_parallelizable` "
            "key in your metadata to True"
        )

        self.aec_env = aec_env

        try:
            self.possible_agents = aec_env.possible_agents
        except AttributeError:
            pass

        self.metadata = aec_env.metadata

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = self.aec_env.state_space
        except AttributeError:
            pass

    @property
    def observation_spaces(self):
        warnings.warn(
            "The `observation_spaces` dictionary is deprecated. Use the `observation_space` function instead."
        )
        try:
            return {
                agent: self.observation_space(agent) for agent in self.possible_agents
            }
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            ) from e

    @property
    def action_spaces(self):
        warnings.warn(
            "The `action_spaces` dictionary is deprecated. Use the `action_space` function instead."
        )
        try:
            return {agent: self.action_space(agent) for agent in self.possible_agents}
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            ) from e

    def observation_space(self, agent):
        return self.aec_env.observation_space(agent)

    def action_space(self, agent):
        return self.aec_env.action_space(agent)

    @property
    def unwrapped(self):
        return self.aec_env.unwrapped

    def reset(self, seed=None, return_info=False, options=None):
        self.aec_env.reset(seed=seed, return_info=return_info, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.observe(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }

        if not return_info:
            return observations
        else:
            infos = dict(**self.aec_env.infos)
            return observations, infos

    def step(self, actions):
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
            agent: self.aec_env.observe(agent) for agent in self.aec_env.agents
        }
        while self.aec_env.agents and (
            self.aec_env.terminations[self.aec_env.agent_selection]
            or self.aec_env.truncations[self.aec_env.agent_selection]
        ):
            self.aec_env.step(None)

        self.agents = self.aec_env.agents
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.aec_env.render()

    def state(self):
        return self.aec_env.state()

    def close(self):
        return self.aec_env.close()


# from sisl.pursuit.pursuit.py
my_parallel_env = my_parallel_wrapper_fn(pursuit_v4.env)

import pygame
if __name__ == "__main__":
    clock = pygame.time.Clock()
    env = my_parallel_env(shared_reward=False, n_evaders=3, n_pursuers=8, render_mode="human")

    env.reset()
    for _ in range(10*5):
        clock.tick(10)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
