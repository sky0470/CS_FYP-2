import pygame
import numpy as np


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        if True:
            self.default_action = 4  # 0
            self.action_mapping = dict()
            self.action_mapping[pygame.K_UP] = 3  # right
            self.action_mapping[pygame.K_DOWN] = 2  # down
            self.action_mapping[pygame.K_LEFT] = 0  # left
            self.action_mapping[pygame.K_RIGHT] = 1  # right

    def __call__(self, observation, agent, rwd):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent_id
        ), f"Manual Policy only applied to agent: {self.agent_id}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        # if we get a key, override action using the dict
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    # escape to end
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # backspace to reset
                    self.env.reset()

                elif event.key == pygame.K_p:
                    print(f"rwd: {rwd}")

                elif event.key == pygame.K_TAB:
                    self.agent_id = (self.agent_id + 1) % self.env.num_agents

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping


if __name__ == "__main__":
    # from pettingzoo.sisl import pursuit_v4
    # from pursuit_msg.pursuit import my_parallel_env as my_env
    from pursuit_msg.pursuit import my_parallel_env_grid_loc as my_env

    clock = pygame.time.Clock()
    max_fps = 5
    game_time = 40

    task_parameter = dict(
        shared_reward=False,
        surround=False,
        freeze_evaders=True,

        x_size=8, # 10 
        y_size=6, #10
        obs_range=3,
        max_cycles=max_fps * game_time, # 40

        n_evaders=1, # 2
        n_pursuers=5, # 5

        catch_reward=0.5,
        urgency_reward=-0.05,
        n_catch=1,
        tag_reward=0,
        render_mode="human", # dont have this in default training param
        catch_reward_ratio=[0, 1, 2, 0.7, 0.2, -0.5]
    )

    env = my_env(**task_parameter)
    obs = env.reset()
    rwd = {}

    manual_policy = ManualPolicy(env)
    num_agents = env.num_agents
    num_actions = 5

    while env.agents:
        clock.tick(max_fps)
        actions = []
        for idx in range(env.num_agents):
            if idx == manual_policy.agent_id:
                actions.append(manual_policy(obs, idx, rwd))
            else:
                actions.append(4)

        bases = 5 ** (num_agents - 1)
        # actions = np.array(actions)
        actions_ = 0
        for i in range(num_agents):
            actions_ = actions_ + bases * actions[i]
            bases = bases // num_actions

        obs, rwd, term, trunc, info = env.step(actions_)
        print(obs.shape)
        print(obs[0,:,:,0], end='\n\n')
        print(obs[0,:,:,1], end='\n\n')
        print(obs[0,:,:,2], end='\n\n')
        print('-'*20)
        # tmp = input()
    env.close()
