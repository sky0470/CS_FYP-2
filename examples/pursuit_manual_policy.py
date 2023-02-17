import pygame


class ManualPolicy:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]

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
            agent == self.agent
        ), f"Manual Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # set the default action
        action = self.default_action

        # if we get a key, override action using the dict
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # escape to end
                    exit()

                elif event.key == pygame.K_BACKSPACE:
                    # backspace to reset
                    self.env.reset()

                elif event.key == pygame.K_p:
                    print(rwd)

                elif event.key == pygame.K_TAB:
                    self.agent_id = (self.agent_id + 1) % self.env.num_agents
                    self.agent = self.env.agents[self.agent_id]

                elif event.key in self.action_mapping:
                    action = self.action_mapping[event.key]

        return action

    @property
    def available_agents(self):
        return self.env.agent_name_mapping


if __name__ == "__main__":
    from pettingzoo.sisl import pursuit_v4

    clock = pygame.time.Clock()
    max_fps = 5
    game_time = 10

    env = pursuit_v4.parallel_env(
        max_cycles=max_fps * game_time,
        render_mode="human",
        freeze_evaders=True,
        n_evaders=5,
        n_pursuers=3,
        shared_reward=False,
        n_catch=1,
        surround=False,
    )
    obs = env.reset()
    rwd = {}

    manual_policy = ManualPolicy(env)

    while env.agents:
        clock.tick(max_fps)
        actions = {}
        for agent in env.agents:
            if agent == manual_policy.agent:
                actions[agent] = manual_policy(obs, agent, rwd)
            else:
                actions[agent] = 4

        obs, rwd, term, trunc, info = env.step(actions)
    env.close()
