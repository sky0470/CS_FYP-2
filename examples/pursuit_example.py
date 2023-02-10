from pettingzoo.sisl import pursuit_v4
import pygame

# defualts env: 
# pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=30, n_pursuers=8,obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01, catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)


if __name__ == "__main__":
    clock = pygame.time.Clock()
    env = pursuit_v4.parallel_env(shared_reward=False, n_evaders=3, n_pursuers=8, render_mode="human")

    env.reset()
    for _ in range(10*5):
        clock.tick(10)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()
