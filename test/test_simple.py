from pettingzoo.mpe import simple_spread_v2

env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode='human')

env.reset()
for _ in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
