# Examples

## PettingZoo
- pursuit_example.py: basic example of visualizing a policy (random now)
- pursuit_manual_policy.py: manual controlling (parallel environment)
    - p to print reward, tab to cycle agent, arrow to control agent, esc to quit, backspace to reset
    - change max_fps and game_time variable to control game time

## Tianshou
- test_dqn.py: demo code that train cartpole with dqn
    - tutorial (code not exactly same) https://tianshou.readthedocs.io/en/master/tutorials/dqn.html
    - github: https://github.com/thu-ml/tianshou/blob/master/test/discrete/test_dqn.py
- test_dqn_pursuit.py: code that train pursuit with dqn
    - the code can compile with small parameter, but have not been tested in a full train
    - the code treat 5 pursurer as 1 agent (centralized execution), action space is action_num^agent_num
    - the code use info returned by environment, currently env info is saved as agent_1's info
