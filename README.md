# CS_FYP

## Getting started
1. setup environment
```
pip install -r requirements.txt
```
1. replace `gym` to `gymnasium`
  - replace `import gym` to `import gymnasium as gym` in `/path/to/site-package/tianshou/env/gym_wrappers`
  - replace `import gym` to `import gymnasium as gym` and `from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete` in `/path/to/site-package/tianshou/policy/base.py`


## TODO

### Main Task
- [X] Run predetor prey baseline with baseline library (DQN)
- [ ] Run predetor prey baseline with baseline library (PPO)
    - Test training time
    - Learn petting zoo
    - Learn RL library
- [ ] Add communication (always communicate)
- [ ] Add special reward

### Other
- [ ] Rewrite our library for pip install to sovle dependency issues
- [ ] Test reproducibility of training code
- [ ] Training code better loggin
    - [ ] Log parameter (env + train)
    - [ ] Save policy object also
    
### Future
- [ ] Logging
- [ ] Hyperparameter tuning
- [ ] Parallel training
- [ ] Another RL library
- [ ] Another RL algorithm
