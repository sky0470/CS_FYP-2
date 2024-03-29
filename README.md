# CS_FYP

## Prerequisites
Our code are only tested under the following setting, correctness is not guaranteed if other settings are used
- python `3.10` (`3.10.9`)
- conda `23.1.0`
- linux `ubuntu 18.04.5 LTS`
- GPU
    - NVIDIA GeForce RTX 3090 
    - Quadro P620 

## Getting started
1. setup conda environment (optional)
    ```
    conda create --name pursuit python=3.10
    ```
1. setup environment
    ```
    pip install .
    ```
1. replace `gym` with `gymnasium` 
    - replace `import gym` with `import gymnasium as gym` in `/path/to/site-package/tianshou/env/gym_wrappers`
    - replace `import gym` with `import gymnasium as gym` and `from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete` in `/path/to/site-package/tianshou/policy/base.py`
1. execute `test/test-package.py` to see if install successful
    ```
    python test/test-package.py
    ```


## Repo structure
**Note**: directly run code under this `pursuit_msg` is not recommended, use import with the installed package!!
```
.
├── examples
├── experiments
├── lib
│   ├── policy
│   └── sisl
├── log
│   └── pursuit_v4
├── pursuit_msg
│   ├── envs
│   ├── my_gym_vector_utils
│   ├── my_sisl
│   └── policy
├── result
│   ├── dqn_ctde
│   ├── ppo
│   └── ppo_lstm
├── src
└── test
```
- `examples/`: env/training examples from *pettingzoo* and *tianshou*
- `experiments/`: experiment codes
- `lib/`: env/policy codes copied from *pettingzoo* and *tianshou* **for reference only** (edit is not allowed)
- `pursuit_msg/`: package of the our library
    - `envs/`: pursuit game environments
    - `my_gym_vector_utils/`: edited library from `Gymnasium/gymnasium/vector/utils`   
    - `my_sisl`: edited libraray from `Pettingzoo/pettingzoo/sisl` 
    - `policy`: edited policies
- `result/`: storage for (useful) training result
- `src/`: training source code
- `test/`: code for setup testing

## TODO
### Main Task
- [x] Run predetor prey baseline with baseline library (DQN)
- [ ] Run predetor prey baseline with baseline library (PPO)
    - Test training time
    - Learn petting zoo
    - Learn RL library
- [ ] Add communication (always communicate)
- [ ] Add special reward

### Other
- [x] Rewrite our library for pip install to solve dependency issues
- [ ] Test reproducibility of training code
- [ ] Training code better loggin
    - [ ] Log parameter (env + train)
    - [ ] Save policy object also
    
### Future
- [x] Logging
- [ ] Hyperparameter tuning
- [ ] Parallel training
- [ ] Another RL library
- [ ] Another RL algorithm

## Troubleshoot
- If you encounter error 
  ```
  "/path/to/site-package/tianshou/env/gym_wrappers", line 45, in __init__
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
  ```
  - solution: replace `import gym` with `import gymnasium as gym` in `/path/to/site-package/tianshou/env/gym_wrappers`