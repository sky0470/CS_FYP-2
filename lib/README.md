Customizing our game environment
===
## Using our customized game environemnt (**recommended**)
- you may import 
  - `my_sisl` instead of `pettingzoo.sisl`
  - `my_gym_vector_utils` instead of `gymnasium.vector.utils` 
- or follow the below instruction to edit the library directly

---
## Editing the library directly (**not recommended**)
### Get my site package path
```bash
python -m site
```
## Edit game field
- file to edit: `two_d_maps.py`
  - location: `/path/to/site-packages/pettingzoo/sisl/pursuit/utils/two_d_maps.py`

### Changes to be made
- to remove obstacles (the building in the middle of the map)
  - comment all lines in `retangle_map()` except `rmap = np.zero(...)` in `two_d_maps.py`

## Set deterministic sampled random action 
- file to edit: `spaces.py`
  - location: `/path/to/site-packages/gymnasium/vector/utils/spaces.py`

### Changes to be made
- replace the `_batch_space_discrete(space, n=1)` to the following code snippet:
```python
def _batch_space_discrete(space, n=1, seed=None):
  if space.start == 0:
      return MultiDiscrete(
          np.full((n,), space.n, dtype=space.dtype),
          dtype=space.dtype,
          seed=seed if seed is not None else deepcopy(space.np_random),
      )
  else:
      return Box(
          low=space.start,
          high=space.start + space.n - 1,
          shape=(n,),
          dtype=space.dtype,
          seed=deepcopy(space.np_random),
      )
```
    

## Edit game rule
- file to edit: `pursuit_base.py`
  - location: `/path/to/site-packages/pettingzoo/sisl/pursuit/pursuit_base.py`

### How to use
- repace pursuit_base.py in library with pursuit_base_new.py
  
- pursuit_base_original.py is the original file in the library

### Changed
I change the following behaviors: (see line with ###)
- do not remove prey after it is caught
- originally purs_sur is a bool array that indicate which predator captured prey, now it is a int array that shows how much reward it should get
(I only changed stuff assuming surround=False)
Feature improved: 
- added name tag for agents

### Code explanation
- petting zoo assume env is ACE (agent environment cycle) game, hence step() is called for 1 agent at atime. Game is changed to parallel with wrapper
- in step(), 1 agent moved, then is_last check if that agent is the last among all agents to move, then prey will be removed and move
- reward() calculate the tagging reward for each predator. An predator get tagging reward if there is a prey in its up/down/left/right
- remove_agents() calculate which prey to be removed, and return array purs_sur which indicate which predator removed an agent

## Troubleshoot
- If you encounter error 
  ```
  "/path/to/site-package/tianshou/env/gym_wrappers", line 45, in __init__
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
  ```
  - solution: replace `import gym` with `import gymnasium as gym` in `/path/to/site-package/tianshou/env/gym_wrappers`